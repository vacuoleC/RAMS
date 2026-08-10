# -*- coding: utf-8 -*-
"""B1 增量预测（残差学习）——探索实验脚本

方向：模型预测变化量 Δ = conc_{t+h} - conc_t（分位数 p10/p50/p90），
评估时预测浓度 = conc_t + Δ_pred。持久化基线就是 Δ=0（M5 实证 τ=1 自相关 r=+0.63，
持久化已猜中水平 80%），模型学会"修正"即可赢。

设计要点：
  - 目标从绝对浓度改为增量（消除强自回归主导）
  - 滚动窗口评估（训练 2 年、测试 3 个月、滚动推进；不用固定 70/15/15）
  - 指标：RMSE（还原原始浓度单位）+ 每视界 CRPS（量化分位积分近似）
  - 对照：持久化（Δ=0）、气候学（训练段季节月均值）
  - 表层 0.5m 浓度（tensor_builder 简化口径），特征全剖面水温 + 气象 + 表层浓度
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # 项目根

from rams.models.rams_net import QUANTILES, SharedGRU  # 复用冻结骨干（GRU）
from rams.data.tensor_builder import METEO_COLS, STRAT_COLS, TensorBuilder  # 复用宽表透视

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- 1. 数据加载（复用冻结 tensor_builder 的宽表逻辑） ----------------

def load_wide(parquet_path: str) -> pd.DataFrame:
    """复用 TensorBuilder._load_wide：长表 → 3h 宽表（temp_* + conc_* + meteo + strat）。"""
    builder = TensorBuilder()
    return builder._load_wide(Path(parquet_path))


def surface_conc_cols(wide: pd.DataFrame) -> list[str]:
    """全部 conc_ 深度列，按深度升序（第 0 个即表层 0.5m）。"""
    return sorted([c for c in wide.columns if c.startswith("conc_")],
                  key=lambda c: float(c.split("_")[1]))


# ---------------- 2. 增量目标 + 窗口张量 ----------------

def make_increment_tensors(wide: pd.DataFrame, T: int, H: int,
                           use_meteo: bool = True, use_strat_feat: bool = False):
    """构建 (X, Δ) 张量。Δ 用原始浓度尺度（不归一化）。

    - 特征列：temp_*（全剖面）+ meteo + 可选 strat + 表层 conc_t（强先验）
    - 目标：Δ_h = conc_{t+h} - conc_t，其中 t = 宽表行 i+T-1（窗口末时刻）
    Returns: (X float32, delta float64, t_idx 样本起点宽表行号, feat_cols)
    """
    temp_cols = sorted([c for c in wide.columns if c.startswith("temp_")],
                       key=lambda c: float(c.split("_")[1]))
    conc_cols = surface_conc_cols(wide)
    surface = conc_cols[0]
    feat_cols = temp_cols
    if use_meteo:
        feat_cols = feat_cols + [c for c in METEO_COLS if c in wide.columns]
    if use_strat_feat:
        feat_cols = feat_cols + [c for c in STRAT_COLS if c in wide.columns]
    if surface not in feat_cols:
        feat_cols = feat_cols + [surface]
    feat_cols = [c for c in feat_cols if c in wide.columns]

    X_all = wide[feat_cols].values.astype(np.float32)
    y_all = wide[surface].values.astype(np.float64)
    n = len(wide)
    m = n - T - H
    i = np.arange(m)
    # 窗口（as_strided 视窗 + copy）
    s0, s1 = X_all.strides
    X = np.lib.stride_tricks.as_strided(
        X_all, shape=(m, T, len(feat_cols)), strides=(s0, s0, s1)).copy()
    base = y_all[i + T - 1]                                    # conc_t（窗口末行）
    # 列 j → 视界 h=j+1：Δ_h = conc_{t+h} - conc_t，目标行 = i+T+h-1 = i+T+j
    target = np.stack([y_all[i + T + j] for j in range(H)], axis=1)
    delta = target - base[:, None]
    return X, delta, i, feat_cols


# ---------------- 3. 模型（复用冻结 GRU 骨干 + 分位数头） ----------------

class IncrementQuantileNet(nn.Module):
    """GRU 骨干 + 单头分位数回归（Δ 的 p10/p50/p90），输出 (B, 3H)。"""

    def __init__(self, feat_dim: int, horizon: int, hidden: int = 64,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden, n_layers, dropout)
        self.horizon = horizon
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))  # (B, 3H): [p10, p50, p90]


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, qs=QUANTILES) -> torch.Tensor:
    """分位数损失（合并全部视界）。pred: (B,3H), target: (B,H)。"""
    H = target.shape[1]
    e = target.unsqueeze(1) - pred.reshape(-1, 3, H)
    losses = [torch.mean(torch.maximum(q * e[:, i], (q - 1) * e[:, i]))
              for i, q in enumerate(qs)]
    return torch.stack(losses).mean()


# ---------------- 4. 指标 ----------------

def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def crps_quantile(obs: np.ndarray, qs: np.ndarray, levels: np.ndarray) -> float:
    """CRPS 的量化分位积分近似：CRPS(F,y) = 2/K Σ_j ρ_{a_j}(y - q_j)。

    其中 ρ_a(u)=(a - 1{u<0})u 为分位损失。点预测（全部 q_j=y_hat）退化精确等于 MAE。
    obs: (N,), qs: (N,K), levels: (K,) ⊂ (0,1)。
    """
    u = obs[:, None] - qs                      # (N,K)
    rho = (levels[None, :] - (u < 0).astype(np.float64)) * u
    return float((2.0 / len(levels) * rho.sum(axis=1)).mean())


# ---------------- 5. 训练 ----------------

def train_increment(model: nn.Module, X_tr, delta_tr, X_va, delta_va,
                    epochs: int, lr: float, batch_size: int = 256, seed: int = 0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X_tr)
    Xt = torch.tensor(X_tr, device=DEVICE)
    dt = torch.tensor(delta_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_va, device=DEVICE)
    dv = torch.tensor(delta_va, dtype=torch.float32, device=DEVICE)
    H_ = delta_tr.shape[1]
    last_loss = 0.0
    val_rmse = float("nan")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        tot = 0.0
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = quantile_loss(model(Xt[idx]), dt[idx])
            loss.backward()
            opt.step()
            tot += loss.item()
        last_loss = tot / max(1, n // batch_size)
        model.eval()
        with torch.no_grad():
            pv = model(Xv).reshape(-1, 3, H_)
            val_rmse = torch.sqrt(((pv[:, 1, :] - dv) ** 2).mean()).item()
    return last_loss, val_rmse


# ---------------- 6. 滚动评估 ----------------

def rolling_evaluate(wide: pd.DataFrame, T: int, Hmax: int, horizons: list[int],
                     train_years: int, test_months: int, stride_months: int,
                     epochs: int, lr: float, hidden: int, seed: int,
                     use_meteo: bool, use_strat_feat: bool, verbose: bool):
    """滚动窗口：每窗训练 train_years 年、测试 test_months 月，步进 stride_months 月。

    样本归属用起点宽表行号 i：
      - 训练样本：窗口末时刻 i+T-1 < 训练段末尾（防泄漏，窗口整体落在训练期）
      - 测试样本：窗口起点 i ≥ 测试段首行，窗口整体落在测试期
    指标在全部 fold 上逐视界取平均。
    """
    surf = surface_conc_cols(wide)[0]
    y_all = wide[surf].values.astype(np.float64)
    times = pd.DatetimeIndex(wide.index)
    idx = np.arange(len(wide))
    t_end = times[-1]

    X_all, delta_all, t_idx, feat_cols = make_increment_tensors(
        wide, T, Hmax, use_meteo=use_meteo, use_strat_feat=use_strat_feat)

    cur = times[0]
    fold = 0
    acc = {h: {"rmse_m": [], "rmse_p": [], "rmse_c": [],
               "crps_m": [], "crps_p": [], "crps_c": []} for h in horizons}
    while True:
        t_train_end = cur + pd.DateOffset(years=train_years)
        t_test_end = t_train_end + pd.DateOffset(months=test_months)
        if t_test_end > t_end:
            break
        tr_rows = idx[(times >= cur) & (times < t_train_end)]
        te_rows = idx[(times >= t_train_end) & (times < t_test_end)]
        if len(tr_rows) < T + Hmax + 20 or len(te_rows) < 20:
            break
        # 训练样本：窗口末 i+T-1 落在训练期末尾之前
        tr_i = np.where((t_idx >= tr_rows[0]) & (t_idx + T - 1 <= tr_rows[-1] - 1))[0]
        # 测试样本：窗口整体落在测试期
        te_i = np.where((t_idx >= tr_rows[-1]) & (t_idx + T - 1 <= te_rows[-1] - 1))[0]
        if len(tr_i) < 300 or len(te_i) < 20:
            break

        # 用训练段统计量归一化特征（防泄漏）
        sub = wide.iloc[tr_rows]
        stats = {c: (float(sub[c].mean()), float(sub[c].std()) + 1e-8) for c in wide.columns}
        wide_norm = wide.copy()
        for c, (mu, sd) in stats.items():
            wide_norm[c] = (wide_norm[c] - mu) / sd
        X_norm, _, _, _ = make_increment_tensors(
            wide_norm, T, Hmax, use_meteo=use_meteo, use_strat_feat=use_strat_feat)

        # 训练/验证（验证取训练样本尾部，不碰测试期）
        tr_i_sorted = tr_i
        va_i = tr_i_sorted[-min(500, len(tr_i_sorted)):]
        X_va, delta_va = X_norm[va_i], delta_all[va_i]

        # 气候学基线：训练段月均值（季节，防泄漏）
        mth_tr = times[tr_rows].month.values
        month_mean = {}
        for mth in range(1, 13):
            mask = mth_tr == mth
            month_mean[mth] = float(y_all[tr_rows][mask].mean()) if mask.sum() > 0 \
                else float(y_all[tr_rows].mean())

        if verbose:
            print(f"  fold{fold} @{cur.date()} train={len(tr_i)} test={len(te_i)}", flush=True)
        model = IncrementQuantileNet(feat_dim=X_norm.shape[2], horizon=Hmax,
                                     hidden=hidden).to(DEVICE)
        train_increment(model, X_norm[tr_i_sorted], delta_all[tr_i_sorted],
                        X_va, delta_va, epochs=epochs, lr=lr, seed=seed)

        model.eval()
        with torch.no_grad():
            pred_q = model(torch.tensor(X_norm[te_i], device=DEVICE))
            pred_q = pred_q.reshape(-1, 3, Hmax).cpu().numpy()  # (N,3,Hmax)
        for h in horizons:
            i_rows = t_idx[te_i]
            base = y_all[i_rows + T - 1]                       # conc_t
            obs = y_all[i_rows + T + h - 1]                    # conc_{t+h}
            pred_conc = base + pred_q[:, 1, h - 1]
            pred_p10 = base + pred_q[:, 0, h - 1]
            pred_p90 = base + pred_q[:, 2, h - 1]
            pred_persist = base
            obs_time = times[i_rows + T + h - 1]               # 目标时刻 t+h 的月份（气候学用）
            pred_clim = np.array([month_mean[m] for m in obs_time.month.values])

            qs_model = np.stack([pred_p10, pred_conc, pred_p90], axis=1)  # (N,3)
            acc[h]["rmse_m"].append(rmse(obs, pred_conc))
            acc[h]["rmse_p"].append(rmse(obs, pred_persist))
            acc[h]["rmse_c"].append(rmse(obs, pred_clim))
            acc[h]["crps_m"].append(crps_quantile(obs, qs_model, np.array([0.1, 0.5, 0.9])))
            acc[h]["crps_p"].append(crps_quantile(obs, np.full((len(obs), 3), base[:, None]), np.array([0.1, 0.5, 0.9])))
            acc[h]["crps_c"].append(crps_quantile(obs, np.repeat(pred_clim[:, None], 3, axis=1), np.array([0.1, 0.5, 0.9])))

        cur = cur + pd.DateOffset(months=stride_months)
        fold += 1

    summary = {}
    for h in horizons:
        a = acc[h]
        summary[h] = {
            "model_rmse": float(np.mean(a["rmse_m"])),
            "persist_rmse": float(np.mean(a["rmse_p"])),
            "clim_rmse": float(np.mean(a["rmse_c"])),
            "model_crps": float(np.mean(a["crps_m"])),
            "persist_crps": float(np.mean(a["crps_p"])),
            "clim_crps": float(np.mean(a["crps_c"])),
            "folds": fold,
        }
    return summary, fold


# ---------------- main ----------------

def run(parquet: str, T: int = 24, Hmax: int = 8, train_years: int = 2,
        test_months: int = 3, stride_months: int = 1, epochs: int = 30,
        lr: float = 1e-3, hidden: int = 64, seed: int = 0,
        use_meteo: bool = True, use_strat_feat: bool = False,
        tag: str = "default", verbose: bool = True):
    t0 = time.time()
    wide = load_wide(parquet)
    if verbose:
        print(f"[{tag}] wide={wide.shape} 时间 {wide.index[0]} ~ {wide.index[-1]}", flush=True)
    horizons = list(range(1, Hmax + 1))
    summary, folds = rolling_evaluate(
        wide, T, Hmax, horizons, train_years=train_years, test_months=test_months,
        stride_months=stride_months, epochs=epochs, lr=lr, hidden=hidden, seed=seed,
        use_meteo=use_meteo, use_strat_feat=use_strat_feat, verbose=verbose)
    dt = time.time() - t0

    print(f"\n[{tag}] 结果（folds={folds}, {dt:.0f}s）")
    print(f"{'h':>3} {'模型RMSE':>9} {'持久化':>9} {'气候学':>9} | {'模型CRPS':>9} {'持久化':>9} {'气候学':>9}")
    for h in horizons:
        s = summary[h]
        print(f"{h:>3} {s['model_rmse']:>9.4f} {s['persist_rmse']:>9.4f} {s['clim_rmse']:>9.4f}"
              f" | {s['model_crps']:>9.4f} {s['persist_crps']:>9.4f} {s['clim_crps']:>9.4f}")
    print("\n超越持久化判定（RMSE）：")
    for h in horizons:
        s = summary[h]
        delta = s["model_rmse"] - s["persist_rmse"]
        mark = "模型√" if delta < 0 else "持久化√"
        print(f"  h={h}: 模型 {s['model_rmse']:.4f} vs 持久化 {s['persist_rmse']:.4f}"
              f"（Δ={delta:+.4f}）→ {mark}   CRPS 模型 {s['model_crps']:.4f} vs 持久化 {s['persist_crps']:.4f}")
    return summary, folds, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="D:/coding/26AIendwork/data/processed/standard.parquet")
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--train-years", type=int, default=2)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--stride-months", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-meteo", action="store_true")
    ap.add_argument("--strat", action="store_true")
    ap.add_argument("--tag", default="default")
    args = ap.parse_args()

    run(args.parquet, T=args.T, Hmax=args.H, train_years=args.train_years,
        test_months=args.test_months, stride_months=args.stride_months,
        epochs=args.epochs, lr=args.lr, hidden=args.hidden, seed=args.seed,
        use_meteo=not args.no_meteo, use_strat_feat=args.strat, tag=args.tag)


if __name__ == "__main__":
    main()
