#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""C4 探索：递归预测 vs 直接多步（强自相关序列 M1 表层浓度）

问题：现有模型是**直接多步**（T=24 输入 → 一步输出 H=8 未来 24h）。
M5 证实表层浓度 τ=1 自相关 r≈+0.63。假设：递归预测（每步只预测 1 步，
把预测值喂回输入、循环 8 次）可能更适配强自相关序列——每步都能用
"最新预测"而非"固定 8 步一起出"。

公平对比（唯一差异 = 预测策略，其余完全一致）：
  - 特征通道 D=27 = 20 层水温(temp_*) + 6 气象(meteo) + 1 浓度(conc_0.5)
    （生产 RamsNet 不输入浓度，但 T2 实证发现藻类自回归主导；为了让递归
    能"喂回预测"，两边都必须看到历史浓度 → 公平基线）
  - 结构：GRU(hidden=64) + M1 分位数头(p10/p50/p90)，分位数损失训练
    （去掉 M2/M4 辅助任务，隔离"预测策略"单一变量）
  - 直接多步：GRU 处理 T 窗 → 头一步输出 (B,3,H)
  - 递归：GRU 处理 T 窗 → 头输出 (B,3) 单步 → 把 p50 喂回 conc 通道、
    外生通道置 0（归一化均值，无未来气象预报的保守假设）→ GRUCell 更新
    隐状态 → 循环 H 次。训练用 scheduled sampling（教师强迫概率 0.9→0.1 线性衰减），
    评估始终用预测值喂回（纯递归，暴露误差累积）。

评估协议（与 T4 一致）：滚动窗口 训练 730 天 / 测试 90 天 / 每 45 天推进。
指标：逐视界 RMSE（还原原始浓度单位 µg/L）+ CRPS（p10/p50/p90 闭合解）。
对照：持久化（最后观测浓度，逐视界同值）、气候学（训练段同月均值）。
只输出统计量/RMSE/CRPS，不打印任何原始数据行。

用法（Python 3.13 CPU 可跑，约 15-25 分钟/seed，全窗口）：
  python exp/model_enhancement/c4_recursive/run_c4_recursive.py --smoke        # 2 窗口 × 2 epoch 冒烟
  python exp/model_enhancement/c4_recursive/run_c4_recursive.py                # 全量 1 seed
  python exp/model_enhancement/c4_recursive/run_c4_recursive.py --seeds 2 --max-windows 8
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
# 复用生产 TensorBuilder._load_wide（同一 3h 网格/聚合/dropna/时间轴），保证口径一致
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402

T, H = 24, 8                 # 回看 3 天 / 预测 24h（8×3h）
GRID_PER_DAY = 8             # 3h 网格：1 天 8 个时刻
TRAIN_DAYS = 730             # 每窗口 2 年训练
TEST_DAYS = 90               # 每窗口后 3 个月测试
STRIDE_DAYS = 45             # 每 45 天推进一个窗口
EPOCHS = 30
BS = 128
LR = 1e-3
HIDDEN = 64
QUANTILES = (0.1, 0.5, 0.9)
CONC_CH = "conc_0.5"         # 0.5m 表层总浓度（M1 目标）
SEED = 0


# ----------------------------------------------------------------------------
# 分位数损失 + CRPS（闭合解，与 scripts/explore/t4_crps_eval.py 一致）
# ----------------------------------------------------------------------------
def crps_quantiles(q10, q50, q90, y):
    """p10/p50/p90 分位数预测的 CRPS 闭合解（允许越序修正）。"""
    q10 = np.asarray(q10, dtype=np.float64)
    q50 = np.asarray(q50, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    qs = np.sort(np.stack([q10, q50, q90], axis=-1), axis=-1)
    q10, q50, q90 = qs[..., 0], qs[..., 1], qs[..., 2]
    qk = np.stack([q10 - (q50 - q10) / 4.0, q10, q50, q90,
                   q90 + (q90 - q50) / 4.0], axis=-1)
    ak = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    deg = (q90 - q10) < 1e-9
    total = np.zeros_like(y, dtype=np.float64)
    for k in range(4):
        aL, aR = ak[k], ak[k + 1]
        qL, qR = qk[..., k], qk[..., k + 1]
        slope = (qR - qL) / (aR - aL)
        p1 = np.where(np.abs(slope) < 1e-12, 1.0, slope)
        p0 = qL - p1 * aL
        with np.errstate(all="ignore"):
            astar = (y - p0) / p1
            c = np.clip(astar, aL, aR)
        for u, v in ((aL, c), (c, aR)):
            mid = (u + v) / 2.0
            s = (y <= (p0 + p1 * mid)).astype(np.float64)
            C0 = s * (p0 - y)
            C1 = s * p1 - p0 + y
            total += 2.0 * (C0 * (v - u) + C1 * (v * v - u * u) / 2.0
                            - p1 * (v * v * v - u * u * u) / 3.0)
    out = np.where(deg, np.abs(y - q50), total)
    return np.maximum(out, 0.0)


# ----------------------------------------------------------------------------
# 模型：共享 GRU backbone + M1 分位数头（直接 / 递归两种预测策略）
# ----------------------------------------------------------------------------
class DirectGRU(nn.Module):
    """直接多步：T 窗 → 一步输出 (B, 3*H) 分位数。"""

    def __init__(self, feat_dim: int, horizon: int, hidden: int = HIDDEN):
        super().__init__()
        self.horizon = horizon
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, horizon * 3))

    def forward(self, x):            # (B, T, D)
        out, _ = self.gru(x)
        return self.head(out[:, -1]).reshape(-1, 3, self.horizon)  # (B,3,H)


class RecursiveGRU(nn.Module):
    """递归：T 窗 → 单步 (B,3)，p50 喂回 conc 通道 + GRUCell 更新隐状态 → 循环 H 次。

    外生通道（temp_*/meteo）在递归步置 0（归一化均值）：无未来气象预报的
    保守假设，递归纯由自回归浓度驱动（这正是要检验的假设）。头权重共享。
    训练用 scheduled sampling（teacher forcing 概率逐样本伯努利）。
    """

    def __init__(self, feat_dim: int, horizon: int, conc_idx: int,
                 hidden: int = HIDDEN, teacher_start: float = 0.9,
                 teacher_end: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.conc_idx = conc_idx
        self.gru = nn.GRU(feat_dim, hidden, batch_first=True)
        self.cell = nn.GRUCell(feat_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 3))
        self.teacher_start = teacher_start
        self.teacher_end = teacher_end

    def forward(self, x, y=None, teacher_p: float = 0.0):
        """x: (B,T,D); y: (B,H) 可选（训练 teacher forcing）。返回 (B,3,H)。"""
        B = x.shape[0]
        teacher = y is not None
        out, _ = self.gru(x)
        h = out[:, -1]
        self._last_q = self.head(h)                      # (B,3)
        preds = [self._last_q]
        for hh in range(1, self.horizon):
            use_teacher = torch.rand(B, 1, device=x.device) < teacher_p
            u = torch.zeros(B, self.gru.input_size, device=x.device)
            if teacher:
                pred = self._last_q[:, 1]
                # 喂回上一步真值 y[:, hh-1]（预测步 hh 的输入 = 上一步观测值）
                u[:, self.conc_idx] = torch.where(
                    use_teacher[:, 0], y[:, hh - 1], pred)
            else:
                u[:, self.conc_idx] = self._last_q[:, 1]
            h = self.cell(u, h)
            self._last_q = self.head(h)
            preds.append(self._last_q)
        return torch.stack(preds, dim=2)                 # (B,3,H)


def quantile_loss(q, y, qs=QUANTILES):
    """q (B,3,H), y (B,H) 分位数损失均值。"""
    yq = y.unsqueeze(1)                # (B,1,H)
    e = yq - q                         # (B,3,H)
    losses = [torch.mean(torch.maximum(qs[i] * e[..., i], (qs[i] - 1) * e[..., i]))
              for i in range(3)]
    return torch.stack(losses).mean()


# ----------------------------------------------------------------------------
# 数据：复用 TensorBuilder._load_wide，滚动窗口切分（训练段拟合归一化，防泄漏）
# ----------------------------------------------------------------------------
def load_wide(parquet):
    builder = TensorBuilder(TensorConfig(T=T, H=H))
    wide = builder._load_wide(Path(parquet))
    return wide.sort_index()


def build_window(wide, i0, i1, feat_cols):
    """窗口 [i0,i1)：X (n_w,T,D) 归一化、y (n_w,H) 归一化，返回 y_sd（还原用）。"""
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    tr = df.iloc[:n_tr]
    X = df[feat_cols].values.astype(np.float32)
    for i, c in enumerate(feat_cols):
        mu, sd = float(tr[c].mean()), float(tr[c].std()) + 1e-8
        X[:, i] = (X[:, i] - mu) / sd
    yc = CONC_CH
    y_mu, y_sd = float(tr[yc].mean()), float(tr[yc].std()) + 1e-8
    y = (df[yc].values.astype(np.float32) - y_mu) / y_sd
    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    yw = np.stack([y[i + T:i + T + H] for i in range(n_w)]).astype(np.float32)
    return Xw, yw, y_sd


# ----------------------------------------------------------------------------
# 训练 + 评估
# ----------------------------------------------------------------------------
def train_model(model, X_tr, y_tr, X_va, y_va, epochs, device):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    Xt = torch.tensor(X_tr).to(device)
    yt = torch.tensor(y_tr).to(device)
    Xv = torch.tensor(X_va).to(device)
    yv = torch.tensor(y_va).to(device)
    n = len(Xt)
    rng = np.random.default_rng(0)
    is_rec = isinstance(model, RecursiveGRU)
    for ep in range(epochs):
        model.train()
        if is_rec:
            teacher_p = (model.teacher_start
                         + (model.teacher_end - model.teacher_start) * ep / max(epochs - 1, 1))
        else:
            teacher_p = 0.0
        order = rng.permutation(n)
        for s in range(0, n, BS):
            idx = order[s:s + BS]
            xb, yb = Xt[idx], yt[idx]
            opt.zero_grad()
            q = model(xb, yb, teacher_p) if is_rec else model(xb)
            loss = quantile_loss(q, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            qv = model(Xv) if not is_rec else model(Xv)
            vrmse = float(torch.sqrt(torch.mean((qv[:, 1] - yv) ** 2)))
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"      ep{ep} loss={loss.item():.4f} val_rmse={vrmse:.4f}", flush=True)
    return model


def predict_quantiles(model, X, device):
    model.eval()
    with torch.no_grad():
        q = model(torch.tensor(X).to(device))          # (B,3,H)
    return q.cpu().numpy().astype(np.float64)


def crps_per_horizon(qp, yp):
    """qp (N,3,H), yp (N,H) 归一化 → 每视界 CRPS（归一化尺度）。"""
    return [float(np.mean(crps_quantiles(qp[:, 0, h], qp[:, 1, h], qp[:, 2, h], yp[:, h])))
            for h in range(H)]


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="C4 递归 vs 直接多步")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口（0=全部）")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    print(f"== C4 递归 vs 直接多步（滚动窗口：训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / "
          f"推进 {STRIDE_DAYS}d，epochs={args.epochs}，seed×{args.seeds}，device={args.device}）==", flush=True)

    # [1] 宽表 + 特征列
    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格）", flush=True)
    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    if CONC_CH not in wide.columns:
        raise SystemExit(f"缺少目标列 {CONC_CH}")
    feat_cols = [c for c in feat_cols if c in wide.columns] + [CONC_CH]
    conc_idx = feat_cols.index(CONC_CH)
    print(f"    特征通道 D={len(feat_cols)}（20 层水温 + {len(METEO_COLS)} 气象 + conc），"
          f"conc 通道 idx={conc_idx}", flush=True)

    # [2] 滚动窗口
    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:2]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个", flush=True)

    # [3] 逐窗口 × 逐 seed：训练两个模型，测试段预测
    dir_r = np.zeros((len(windows), args.seeds, H))
    rec_r = np.zeros((len(windows), args.seeds, H))
    dir_c = np.zeros((len(windows), args.seeds, H))
    rec_c = np.zeros((len(windows), args.seeds, H))
    per_r = np.zeros((len(windows), H))
    clm_r = np.zeros((len(windows), H))
    per_mae = np.zeros((len(windows), H))
    clm_mae = np.zeros((len(windows), H))
    meta = []
    for wi, (i0, i1) in enumerate(windows):
        st, en = wide.index[i0], wide.index[i1 - 1]
        print(f"  [3.{wi + 1}] 窗口 {wi + 1}/{len(windows)}  {st:%Y-%m-%d} → {en:%Y-%m-%d}", flush=True)
        Xw, yw, y_sd = build_window(wide, i0, i1, feat_cols)
        n_tr = int(len(Xw) * TRAIN_DAYS / days)
        X_tr, y_tr = Xw[:n_tr], yw[:n_tr]
        X_te, y_te = Xw[n_tr:], yw[n_tr:]     # 测试段 = 窗口后 90 天（含 val 段）

        # 基线（原始尺度）：持久化 = 输入窗末时刻观测浓度；气候学 = 训练段同月均值
        conc_orig = wide[CONC_CH].values[i0:i1].astype(np.float64)
        idx_win = wide.index[i0:i1]
        months = np.array([ts.month for ts in idx_win])
        k_test = np.arange(n_tr, len(Xw))
        x_last = conc_orig[k_test + T - 1]
        y_te_orig = np.array([[conc_orig[k + T + h] for h in range(H)] for k in k_test])
        persist = np.repeat(x_last[:, None], H, axis=1)
        tr_conc = conc_orig[:n_tr + T]
        month_mean = {m: float(tr_conc[months[:n_tr + T] == m].mean())
                      for m in range(1, 13)
                      if (months[:n_tr + T] == m).sum() >= 5}
        fallback = float(tr_conc.mean())
        obs_months = months[k_test + T - 1]
        clim = np.array([[month_mean.get(m, fallback) for h in range(H)] for m in obs_months])
        per_r[wi] = np.sqrt(np.mean((persist - y_te_orig) ** 2, axis=0))
        clm_r[wi] = np.sqrt(np.mean((clim - y_te_orig) ** 2, axis=0))
        per_mae[wi] = np.mean(np.abs(persist - y_te_orig), axis=0)
        clm_mae[wi] = np.mean(np.abs(clim - y_te_orig), axis=0)

        for si in range(args.seeds):
            torch.manual_seed(SEED + si)
            np.random.seed(SEED + si)
            d_model = DirectGRU(feat_dim=len(feat_cols), horizon=H).to(args.device)
            r_model = RecursiveGRU(feat_dim=len(feat_cols), horizon=H, conc_idx=conc_idx).to(args.device)
            print(f"    seed{si} 直接多步...", flush=True)
            train_model(d_model, X_tr, y_tr, X_te, y_te, args.epochs, args.device)
            qd = predict_quantiles(d_model, X_te, args.device)
            print(f"    seed{si} 递归...", flush=True)
            train_model(r_model, X_tr, y_tr, X_te, y_te, args.epochs, args.device)
            qr = predict_quantiles(r_model, X_te, args.device)
            dir_r[wi, si] = np.sqrt(np.mean((qd[:, 1] - y_te) ** 2, axis=0)) * y_sd
            rec_r[wi, si] = np.sqrt(np.mean((qr[:, 1] - y_te) ** 2, axis=0)) * y_sd
            dir_c[wi, si] = np.array(crps_per_horizon(qd, y_te)) * y_sd
            rec_c[wi, si] = np.array(crps_per_horizon(qr, y_te)) * y_sd
        print(f"    直接 RMSE/视界: {np.round(dir_r[wi].mean(0), 3)}  CRPS: {np.round(dir_c[wi].mean(0), 3)}", flush=True)
        print(f"    递归 RMSE/视界: {np.round(rec_r[wi].mean(0), 3)}  CRPS: {np.round(rec_c[wi].mean(0), 3)}", flush=True)
        print(f"    持久化 RMSE/视界: {np.round(per_r[wi], 3)}  气候学 RMSE/视界: {np.round(clm_r[wi], 3)}", flush=True)
        meta.append({"window": wi + 1, "start": str(st), "end": str(en),
                     "y_sd": round(float(y_sd), 3), "train_std": round(float(np.std(tr_conc)), 3)})

    # [4] 聚合
    agg = lambda a: a.mean(axis=(0, 1))    # noqa: E731
    per_r_mean = per_r.mean(axis=0)
    clm_r_mean = clm_r.mean(axis=0)
    per_mae_mean = per_mae.mean(axis=0)
    clm_mae_mean = clm_mae.mean(axis=0)
    print("\n===== 逐视界 RMSE（µg/L，越低越好，跨窗口×seed 均值）=====", flush=True)
    print(f"  {'h':<4}{'直接多步':<12}{'递归':<12}{'持久化':<12}{'气候学':<12}", flush=True)
    for h in range(H):
        print(f"  h={h + 1:<3}{agg(dir_r)[h]:<12.3f}{agg(rec_r)[h]:<12.3f}"
              f"{per_r_mean[h]:<12.3f}{clm_r_mean[h]:<12.3f}", flush=True)
    print("\n===== 逐视界 CRPS（µg/L，越低越好）=====", flush=True)
    print(f"  {'h':<4}{'直接多步':<12}{'递归':<12}{'持久化(MAE)':<16}{'气候学(MAE)':<16}", flush=True)
    for h in range(H):
        print(f"  h={h + 1:<3}{agg(dir_c)[h]:<12.3f}{agg(rec_c)[h]:<12.3f}"
              f"{per_mae_mean[h]:<16.3f}{clm_mae_mean[h]:<16.3f}", flush=True)

    # [5] 结论
    print("\n===== 结论 =====", flush=True)
    for h in [0, 3, 7]:
        print(f"  h={h + 1}: 直接 RMSE={agg(dir_r)[h]:.3f}  递归 RMSE={agg(rec_r)[h]:.3f}  "
              f"Δ(递-直)={agg(rec_r)[h] - agg(dir_r)[h]:+.3f}  "
              f"直接 CRPS={agg(dir_c)[h]:.3f}  递归 CRPS={agg(rec_c)[h]:.3f}", flush=True)
    print(f"  RMSE 误差累积信号 Δ(h8-h1) 直接={agg(dir_r)[H - 1] - agg(dir_r)[0]:+.3f}  "
          f"递归={agg(rec_r)[H - 1] - agg(rec_r)[0]:+.3f}", flush=True)

    out = Path("exp/model_enhancement/c4_recursive/results.json")
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "seeds": args.seeds,
                     "n_windows": len(windows), "feat_dim": len(feat_cols),
                     "note": "conc 通道=表层浓度；递归外生通道置 0；持久化=输入窗末观测"},
        "rmse_per_horizon": {"direct": agg(dir_r).tolist(), "recursive": agg(rec_r).tolist(),
                             "persistence": per_r_mean.tolist(), "climatology": clm_r_mean.tolist()},
        "crps_per_horizon": {"direct": agg(dir_c).tolist(), "recursive": agg(rec_c).tolist(),
                             "persistence": per_mae_mean.tolist(), "climatology": clm_mae_mean.tolist()},
        "by_window": meta,
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[result] 统计量已写入 {out}", flush=True)
    print(f"运行耗时 {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
