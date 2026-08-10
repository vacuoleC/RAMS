# -*- coding: utf-8 -*-
"""J 方向探索：中长程慢变量特征（慢变量喂增量，3-seed）

背景（已证实）：
  - 当前输入是 T=24（3 天）窗口 + conc_t，只有短期记忆
  - M5：藻类 ACF 衰减需 75-90 天（强季节性/长记忆）
  - 短板识别：特征只有短期窗口，没有"中长程季节状态"——方向判别弱（B_direction AUC≈0.59）
    与区间校准的窗口间漂移（H：std 0.09）可能都源于此

假设：给增量模型加"慢变量特征"（过去 30/60/90 天的均值/趋势/方差，只对 conc_t 与
关键气象 air_temp 计算），显式编码"当前季节状态"，可能同时：
  (a) 方向判别：季节状态 → 当前处于上升/下降季，帮 P(Δ>0) 判别（短板 2）
  (b) 区间校准：季节状态 → 模型不确定性跟状态走，降覆盖窗口间 std（短板 1）
  (c) 点精度：CRPS/RMSE

协议（复用 A/B/C/B7/H：训练 730d / 测试 90d / 步长 45d，17 窗口）：
  - 2 变体 × 3 seed（0/1/2），每窗口独立训练
    base = 现输入（全剖面水温 + 气象 + conc_t，与 A/B7 完全同口径）         —— 基线
    slow = base + 慢变量通道（conc 均值/趋势/方差 × {30,60,90}d + air_temp 均值 × {30,60,90}d）
  - 模型：RamsNet（GRU 骨干 + p10/p50/p90 分位数头 + M2/M4 多任务，w=1/3/2，与 B7/A 一致）
  - 目标：abs_delta = conc_{t+h} − conc_t，窗口训练段拟合 scale 归一化（防泄漏）

防泄漏：
  - 慢变量只用 [t−W, t) 的**过去**数据计算（shift(1) 后 rolling，因果），W ∈ {30,60,90} 天
  - 慢变量在窗口**训练段**拟合均值/方差标准化（与既有特征一致）
  - 评估段慢变量取值只依赖测试段之前的信息（窗口起点有 W 天预热缓冲，NaN 用历史可得值前向填充）

评估（全部还原 conc 单位，同一测试段）：
  a. 每视界 CRPS / p50 RMSE（分位数闭合形式，与 T4/B2/B7/A 一致）
  b. 区间覆盖率 [p10,p90]（重点看窗口间 std：短板 1）
  c. 方向判别：P(Δ>0)=Φ(q50/σ_gauss) 的 ROC-AUC + p50 符号命中率（短板 2，B_direction 口径）
  d. 慢变量对以上三个短板的净效果 → 值得加吗？

保密：只输出聚合统计量，不打印原始数据行。
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch  # noqa: E402

from scipy.stats import norm  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402
from rams.models.rams_net import RamsNet  # noqa: E402
from rams.training.trainer import Trainer  # noqa: E402

T, H = 24, 8                 # 回看 3 天 / 预测 24h（8×3h 步）
EPOCHS = 30
SEEDS = [0, 1, 2]

# ---- 滚动窗口参数（天，3h 网格：1 天 = 8 个时刻）----
TRAIN_DAYS = 730             # 每窗口用 2 年训练
TEST_DAYS = 90               # 每窗口测试后 3 个月
STRIDE_DAYS = 45             # 每 45 天推进一个窗口
GRID_PER_DAY = 8

# 多任务权重（与冻结 Trainer/rams 口径一致：w=1/3/2）
W_M1, W_M2, W_M4 = 1.0, 3.0, 2.0

# 慢变量配置
SLOW_WINDOW_DAYS = [30, 60, 90]            # 滚动窗口长度（天）
SLOW_BASE_COLS = ["conc_0.5", "air_temp"]   # 对 conc 与关键气象算慢变量
EPS = 1e-8


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 T4/B2/B7/A 一致实现）。"""
    q10 = np.asarray(q10, dtype=np.float64)
    q50 = np.asarray(q50, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    qs = np.sort(np.stack([q10, q50, q90], axis=-1), axis=-1)
    q10, q50, q90 = qs[..., 0], qs[..., 1], qs[..., 2]
    qk = np.stack([
        q10 - (q50 - q10) / 4.0,
        q10, q50, q90,
        q90 + (q90 - q50) / 4.0,
    ], axis=-1)
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


def load_wide(parquet):
    loader = TensorBuilder(TensorConfig(T=T, H=H))
    wide = loader._load_wide(Path(parquet))
    return wide.sort_index()


def _rolling_slope(vals):
    """窗口内线性斜率（x = 时刻序号 0..n-1）。"""
    if len(vals) < 2:
        return float("nan")
    x = np.arange(len(vals), dtype=np.float64)
    y = vals.astype(np.float64)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom < 1e-12:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def add_slow_features(wide: pd.DataFrame) -> pd.DataFrame:
    """对 conc_0.5 / air_temp 计算过去 {30,60,90} 天的滚动均值/斜率/方差。

    防泄漏：只用 [t−W, t) 的**过去**值（shift(1) 后 rolling），因果；方差 ddof=1。
    斜率：窗口内线性回归斜率，除以窗口天数做量纲无关（≈每 24h 的平均变化）。
    列名：slow_{col}_mean_{W}d / slow_{col}_slope_{W}d / (conc 仅) slow_conc_std_{W}d
    """
    out = wide.copy()
    for col in SLOW_BASE_COLS:
        s = out[col]
        for W in SLOW_WINDOW_DAYS:
            win = W * GRID_PER_DAY
            past = s.shift(1)  # 只用 t 之前的值（防泄漏）
            out[f"slow_{col}_mean_{W}d"] = past.rolling(win, min_periods=win // 2).mean()
            out[f"slow_{col}_slope_{W}d"] = past.rolling(win, min_periods=win // 2) \
                .apply(_rolling_slope, raw=True) / float(W)
            if col == "conc_0.5":
                out[f"slow_conc_std_{W}d"] = past.rolling(win, min_periods=win // 2).std(ddof=1)
    # 头部预热期 NaN：ffill（用更早历史），剩余头部块用首个有效值的常数扩展填
    # （数据从 2021-03 开始，头部 30-90 天没有更早历史；常数扩展不引入逐行未来信息，
    #   且该块只落在第 1 窗口训练段内，防泄漏可辩护）
    out = out.ffill()
    for col in [c for c in out.columns if c.startswith("slow_")]:
        first_valid = out[col].dropna()
        if len(first_valid) > 0:
            out[col] = out[col].fillna(float(first_valid.iloc[0]))
    return out


def build_window(wide, i0, i1, base_feat_cols, slow_feat_cols):
    """返回窗口 [i0,i1) 的 base / base+slow 标准化特征、原始浓度、abs_delta 目标。

    Returns:
      Xw_base  (n_w, T, F_base)   基线特征窗口（训练段统计标准化）
      Xw_slow  (n_w, T, F_slow)   基线 + 慢变量特征窗口
      cur_raw  (n_w,) conc_t 原始尺度
      y_abs    (n_w, H) conc_{t+h} 原始尺度
      strat_w, warn_w  M2/M4 标签（复用 B2/B7/A）
    """
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))

    def _norm(cols):
        Xtr = df[cols].values[:n_tr].astype(np.float32)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + EPS
        X = ((df[cols].values.astype(np.float32) - mu) / sd).astype(np.float32)
        return X

    X_base = _norm(base_feat_cols)
    X_slow = _norm(base_feat_cols + slow_feat_cols)

    y_raw = df["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    Xw_base = np.stack([X_base[i:i + T] for i in range(n_w)]).astype(np.float32)
    Xw_slow = np.stack([X_slow[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    # M2 分层标签（B2/B7/A 复用）
    delta = df["delta_T"].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（B2/B7/A 复用）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw_base, Xw_slow, cur_raw, y_abs, strat_w, warn_w


def make_targets_abs_delta(cur_raw, y_abs):
    """abs_delta 目标（A/B7 最优口径）：Δ = conc_{t+h} - conc_t。"""
    return y_abs - cur_raw[:, None]


def train_model(Xw, yw, strat_w, warn_w, n_tr, epochs, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RamsNet(feat_dim=Xw.shape[2], horizon=H, use_m4=True)
    trainer = Trainer(model, device=device)
    trainer.fit(Xw[:n_tr], yw[:n_tr], strat_w[:n_tr],
                Xw[n_tr:], yw[n_tr:], strat_w[n_tr:],
                warn_tr=warn_w[:n_tr], warn_va=warn_w[n_tr:],
                epochs=epochs, batch_size=128)
    return model


def predict_quantiles(model, X_te, device):
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def back_to_conc_abs(cur_te, q_norm, scale):
    """abs_delta 还原：conc = cur + q*scale。q_norm (N,3,H) → (N,3,H) conc 单位。"""
    return cur_te[:, None, None] + q_norm * scale


def p_gauss(q10, q50, q90):
    """P(Δ>0) = Φ(q50/σ_gauss)，σ_gauss=(q90−q10)/2.563（与 B_direction 完全同口径）。"""
    sigma = (q90 - q10) / 2.563
    with np.errstate(all="ignore"):
        z = np.where(sigma > 1e-9, q50 / np.maximum(sigma, 1e-12),
                     np.where(q50 > 0, 5.0, np.where(q50 < 0, -5.0, 0.0)))
        p = norm.cdf(z)
    return np.clip(p, 1e-6, 1 - 1e-6)


def roc_auc_safe(p, y_positive):
    """两类都出现才可算 AUC，否则 nan。"""
    if p.min() == p.max():
        return float("nan")
    if y_positive.min() == y_positive.max():
        return float("nan")
    try:
        return float(roc_auc_score(y_positive, p))
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description="J 方向：慢变量特征（3-seed 对照）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 1 seed × 2 epoch")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out-json", default="exp/model_enhancement/j_slow_vars/results.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    print(f"== J 方向：中长程慢变量特征（{len(seeds)} seed）==", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d；"
          f"GRU 骨干 + p10/p50/p90 + M2/M4 多任务；目标 {H}h", flush=True)
    print(f"   慢变量: {SLOW_BASE_COLS} 的过去 {SLOW_WINDOW_DAYS}d 滚动均值/斜率"
          f"（conc 另加 std），只用过去数据防泄漏", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    base_feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    base_feat_cols = [c for c in base_feat_cols if c in wide.columns]
    if "conc_0.5" not in base_feat_cols:
        base_feat_cols = base_feat_cols + ["conc_0.5"]
    print(f"   基线特征列 {len(base_feat_cols)} 个（temp_* + 气象 + conc_t）", flush=True)

    wide = add_slow_features(wide)
    slow_feat_cols = [c for c in wide.columns if c.startswith("slow_")]
    print(f"   慢变量特征列 {len(slow_feat_cols)} 个: {slow_feat_cols}", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
        seeds = seeds[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个 × {len(seeds)} seed", flush=True)

    Nw = len(windows)
    arms = ["base", "slow"]
    agg = {a: {s: {
        "crps": np.zeros(Nw), "rmse": np.zeros(Nw), "cover": np.zeros(Nw),
        "dir_acc": np.zeros(Nw), "auc": np.zeros(Nw),
        "crps_h": np.zeros((Nw, H)), "rmse_h": np.zeros((Nw, H)),
        "auc_h": np.zeros((Nw, H)),
    } for s in seeds} for a in arms}

    # 慢变量诊断（全数据集，聚合统计，无原始数据行）
    delta_all = wide["conc_0.5"].diff().values
    conc_all = wide["conc_0.5"].values.astype(np.float64)
    diag_rows = []
    for col in slow_feat_cols:
        s = wide[col].values.astype(np.float64)
        m1 = np.isfinite(s) & np.isfinite(conc_all)
        m2 = np.isfinite(s) & np.isfinite(delta_all)
        corr_t = float(np.corrcoef(s[m1], conc_all[m1])[0, 1]) if m1.sum() > 2 else float("nan")
        corr_d = float(np.corrcoef(s[m2], delta_all[m2])[0, 1]) if m2.sum() > 2 else float("nan")
        diag_rows.append({
            "feature": col,
            "corr_conc_t": round(corr_t, 3),
            "corr_delta": round(corr_d, 3),
        })
    print("\n[slow] 慢变量与 conc_t / Δconc 的相关系数（诊断，非模型输入）:", flush=True)
    print(pd.DataFrame(diag_rows).to_string(index=False), flush=True)

    persist_crps_h = np.zeros((Nw, H))
    persist_rmse_h = np.zeros((Nw, H))

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        Xw_base, Xw_slow, cur_raw, y_abs, strat_w, warn_w = build_window(
            wide, i0, i1, base_feat_cols, slow_feat_cols)
        n_win = len(Xw_base)
        n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
        te_sl = slice(n_tr, n_win)

        cur_te = cur_raw[te_sl]
        y_te = y_abs[te_sl]                     # (N,H) 原始 conc 观测
        Nte = len(y_te)

        # abs_delta 目标 + 尺度（窗口训练段拟合，防泄漏）
        raw = make_targets_abs_delta(cur_raw, y_abs)
        scale = float(np.std(raw[:n_tr])) + EPS
        y_norm = (raw / scale).astype(np.float32)

        # 持久化基线（Δ≡0 → conc_{t+h}=conc_t）
        q_p = np.repeat(cur_te[:, None, None], H, axis=2).astype(np.float64)
        crps_p_h = np.array([np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 0, h],
                                                    q_p[:, 0, h], y_te[:, h]))
                             for h in range(H)])
        persist_crps_h[wi] = crps_p_h
        persist_rmse_h[wi] = np.array([np.sqrt(np.mean((q_p[:, 0, h] - y_te[:, h]) ** 2))
                                       for h in range(H)])

        # 方向标签：up = Δ>0（Δ=0 记作"不上"；与 B_direction 完全同口径）
        up = (y_te - cur_te[:, None]) > 0       # (N,H) bool

        for a in arms:
            Xw = Xw_base if a == "base" else Xw_slow
            Xte = Xw[te_sl]
            for s in seeds:
                model = train_model(Xw, y_norm, strat_w, warn_w, n_tr,
                                    args.epochs, args.device, s)
                q_norm = predict_quantiles(model, Xte, args.device)
                q_conc = back_to_conc_abs(cur_te, q_norm, scale)   # (N,3,H) conc 单位

                # Δ 增量单位分位数（与 B_direction 完全同口径）：q_delta = q_norm*scale
                q_delta = q_norm * scale

                obs = y_te
                crps_h = np.array([np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                          q_conc[:, 2, h], obs[:, h]))
                                   for h in range(H)])
                rmse_h = np.array([np.sqrt(np.mean((q_conc[:, 1, h] - obs[:, h]) ** 2))
                                   for h in range(H)])
                cover = float(np.mean((obs >= q_conc[:, 0]) & (obs <= q_conc[:, 2])))

                # 方向命中：p50 增量符号 vs 真实增量符号（conc 单位，B7 口径）
                pred_delta = q_conc[:, 1] - cur_te[:, None]
                true_delta = obs - cur_te[:, None]
                dir_acc = float(np.mean(np.sign(pred_delta) == np.sign(true_delta)))

                # 方向 AUC：P(Δ>0)=Φ(q50/σ) 用**Δ 增量单位**分位数（B_direction 口径）
                P_up = p_gauss(q_delta[:, 0], q_delta[:, 1], q_delta[:, 2])  # (N,H)
                auc_h = np.array([roc_auc_safe(P_up[:, h], up[:, h]) for h in range(H)])

                g = agg[a][s]
                g["crps_h"][wi] = crps_h
                g["rmse_h"][wi] = rmse_h
                g["crps"][wi] = float(crps_h.mean())
                g["rmse"][wi] = float(np.sqrt(np.mean(rmse_h ** 2)))
                g["cover"][wi] = cover
                g["dir_acc"][wi] = dir_acc
                g["auc"][wi] = float(np.nanmean(auc_h))
                g["auc_h"][wi] = auc_h

        # 窗口摘要（seed 均值）
        for a in arms:
            c = np.mean([agg[a][s]["crps"][wi] for s in seeds])
            r = np.mean([agg[a][s]["rmse"][wi] for s in seeds])
            cv = np.mean([agg[a][s]["cover"][wi] for s in seeds])
            ac = np.mean([agg[a][s]["auc"][wi] for s in seeds])
            print(f"    {a:<5} CRPS={c:.4f}  RMSE={r:.3f}  覆盖={cv:.3f}  "
                  f"AUC={ac:.3f}  n_test={Nte}", flush=True)

    # ---- 聚合（seed×窗口 展平，均值±std）----
    def _scalar(arm, key):
        vals = [agg[arm][s][key][wi] for s in seeds for wi in range(Nw)]
        return float(np.mean(vals)), float(np.std(vals))

    cp_mean = float(persist_crps_h.mean())
    rp_mean = float(np.sqrt(np.mean(persist_rmse_h ** 2)))

    print("\n===== 平均对照（seed×窗口 展平，均值±std）=====", flush=True)
    print(f"  {'arm':<6}{'CRPS':<22}{'p50RMSE':<22}{'覆盖':<20}{'覆盖std':<10}"
          f"{'方向命中':<10}{'方向AUC':<10}", flush=True)
    summary = {}
    for a in arms:
        c_m, c_s = _scalar(a, "crps")
        r_m, r_s = _scalar(a, "rmse")
        cv_m, cv_s = _scalar(a, "cover")
        da_m, _ = _scalar(a, "dir_acc")
        au_m, _ = _scalar(a, "auc")
        skill = (cp_mean - c_m) / cp_mean * 100
        print(f"  {a:<6}{c_m:<10.4f}±{c_s:<10.4f}{r_m:<10.3f}±{r_s:<10.3f}"
              f"{cv_m:<10.3f}±{cv_s:<8.3f}{cv_s:<10.3f}{da_m:<10.3f}{au_m:<10.3f}",
              flush=True)
        summary[a] = {
            "crps": round(c_m, 4), "crps_std": round(c_s, 4),
            "rmse": round(r_m, 4), "rmse_std": round(r_s, 4),
            "coverage": round(cv_m, 4), "coverage_std": round(cv_s, 4),
            "dir_acc": round(da_m, 4),
            "dir_auc": round(au_m, 4) if not np.isnan(au_m) else None,
            "skill_vs_persist_pct": round(skill, 2),
        }
    print(f"  持久化: CRPS={cp_mean:.4f}  p50RMSE={rp_mean:.3f}", flush=True)

    print("\n===== 逐视界 CRPS（seed×窗口 展平均值）=====", flush=True)
    print(f"  {'h':<4}{'base':<10}{'slow':<10}{'Δ':<10}", flush=True)
    crps_h_out = {}
    for h in range(H):
        b = float(np.mean([agg["base"][s]["crps_h"][wi, h] for s in seeds for wi in range(Nw)]))
        sl = float(np.mean([agg["slow"][s]["crps_h"][wi, h] for s in seeds for wi in range(Nw)]))
        print(f"  {h + 1:<4}{b:<10.4f}{sl:<10.4f}{sl - b:<+10.4f}", flush=True)
        crps_h_out[str(h + 1)] = {"base": round(b, 4), "slow": round(sl, 4)}

    print("\n===== 逐视界方向 AUC（P(Δ>0)=Φ(q50/σ)，Δ>0 vs cur_t，seed×窗口 展平均值）=====",
          flush=True)
    print(f"  {'h':<4}{'base':<10}{'slow':<10}{'Δ':<10}", flush=True)
    auc_h_out = {}
    for h in range(H):
        b = float(np.nanmean([agg["base"][s]["auc_h"][wi, h] for s in seeds for wi in range(Nw)]))
        sl = float(np.nanmean([agg["slow"][s]["auc_h"][wi, h] for s in seeds for wi in range(Nw)]))
        print(f"  {h + 1:<4}{b:<10.4f}{sl:<10.4f}{sl - b:<+10.4f}", flush=True)
        auc_h_out[str(h + 1)] = {"base": round(b, 4), "slow": round(sl, 4)}

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "n_windows": Nw,
                     "seeds": seeds, "w": {"m1": W_M1, "m2": W_M2, "m4": W_M4},
                     "variant": "abs_delta",
                     "slow_windows_days": SLOW_WINDOW_DAYS,
                     "slow_base_cols": SLOW_BASE_COLS,
                     "slow_feat_cols": slow_feat_cols,
                     "slow_note": "只用过去数据计算（shift+rolling 因果），训练段标准化，防泄漏"},
        "arms": summary,
        "persist": {"crps": round(cp_mean, 4), "rmse": round(rp_mean, 4)},
        "crps_h": crps_h_out,
        "auc_h": auc_h_out,
        "slow_diag_corr": diag_rows,
        "per_window": [],
    }
    for wi in range(Nw):
        res["per_window"].append({
            "window": wi + 1,
            "base_crps": round(float(np.mean([agg["base"][s]["crps"][wi] for s in seeds])), 4),
            "slow_crps": round(float(np.mean([agg["slow"][s]["crps"][wi] for s in seeds])), 4),
            "base_rmse": round(float(np.mean([agg["base"][s]["rmse"][wi] for s in seeds])), 4),
            "slow_rmse": round(float(np.mean([agg["slow"][s]["rmse"][wi] for s in seeds])), 4),
            "base_cover": round(float(np.mean([agg["base"][s]["cover"][wi] for s in seeds])), 4),
            "slow_cover": round(float(np.mean([agg["slow"][s]["cover"][wi] for s in seeds])), 4),
            "base_auc": round(float(np.nanmean([agg["base"][s]["auc"][wi] for s in seeds])), 4),
            "slow_auc": round(float(np.nanmean([agg["slow"][s]["auc"][wi] for s in seeds])), 4),
        })
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
