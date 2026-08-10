# -*- coding: utf-8 -*-
"""H 探索：区间覆盖率的分布漂移修复（seasonal / weighted / larger cal / season-reg）

背景：F 共形校准证实——增量分位数模型的覆盖率**均值准**（raw α=0.2 精确 0.8003）但
**窗口间漂移大**（std 0.090，逐窗口 0.66~0.94）。根因是 30d 校准段与 90d 测试段
**季节分布漂移**（corr(cal_cov, test_cov)=0.339，30d 代表不了后 90d）。本探索对比
多种"漂移修复"方法，目标：把覆盖率窗口 std 从 0.090 降到 <0.05。

协议沿用 F（训练 730d / 校准段 / 测试 90d / 步长 45d）。**关键设计**：所有变体共用
F 的 17 个测试窗口（同一批 90d 测试段），校准段在测试段前向外延伸（cal=30d 时与 F
逐项复现）——保证覆盖率 std 在"同一批窗口"上可比。方法（全部基于 CQR 共形，α=0.2
目标 80%）：
  1. raw            : 不校准（B2/B7/F 基线，预期 std≈0.090）
  2. cal30_cqr      : F 的 split conformal CQR（最近 30d 校准）——参照点，应复现 F
  3. seas_cal       : 季节分段校准——校准样本用"过去所有年份的同月/同季节"（测试月±1），
                      让校准段与测试段季节对齐（含训练段内样本 → in-sample 偏置，如实标注）
  4. time_weight    : 时间加权共形——校准样本按"与测试段季节距离"软加权（高斯核 σ=2 月），
                      池=测试前全部数据（含 30d 校准块 out-of-sample 锚点）
  5. seas_reg       : 残差季节回归 + 共形——先对模型残差的季节成分（月份因子）回归，
                      消除季节后再做共形校准（区间随测试月残差水平自适应）
  6. cal90_cqr / cal365_cqr : 加大校准段 30d→90d→365d（holdout，覆盖全年季节）
  7. seas_holdout   : 诊断——把"前一年同季节 90d 段"从训练中挖出做**诚实 out-of-sample**
                      季节校准（量化 seas_cal 的 in-sample 偏置）

共形空间：归一化 Δ（affine 可逆，conc 单位保证等价）；CQR（Romano 2019）下/上分开
校准；有限样本校正 Q=⌈(n+1)(1-α)⌉（加权版用加权分位数）。

评估（还原 conc 单位，α=0.2 目标 80%）：
  覆盖率均值 / **覆盖率窗口 std（核心，目标<0.05）** / 区间宽 vs raw / CRPS vs raw。
诚实记录：逐窗口哪些被修、哪些修不了（个别极端季节窗口）。

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

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402
from rams.models.rams_net import RamsNet  # noqa: E402
from rams.training.trainer import Trainer  # noqa: E402

T, H = 24, 8
EPOCHS = 30
SEED = 0

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）
TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
GRID_PER_DAY = 8

# F 协议：训练 730d / 校准 30d / 测试 90d → 窗口总长 850d（17 窗口）
F_CAL_DAYS = 30

ALPHA = 0.2          # 目标不覆盖率（1-α = 0.80 覆盖率）
SEASON_WIDTH = 1     # 季节校准窗口：测试月 ±1
WEIGHT_SIGMA = 2.0   # 时间加权高斯核 σ（月）

# 校准段变体（天）——方法 4
CAL_VARIANTS = [30, 90, 365]

# 季节回归平滑核（残差月因子，3 个月窗）
MONTH_KERNEL = np.array([0.25, 0.5, 0.25])


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 T4/B2/B7/F 一致实现）。"""
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


def finite_q(level, n_cal):
    """有限样本校正的阶统计量索引（1-indexed）：⌈(n+1)(1-α)⌉。"""
    k = int(np.ceil((n_cal + 1) * level))
    return max(1, min(k, n_cal))


def month_dist(a, b):
    """环上月份距离（12 个月圆环，0~6 月）。"""
    d = abs(int(a) - int(b)) % 12
    return min(d, 12 - d)


def season_mask(months, center, width=SEASON_WIDTH):
    return np.array([month_dist(m, center) <= width for m in months], dtype=bool)


def season_weight(months, center, sigma=WEIGHT_SIGMA):
    d = np.array([month_dist(m, center) for m in months], dtype=np.float64)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


def weighted_quantile(scores, weights, level):
    """加权分位数：排序后取累积权重达到 level·总权重的第一个 score。"""
    order = np.argsort(scores)
    ss = scores[order]
    ww = weights[order]
    cw = np.cumsum(ww)
    idx = np.searchsorted(cw, level * cw[-1])
    idx = int(min(idx, len(ss) - 1))
    return ss[idx]


def load_wide(parquet):
    loader = TensorBuilder(TensorConfig(T=T, H=H))
    wide = loader._load_wide(Path(parquet))
    return wide.sort_index()


def build_segments(wide, train_start_row, cal_start_row, test_start_row, df_end_row,
                   feat_cols, cal_rows, strat_col="delta_T"):
    """从绝对行边界构建窗口样本（train / cal / test 三段）。

    df 覆盖 [train_start_row, df_end_row)。样本 i 的"当前行" = row0 + i + T - 1。
    段归属按当前行绝对位置：train / cal / test。
    cal=30d 时 df_end 取 F 原布局（=测试段尾），n_test=712 与 F 逐项复现。
    Returns:
      Xw        (n_w, T, F) 标准化特征窗口（训练段统计）
      cur_raw   (n_w,) conc_t 原始尺度
      y_abs     (n_w, H) conc_{t+h} 原始尺度
      seg       (n_w,) 0=训练 / 1=校准 / 2=测试
      strat_w, warn_w  M2/M4 标签
      ts        (n_w,) 样本当前时刻时间戳
      delta_s   (n_w,) 分层温差
      n_tr_samples  训练样本数（cal 段切片起点，与 F 一致 = 训练行数 - T）
    """
    row0 = train_start_row
    df = wide.iloc[row0:df_end_row]
    n = len(df)
    n_tr_rows = cal_start_row - train_start_row

    Xtr = df[feat_cols].values[:n_tr_rows].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((df[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    y_raw = df["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)
    ts = np.array([df.index[i + T - 1] for i in range(n_w)])

    # 段标记：当前绝对行 = row0 + i + T - 1
    seg = np.zeros(n_w, dtype=np.int64)
    cur_abs = row0 + np.arange(n_w) + T - 1
    seg[cur_abs >= test_start_row] = 2
    seg[(cur_abs >= cal_start_row) & (cur_abs < test_start_row)] = 1

    # M2 分层标签（训练段阈值，防泄漏）——当前行 = i+T-1（与 F 一致）
    delta = df[strat_col].values
    thr = float(np.median(delta[:n_tr_rows]))
    delta_cur = np.array([delta[i + T - 1] for i in range(n_w)], dtype=np.float64)
    strat_w = np.array(delta_cur > thr, dtype=np.int64)
    delta_s = delta_cur

    # M4 预警标签（训练段峰值分位数，防泄漏）——与 F 一致用 n_tr_rows（=训练行数）
    warn_val = y_abs.max(axis=1)
    qs = np.quantile(warn_val[:n_tr_rows], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    n_tr_samples = n_tr_rows - T
    return Xw, cur_raw, y_abs, seg, strat_w, warn_w, ts, delta_s, n_tr_samples


def aux_labels(y_abs_sub, delta_sub):
    """对给定子集（行对齐）重算 M2/M4 标签阈值（防挖出后阈值泄漏）。"""
    thr = float(np.median(delta_sub))
    strat_sub = np.array(delta_sub > thr, dtype=np.int64)
    warn_val = y_abs_sub.max(axis=1)
    qs = np.quantile(warn_val, [0.75, 0.90, 0.97])
    warn_sub = np.searchsorted(qs, warn_val).astype(np.int64)
    return strat_sub, warn_sub


def train_model(Xw, yw, strat_w, warn_w, n_tr_samples, epochs, device,
                tr_idx=None, va_idx=None):
    """训练 RamsNet。默认：训练 = 前 n_tr_samples（F 语义），验证 = 其后（含校准/测试段）。
    tr_idx/va_idx 显式给定时（holdout），用给定索引。"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = RamsNet(feat_dim=Xw.shape[2], horizon=H, use_m4=True)
    trainer = Trainer(model, device=device)
    if tr_idx is None:
        tr_idx = np.arange(n_tr_samples)
        va_idx = np.arange(n_tr_samples, len(Xw))
    else:
        tr_idx = np.asarray(tr_idx)
        va_idx = np.asarray(va_idx) if va_idx is not None else np.arange(n_tr_samples, len(Xw))
    trainer.fit(Xw[tr_idx], yw[tr_idx], strat_w[tr_idx],
                Xw[va_idx], yw[va_idx], strat_w[va_idx],
                warn_tr=warn_w[tr_idx], warn_va=warn_w[va_idx],
                epochs=epochs, batch_size=128)
    return model


def predict_quantiles(model, X, device):
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def calibrate_cqr(q_cal, y_cal, alpha):
    """CQR（下/上分开校准）未加权：返回每视界 (Q_low, Q_up)。"""
    q10, q90 = q_cal[:, 0], q_cal[:, 2]
    sc_low = np.maximum(q10 - y_cal, 0.0)
    sc_up = np.maximum(y_cal - q90, 0.0)
    n_cal = len(y_cal)
    k = finite_q(1 - alpha, n_cal)
    Q_low = np.array([np.sort(sc_low[:, h])[k - 1] for h in range(H)])
    Q_up = np.array([np.sort(sc_up[:, h])[k - 1] for h in range(H)])
    return Q_low, Q_up


def calibrate_cqr_weighted(q_pool, y_pool, w_pool, alpha):
    """CQR 加权版：按权重做加权分位数（软季节加权共形）。"""
    q10, q90 = q_pool[:, 0], q_pool[:, 2]
    sc_low = np.maximum(q10 - y_pool, 0.0)
    sc_up = np.maximum(y_pool - q90, 0.0)
    Q_low = np.array([weighted_quantile(sc_low[:, h], w_pool, 1 - alpha) for h in range(H)])
    Q_up = np.array([weighted_quantile(sc_up[:, h], w_pool, 1 - alpha) for h in range(H)])
    return Q_low, Q_up


def smooth_circular(a, kernel=MONTH_KERNEL):
    """12 维圆环平滑（3 个月窗）：out[m]=0.25·a[m-1]+0.5·a[m]+0.25·a[m+1]（环上）。"""
    padded = np.concatenate([a[-2:], a, a[:2]])          # 16 元素
    conv = np.convolve(padded, kernel, mode="valid")      # 14 元素
    return conv[1:13]                                     # 取中间 12 个


def calibrate_cqr_seasonal(q_pool, y_pool, pool_months, te_month, alpha):
    """残差季节回归 + 共形：按月份因子消除季节后再做共形校准（方法 5）。

    对每视界每侧：pool 样本按月份分组的残差均值（圆环平滑）作"季节水平"，
    去季节残差 (sc - month_mean) 的 1-α 分位数为季节无关校准量，
    测试区间的校准量 = 季节水平(测试月) + 去季节残差分位数。
    Returns: (Q_low, Q_up) 逐视界（H,），已含测试月季节水平。
    """
    q10, q90 = q_pool[:, 0], q_pool[:, 2]
    sc_low = np.maximum(q10 - y_pool, 0.0)
    sc_up = np.maximum(y_pool - q90, 0.0)
    months = np.asarray(pool_months, dtype=int)
    Q_low = np.zeros(H)
    Q_up = np.zeros(H)
    for h in range(H):
        for side, sc, out in (("low", sc_low, Q_low), ("up", sc_up, Q_up)):
            mmean = np.zeros(12)
            for m in range(1, 13):
                sel = months == m
                mmean[m - 1] = float(np.mean(sc[sel, h])) if sel.sum() > 0 else float(np.nan)
            nan_m = np.isnan(mmean)
            mmean[nan_m] = float(np.mean(sc[:, h]))      # 无样本月份回退全局均值
            mmean = smooth_circular(mmean)
            deseason = sc[:, h] - mmean[months - 1]
            k = finite_q(1 - alpha, len(sc))
            Q_adj = np.sort(deseason)[k - 1]
            out[h] = mmean[te_month - 1] + Q_adj
    return Q_low, Q_up


def adjust_cqr(q, Q_low, Q_up):
    """CQR 调整：保 p50；强制 q10≤p50≤q90 排序。返回 (q10, q50, q90)。"""
    q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
    q10a = np.minimum(q10 - Q_low[None, :], q50)
    q90a = np.maximum(q90 + Q_up[None, :], q50)
    return q10a, q50, q90a


def eval_method(q_te_norm, cur_te, obs, scale, Qs=None):
    """给定归一化 Δ 测试分位数与可选校准量，评估覆盖/宽/CRPS（还原 conc 单位）。

    Qs=None → raw；否则 (Q_low, Q_up) 用于 CQR。
    Returns: cov(float), cov_h(H,), width_h(H,), crps(float)
    """
    if Qs is not None:
        q10a, q50, q90a = adjust_cqr(q_te_norm, Qs[0], Qs[1])
    else:
        q10a, q50, q90a = q_te_norm[:, 0], q_te_norm[:, 1], q_te_norm[:, 2]
    q10c = cur_te[:, None] + q10a * scale
    q90c = cur_te[:, None] + q90a * scale
    q50c = cur_te[:, None] + q50 * scale
    cov_h = np.mean((obs >= q10c) & (obs <= q90c), axis=0)   # (H,)
    width_h = np.mean(q90c - q10c, axis=0)                    # (H,) conc 单位
    crps_h = np.array([float(np.mean(crps_quantiles(
        q10c[:, h], q50c[:, h], q90c[:, h], obs[:, h]))) for h in range(H)])
    return float(np.mean(cov_h)), cov_h, width_h, float(np.mean(crps_h))


def main():
    ap = argparse.ArgumentParser(description="H 区间覆盖率分布漂移修复探索")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="每变体最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch × 全部变体")
    ap.add_argument("--variants", default=",".join(map(str, CAL_VARIANTS)))
    ap.add_argument("--with-holdout", action="store_true", help="跑 seas_holdout（每窗口第 2 次训练）")
    ap.add_argument("--out-json", default="exp/model_enhancement/h_drift_fix/results.json")
    args = ap.parse_args()

    variants = [int(v.strip()) for v in args.variants.split(",") if v.strip()]
    t0 = time.time()
    print("== H 区间覆盖率分布漂移修复（seasonal / weighted / larger cal / season-reg）==", flush=True)
    print(f"   协议: 训练{TRAIN_DAYS}d / 校准段变体 {variants}d / 测试{TEST_DAYS}d / 步长{STRIDE_DAYS}d；"
          f"GRU 骨干 + p10/p50/p90 分位数头 + M2/M4 多任务；目标 Δ={H}h；α={ALPHA}", flush=True)
    print(f"   季节窗口: 测试月±{SEASON_WIDTH}；加权高斯核 σ={WEIGHT_SIGMA}月；"
          f"holdout={args.with_holdout}", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    if "conc_0.5" not in feat_cols:
        feat_cols = feat_cols + ["conc_0.5"]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象 + conc_t）", flush=True)

    # ---- F 的 17 个测试窗口（校准30d 布局：训练730d + 校准30d + 测试90d，步长45d）----
    f_days = TRAIN_DAYS + F_CAL_DAYS + TEST_DAYS
    test_windows = []   # (test_start_row, test_end_row)
    for i0 in range(0, n - f_days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        test_start = i0 + (TRAIN_DAYS + F_CAL_DAYS) * GRID_PER_DAY
        test_windows.append((test_start, test_start + TEST_DAYS * GRID_PER_DAY))
    if args.smoke:
        test_windows = test_windows[:1]
    elif args.max_windows:
        test_windows = test_windows[:args.max_windows]
    Nw = len(test_windows)
    print(f"[2] 共用 F 的 {Nw} 个测试窗口（每个 90d，步长45d）——所有变体在同一批窗口上评估", flush=True)

    agg_all = {}
    window_rows_all = []

    for X in variants:
        cal_rows = X * GRID_PER_DAY
        methods = ["raw", f"cal{X}_cqr"]
        if X == F_CAL_DAYS:
            methods += ["seas_cal", "time_weight", "seas_reg"]
            if args.with_holdout:
                methods += ["seas_holdout"]
        print(f"\n[2] 变体 校准{X}d：方法 {methods}", flush=True)

        agg = {m: {"cov": np.full(Nw, np.nan), "cov_h": np.full((Nw, H), np.nan),
                   "width": np.full(Nw, np.nan), "width_h": np.full((Nw, H), np.nan),
                   "crps": np.full(Nw, np.nan)} for m in methods}
        n_valid = 0

        for wi, (test_start, test_end) in enumerate(test_windows):
            cal_start = test_start - cal_rows
            train_start = max(0, cal_start - TRAIN_DAYS * GRID_PER_DAY)
            # 训练段不足 730d（数据起点不够）→ 跳过该窗口
            if cal_start - train_start < TRAIN_DAYS * GRID_PER_DAY:
                print(f"\n  [3.{X}.{wi + 1}] 窗口 {wi + 1}/{Nw}  训练段不足（需 730d），跳过", flush=True)
                continue
            # df 覆盖 [train_start, test_end)（与 F 原布局一致）：cal=30d 时逐项复现 F；
            # cal>30d 时训练段前移（校准段在测试段前向外延伸），测试段不变
            df_end = test_end
            st = wide.index[train_start]
            en = wide.index[test_end - 1]
            print(f"\n  [3.{X}.{wi + 1}] 窗口 {wi + 1}/{Nw}  训练{st:%Y-%m-%d} → 测试{en:%Y-%m-%d}  "
                  f"({(en - st).days} 天)", flush=True)

            Xw, cur_raw, y_abs, seg, strat_w, warn_w, ts, delta_s, n_tr_samples = build_segments(
                wide, train_start, cal_start, test_start, df_end, feat_cols, cal_rows)
            n_cal = cal_rows                              # 校准样本数
            n_te = int((seg == 2).sum())
            cal_sl = slice(n_tr_samples, n_tr_samples + n_cal)
            te_sl = slice(n_tr_samples + n_cal, None)

            # Δ 目标（归一化）：scale 用训练段 Δ 的 std（防泄漏）
            delta_raw = y_abs - cur_raw[:, None]
            scale = float(np.std(delta_raw[:n_tr_samples])) + 1e-8
            y_norm = (delta_raw / scale).astype(np.float32)

            model = train_model(Xw, y_norm, strat_w, warn_w, n_tr_samples,
                                args.epochs, args.device)

            q_te = predict_quantiles(model, Xw[te_sl], args.device)      # (n_te,3,H)
            cur_te = cur_raw[te_sl]
            obs = y_abs[te_sl]
            n_test = len(obs)
            te_dt = pd.DatetimeIndex(ts[te_sl])
            te_center = int(np.median(te_dt.month))

            q_cal = predict_quantiles(model, Xw[cal_sl], args.device)
            y_cal_norm = y_norm[cal_sl]

            # 季节池：测试前、月份在测试月±SEASON_WIDTH 的样本
            before_test = np.arange(n_tr_samples + n_cal)   # 样本索引 < 测试起始
            pool_ts = pd.DatetimeIndex(ts[before_test])
            pool_months = np.array(pool_ts.month, dtype=int)
            season_ok = season_mask(pool_months, te_center, SEASON_WIDTH)
            pool_idx = before_test[season_ok]
            n_pool_is = int((pool_idx < n_tr_samples).sum())
            n_pool_oos = int(len(pool_idx) - n_pool_is)

            row = {"variant": X, "window": wi + 1, "start": str(st), "end": str(en),
                   "test_center_month": int(te_center), "n_test": n_test,
                   "n_train_samples": int(n_tr_samples),
                   "n_pool_season": int(len(pool_idx)),
                   "n_pool_season_is": n_pool_is, "n_pool_season_oos": n_pool_oos,
                   "cal_cov_raw": round(float(np.mean(
                       (y_cal_norm >= q_cal[:, 0]) & (y_cal_norm <= q_cal[:, 2]))), 3),
                   "sd_inc": round(scale, 3)}

            # ---- 各方法校准量（归一化 Δ 空间）----
            Qs = {}
            Qs[f"cal{X}_cqr"] = calibrate_cqr(q_cal, y_cal_norm, ALPHA)
            if X == F_CAL_DAYS:
                if len(pool_idx) > 0:
                    q_pool = predict_quantiles(model, Xw[pool_idx], args.device)
                    y_pool_norm = y_norm[pool_idx]
                    Qs["seas_cal"] = calibrate_cqr(q_pool, y_pool_norm, ALPHA)
                    # 时间加权：池=测试前全部样本，按季节距离加权（软替代硬切分）
                    w_pool = season_weight(pool_months, te_center, WEIGHT_SIGMA)
                    q_pool_all = predict_quantiles(model, Xw[before_test], args.device)
                    y_pool_all = y_norm[before_test]
                    Qs["time_weight"] = calibrate_cqr_weighted(
                        q_pool_all, y_pool_all, w_pool, ALPHA)
                    # 残差季节回归 + 共形（方法 5）：池=测试前全部样本，月因子
                    Qs["seas_reg"] = calibrate_cqr_seasonal(
                        q_pool_all, y_pool_all, pool_months, te_center, ALPHA)
                else:
                    Qs["seas_cal"] = (np.zeros(H), np.zeros(H))
                    Qs["time_weight"] = (np.zeros(H), np.zeros(H))
                    Qs["seas_reg"] = (np.zeros(H), np.zeros(H))

                # seas_holdout：挖出前一年同季节块做诚实 out-of-sample 校准
                if args.with_holdout:
                    test_year = int(te_dt.year[0])
                    ts_train = pd.DatetimeIndex(ts[:n_tr_samples])
                    hold_ok = (season_mask(np.array(ts_train.month, dtype=int), te_center,
                                           SEASON_WIDTH)
                               & (np.array(ts_train.year, dtype=int) == test_year - 1))
                    hold_idx = np.where(hold_ok)[0]
                    tr_idx = np.where(~hold_ok)[0]
                    if len(hold_idx) > 0:
                        strat_h = strat_w.copy()
                        warn_h = warn_w.copy()
                        s_hold, w_hold = aux_labels(y_abs[hold_idx], delta_s[hold_idx])
                        strat_h[hold_idx], warn_h[hold_idx] = s_hold, w_hold
                        s_tr, w_tr = aux_labels(y_abs[tr_idx], delta_s[tr_idx])
                        strat_h[tr_idx], warn_h[tr_idx] = s_tr, w_tr
                        model_h = train_model(
                            Xw, y_norm, strat_h, warn_h, n_tr_samples,
                            args.epochs, args.device,
                            tr_idx=tr_idx, va_idx=hold_idx)
                        q_hold = predict_quantiles(model_h, Xw[hold_idx], args.device)
                        Qs["seas_holdout"] = calibrate_cqr(q_hold, y_norm[hold_idx], ALPHA)
                        row["n_holdout"] = int(len(hold_idx))
                        row["n_train_after"] = int(len(tr_idx))
                    else:
                        Qs["seas_holdout"] = (np.zeros(H), np.zeros(H))
                        row["n_holdout"] = 0

            # ---- 评估 ----
            for m in methods:
                Qm = Qs.get(m) if m != "raw" else None
                cov, cov_h, width_h, crps = eval_method(
                    q_te, cur_te, obs, scale, Qs=Qm)
                agg[m]["cov"][wi] = cov
                agg[m]["cov_h"][wi] = cov_h
                agg[m]["width"][wi] = float(np.mean(width_h))
                agg[m]["width_h"][wi] = width_h
                agg[m]["crps"][wi] = crps
                row[f"{m}_cov"] = round(cov, 4)
                row[f"{m}_w"] = round(float(np.mean(width_h)), 3)
                row[f"{m}_crps"] = round(crps, 4)
            n_valid += 1

            raw_cov = row.get("raw_cov", float("nan"))
            print(f"        测试月={te_center} n_tr={n_tr_samples} 校准段raw覆盖={row['cal_cov_raw']:.3f} "
                  f"| raw覆盖={raw_cov:.3f} raw宽={row.get('raw_w', float('nan')):.2f} "
                  f"rawCRPS={row.get('raw_crps', float('nan')):.4f}", flush=True)
            for m in methods:
                if m == "raw":
                    continue
                print(f"        {m:<13}覆盖={row[f'{m}_cov']:.3f} 宽={row[f'{m}_w']:.2f} "
                      f"CRPS={row[f'{m}_crps']:.4f}", flush=True)
            if X == F_CAL_DAYS and args.with_holdout:
                print(f"        [holdout] 季节块 n={row.get('n_holdout', 0)}（挖后训练 {row.get('n_train_after', 0)}）",
                      flush=True)

            window_rows_all.append(row)

        # ---- 变体聚合（NaN 感知：跳过训练段不足的窗口）----
        print(f"\n===== 变体 校准{X}d（α={ALPHA}，目标覆盖 {1 - ALPHA}；有效窗口 {n_valid}/{Nw}）=====", flush=True)
        hdr = f"  {'方法':<13}{'覆盖':<8}{'覆盖std':<10}{'区间宽':<9}{'CRPS':<9}{'CRPS vs raw'}"
        print(hdr, flush=True)
        for m in methods:
            a = agg[m]
            cov_std = float(np.nanstd(a["cov"]))
            w = float(np.nanmean(a["width"]))
            cr = float(np.nanmean(a["crps"]))
            rel = (np.nanmean(agg["raw"]["crps"]) - cr) / np.nanmean(agg["raw"]["crps"]) * 100
            print(f"  {m:<13}{np.nanmean(a['cov']):<8.3f}{cov_std:<10.3f}{w:<9.2f}{cr:<9.4f}{rel:+.1f}%", flush=True)
        agg_all[X] = agg

    # ---- 公共窗口子集对照（所有变体都能评估的窗口——诚实对比窗口 std）----
    # cal90 跳过窗口 1-2（训练段不足）、cal365 跳过窗口 1-8；公共子集 = 各变体都有效的窗口。
    # 各变体保留自己的 raw 基线（训练段不同，raw 也不同），只在公共窗口上重算指标。
    common_idx = None
    for X, agg in agg_all.items():
        valid = ~np.isnan(agg["raw"]["cov"])
        common_idx = valid if common_idx is None else (common_idx & valid)
    common_idx = np.asarray(common_idx)
    if common_idx.sum() > 0:
        print(f"\n===== 公共窗口子集对照（{int(common_idx.sum())} 个窗口，全部变体可比；各变体用自己的 raw 基线）=====", flush=True)
        print(f"  {'变体/方法':<16}{'覆盖':<8}{'覆盖std':<10}{'区间宽':<9}{'CRPS':<9}{'CRPS vs 本变体raw'}", flush=True)
        for X in variants:
            base = agg_all[X]["raw"]
            base_cr = float(np.mean(base["crps"][common_idx]))
            for m in agg_all[X]:
                a = agg_all[X][m]
                c = a["cov"][common_idx]
                cov_std = float(np.std(c))
                w = float(np.mean(a["width"][common_idx]))
                cr = float(np.mean(a["crps"][common_idx]))
                rel = (base_cr - cr) / base_cr * 100 if base_cr else 0.0
                tag = f"cal{X}/{m}"
                print(f"  {tag:<16}{c.mean():<8.3f}{cov_std:<10.3f}{w:<9.2f}{cr:<9.4f}{rel:+.1f}%", flush=True)
    else:
        common_cov = {}

    # ---- 输出 JSON ----
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                     "stride_days": STRIDE_DAYS, "T": T, "H": H, "epochs": args.epochs,
                     "alpha": ALPHA, "season_width": SEASON_WIDTH, "weight_sigma": WEIGHT_SIGMA,
                     "cal_variants": variants, "with_holdout": args.with_holdout,
                     "cqr": "Romano 2019 下/上分开校准",
                     "finite_sample": "Q=⌈(n+1)(1-α)⌉ 阶统计量",
                     "shared_test_windows": f"{Nw} 个 F 协议测试窗口（所有变体同一批；cal>30 时前若干窗口因训练段不足跳过）"},
        "variants": {},
        "windows": window_rows_all,
    }
    for X, agg in agg_all.items():
        res["variants"][f"cal{X}"] = {}
        for m, a in agg.items():
            res["variants"][f"cal{X}"][m] = {
                "coverage_mean": float(np.nanmean(a["cov"])),
                "coverage_std": float(np.nanstd(a["cov"])),
                "coverage_windows": np.nan_to_num(a["cov"], nan=-1).tolist(),
                "coverage_h": np.nanmean(a["cov_h"], axis=0).tolist(),
                "width_mean": float(np.nanmean(a["width"])),
                "width_h": np.nanmean(a["width_h"], axis=0).tolist(),
                "crps_mean": float(np.nanmean(a["crps"])),
            }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
