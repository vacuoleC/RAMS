# -*- coding: utf-8 -*-
"""B2 探索：增量分位数预测（Δ = conc_{t+h} - conc_t 的 p10/p50/p90）

假设：分位数区间应预测"浓度变化量 Δ"的不确定性，而非绝对浓度本身——
  区间表示"未来 24h 浓度可能上升/下降多少"，对预警（变化方向）更有意义。
  例如"未来 24h 浓度可能上升 2~8 个单位"，比"绝对浓度在 [x, y]"更可操作。

本脚本在**同一滚动窗口协议**下对比三种方法：
  1. 增量分位数（B2 假设）：target = Δ；输出 Δ 的 p10/p50/p90
  2. 绝对分位数（现有 M1 协议）：target = conc_{t+h}；输出 conc 的 p10/p50/p90
  3. 持久化（平凡基线）：Δ≡0（p50）；绝对浓度 = 当前值（逐视界）

评估（全部在测试段真实观测上，归一化尺度报告 + 用 y_sd 还原一个 RMSE 参考）：
  a. 区间覆盖率：真实 Δ 落在 [p10,p90] 的比例（增量法应 ≈80%）
  b. CRPS：闭合形式分位数 CRPS（与 T4 相同实现，proper scoring rule，越低越好）
  c. p50 RMSE：中位数预测的均方根误差
  对预警"变化方向"的判别：Δ 符号命中率（预测方向 = p50>0 时与真实 Δ 同号比例）

滚动窗口：训练 2 年、测试 3 个月、每 45 天推进（与 T4 协议一致）。

保密：只输出统计量 / 覆盖率 / CRPS / RMSE，绝不打印原始数据行。

用法（算力机 /data/RAMS/proj）：
  python3 exp/model_enhancement/b2_increment_quantile/run_b2.py
  python3 exp/model_enhancement/b2_increment_quantile/run_b2.py --smoke --max-windows 2
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
from rams.models.rams_net import QUANTILES, RamsNet  # noqa: E402
from rams.training.trainer import Trainer  # noqa: E402

T, H = 24, 8                 # 回看 3 天 / 预测 24h（8×3h 步）
EPOCHS = 30
SEED = 0
N_QUANT = len(QUANTILES)     # 3（p10/p50/p90）

# ---- 滚动窗口参数（天，3h 网格：1 天 = 8 个时刻）----
TRAIN_DAYS = 730             # 每个窗口用 2 年训练
TEST_DAYS = 90               # 每窗口测试后 3 个月
STRIDE_DAYS = 45             # 每 45 天推进一个窗口（重叠覆盖）
GRID_PER_DAY = 8


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 T4 一致实现）。

    通过分位点 (0.1,0.5,0.9) 与两个对称外推端点构造分段线性 CDF 的反函数，
    再对 CRPS 积分给出闭合解；三点重合时退化为 CRPS=MAE=|y-q50|。
    """
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


def build_window(wide, i0, i1, feat_cols, y_mu, y_sd):
    """窗口 [i0,i1)：归一化 X（训练段统计）、原始浓度 y、绝对分位数用归一化 y。

    返回：
      Xw   (n_w, T, F)  归一化特征窗口
      y_abs_raw (n_w, H)  绝对浓度观测（原始尺度，用于增量 Δ 计算）
      y_abs_norm (n_w, H)  绝对浓度观测（归一化尺度，绝对分位数目标/评估）
      cur_raw  (n_w,)  conc_t（原始尺度，每样本回看窗口末时刻浓度）
      strat_w, warn_w  M2/M4 多任务标签
    """
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))

    X = df[feat_cols].values.astype(np.float32)
    for i, c in enumerate(feat_cols):
        mu, sd = float(df[c].iloc[:n_tr].mean()), float(df[c].iloc[:n_tr].std()) + 1e-8
        X[:, i] = (X[:, i] - mu) / sd

    y_raw = df["conc_0.5"].values.astype(np.float64)
    y_norm = ((y_raw - y_mu) / y_sd).astype(np.float32)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs_raw = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    y_abs_norm = np.stack([y_norm[i + T:i + T + H] for i in range(n_w)]).astype(np.float32)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    delta = df["delta_T"].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)])

    warn_val = y_abs_raw.max(axis=1)
    n_win_tr = int(len(y_abs_raw) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, y_abs_raw, y_abs_norm, cur_raw, strat_w, warn_w


def train_model(Xw, yw, strat_w, warn_w, n_tr, epochs, device):
    """训练一个分位数 RamsNet（M1 分位数 + M2 多任务），返回模型。"""
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = RamsNet(feat_dim=Xw.shape[2], horizon=H, use_m4=True)
    trainer = Trainer(model, device=device)
    trainer.fit(Xw[:n_tr], yw[:n_tr], strat_w[:n_tr],
                Xw[n_tr:], yw[n_tr:], strat_w[n_tr:],
                warn_tr=warn_w[:n_tr], warn_va=warn_w[n_tr:],
                epochs=epochs, batch_size=128)
    return model


def predict_quantiles(model, X_te, device):
    """前向 → (N, 3, H) 分位数（归一化尺度）。"""
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def summarize(qp, obs, scale, label):
    """对 (N,3,H) 分位数预测 qp 与 (N,H) 观测 obs（同一归一化尺度）汇总统计。

    obs 传的尺度与 qp 一致（同归一化）。scale 是该目标的原始尺度乘子
    （Δ 目标用 sd_inc，绝对浓度目标用 y_sd），用于还原 width/RMSE。返回 dict。
    """
    q10, q50, q90 = qp[:, 0], qp[:, 1], qp[:, 2]
    cover = float(np.mean((obs >= q10) & (obs <= q90)))
    width = float(np.mean(q90 - q10))                    # 归一化尺度
    crps_h = [float(np.mean(crps_quantiles(qp[:, 0, h], qp[:, 1, h], qp[:, 2, h], obs[:, h])))
              for h in range(H)]
    crps_avg = float(np.mean(crps_h))
    rmse_norm = float(np.sqrt(np.mean((q50 - obs) ** 2)))
    return {
        f"{label}_coverage": cover,
        f"{label}_width": width,
        f"{label}_width_raw": width * scale,
        f"{label}_crps_h": crps_h,
        f"{label}_crps": crps_avg,
        f"{label}_rmse_norm": rmse_norm,
        f"{label}_rmse_raw": rmse_norm * scale,
    }


def main():
    ap = argparse.ArgumentParser(description="B2 增量分位数 vs 绝对分位数 vs 持久化")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch")
    ap.add_argument("--out-json", default="exp/model_enhancement/b2_increment_quantile/results.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"== B2 增量分位数预测 vs 绝对分位数 vs 持久化（滚动窗口）==", flush=True)
    print(f"   target_inc = conc[t+h] - conc[t]；绝对法 target = conc[t+h]", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象）", flush=True)

    # 全数据集训练段（前 2 年）拟合 y 的归一化参数（绝对法用）
    y_all = wide["conc_0.5"].values.astype(np.float64)
    n_tr_global = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS + STRIDE_DAYS))
    y_mu, y_sd = float(np.mean(y_all[:n_tr_global])), float(np.std(y_all[:n_tr_global])) + 1e-8
    print(f"   y 归一化参数（前 {n_tr_global} 点≈2y 拟合）: mu={y_mu:.3f} sd={y_sd:.3f}", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个（训练 {TRAIN_DAYS}d + 测试 {TEST_DAYS}d）", flush=True)

    # 聚合器
    keys = ["inc", "abs", "persist", "zero"]
    agg_crps_h = {k: np.zeros((len(windows), H)) for k in keys}
    agg_cover = {k: np.zeros(len(windows)) for k in keys}
    agg_width = {k: np.zeros(len(windows)) for k in keys}
    agg_rmse_n = {k: np.zeros(len(windows)) for k in keys}
    ntest_inc = 0
    n_up = 0
    n_up_correct = 0
    n_down = 0
    n_down_correct = 0
    n_pred_up = 0
    y_sd_list = []
    rows = []

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{len(windows)}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        Xw, y_abs_raw, y_abs_norm, cur_raw, strat_w, warn_w = build_window(
            wide, i0, i1, feat_cols, y_mu, y_sd)
        n_win = len(Xw)
        n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
        te_sl = slice(n_tr, n_win)

        Xte = Xw[te_sl]
        y_te_raw = y_abs_raw[te_sl]     # (N, H) 原始浓度观测
        y_te_norm = y_abs_norm[te_sl]   # (N, H) 归一化浓度观测

        # ---- 训练两个模型：增量（Δ 目标）与绝对（conc 目标）----
        # Δ 目标：y_inc = (conc[t+h]-conc[t]) / sd_inc（窗口训练段拟合 sd_inc）
        delta_raw = y_abs_raw - cur_raw[:, None]                          # (N,H) 原始 Δ
        sd_inc = float(np.std(delta_raw[:n_tr])) + 1e-8
        y_inc = (delta_raw / sd_inc).astype(np.float32)                   # (N,H) 归一化 Δ

        model_inc = train_model(Xw, y_inc, strat_w, warn_w, n_tr, args.epochs, args.device)
        model_abs = train_model(Xw, y_abs_norm, strat_w, warn_w, n_tr, args.epochs, args.device)

        # ---- 预测（归一化尺度）----
        q_inc = predict_quantiles(model_inc, Xte, args.device)   # Δ 的 p10/p50/p90（归一化 Δ）
        q_abs = predict_quantiles(model_abs, Xte, args.device)   # conc 的 p10/p50/p90（归一化 conc）

        # Δ 观测（归一化 Δ）：真实 Δ 的归一化 = (y_te_raw - cur) / sd_inc
        cur_te = cur_raw[te_sl]
        delta_te_norm = (y_te_raw - cur_te[:, None]) / sd_inc
        # 绝对浓度观测（归一化 conc）
        conc_te_norm = y_te_norm

        # ---- 方法 1：增量分位数（核心假设，原始尺度乘子 = sd_inc）----
        s_inc = summarize(q_inc, delta_te_norm, sd_inc, "inc")

        # ---- 方法 2：绝对分位数（现有 M1，原始尺度乘子 = y_sd）----
        s_abs = summarize(q_abs, conc_te_norm, y_sd, "abs")

        # ---- 方法 3a：增量持久化（Δ≡0：p10=p50=p90=0）----
        z = np.zeros_like(q_inc)
        s_zero = summarize(z, delta_te_norm, sd_inc, "zero")

        # ---- 方法 3b：绝对持久化（逐视界：conc_{t+h}=conc_t）----
        cur_norm = (cur_te - y_mu) / y_sd
        persist_conc = np.repeat(cur_norm[:, None], H, axis=1)
        qp = np.stack([persist_conc, persist_conc, persist_conc], axis=1)
        s_persist = summarize(qp, conc_te_norm, y_sd, "persist")

        # ---- 预警方向判别（增量法 p50>0 vs 真实 Δ>0）----
        pred_dir = q_inc[:, 1] > 0.0
        true_dir = delta_te_norm > 0.0
        n_up += int(true_dir.sum())
        n_down += int((~true_dir).sum())
        n_up_correct += int((pred_dir & true_dir).sum())
        n_down_correct += int((~pred_dir & ~true_dir).sum())
        n_pred_up += int(pred_dir.sum())
        ntest_inc += int(true_dir.size)

        for k in keys:
            agg_crps_h[k][wi] = {"inc": s_inc["inc_crps_h"], "abs": s_abs["abs_crps_h"],
                                 "zero": s_zero["zero_crps_h"], "persist": s_persist["persist_crps_h"]}[k]
            agg_cover[k][wi] = {"inc": s_inc["inc_coverage"], "abs": s_abs["abs_coverage"],
                                "zero": s_zero["zero_coverage"], "persist": s_persist["persist_coverage"]}[k]
            agg_width[k][wi] = {"inc": s_inc["inc_width"], "abs": s_abs["abs_width"],
                                "zero": s_zero["zero_width"], "persist": s_persist["persist_width"]}[k]
            agg_rmse_n[k][wi] = {"inc": s_inc["inc_rmse_norm"], "abs": s_abs["abs_rmse_norm"],
                                 "zero": s_zero["zero_rmse_norm"], "persist": s_persist["persist_rmse_norm"]}[k]

        y_sd_list.append(y_sd)
        rows.append({
            "window": wi + 1, "start": str(st), "end": str(en),
            "n_test": len(Xte), "n_train": n_tr, "y_sd": round(y_sd, 3),
            "sd_inc": round(sd_inc, 3),
            "inc_cov": round(s_inc["inc_coverage"], 4), "abs_cov": round(s_abs["abs_coverage"], 4),
            "inc_crps": round(s_inc["inc_crps"], 4), "abs_crps": round(s_abs["abs_crps"], 4),
            "zero_crps": round(s_zero["zero_crps"], 4), "persist_crps": round(s_persist["persist_crps"], 4),
            "inc_rmse_raw": round(s_inc["inc_rmse_raw"], 4), "abs_rmse_raw": round(s_abs["abs_rmse_raw"], 4),
            "inc_width_raw": round(s_inc["inc_width_raw"], 3), "abs_width_raw": round(s_abs["abs_width_raw"], 3),
        })
        print(f"        增量法: 覆盖={s_inc['inc_coverage']:.3f}  CRPS={s_inc['inc_crps']:.4f}  "
              f"p50ΔRMSE={s_inc['inc_rmse_raw']:.3f}  区间宽={s_inc['inc_width_raw']:.2f}", flush=True)
        print(f"        绝对法: 覆盖={s_abs['abs_coverage']:.3f}  CRPS={s_abs['abs_crps']:.4f}  "
              f"p50RMSE={s_abs['abs_rmse_raw']:.3f}  区间宽={s_abs['abs_width_raw']:.2f}", flush=True)
        print(f"        持久化: Δ≡0 CRPS={s_zero['zero_crps']:.4f} | 绝对 CRPS={s_persist['persist_crps']:.4f}",
              flush=True)

    # ---- 聚合 ----
    print("\n===== 逐窗口明细 =====", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    print("\n===== 均值对照（Δ 口径，CRPS 越低越好 / 覆盖率目标 80%）=====", flush=True)
    print(f"  {'方法':<22}{'CRPS(Δ)':<12}{'覆盖(Δ)':<12}{'区间宽':<12}{'p50RMSE'}", flush=True)
    print(f"  {'增量分位数':<22}{agg_crps_h['inc'].mean():<12.4f}{agg_cover['inc'].mean():<12.3f}"
          f"{agg_width['inc'].mean():<12.4f}{agg_rmse_n['inc'].mean():.4f}", flush=True)
    print(f"  {'持久化(Δ≡0)':<22}{agg_crps_h['zero'].mean():<12.4f}{agg_cover['zero'].mean():<12.3f}"
          f"{agg_width['zero'].mean():<12.4f}{agg_rmse_n['zero'].mean():.4f}", flush=True)

    print("\n===== 均值对照（绝对 conc 口径）=====", flush=True)
    print(f"  {'方法':<22}{'CRPS':<12}{'覆盖':<12}{'区间宽(norm)':<16}{'p50RMSE'}", flush=True)
    print(f"  {'绝对分位数':<22}{agg_crps_h['abs'].mean():<12.4f}{agg_cover['abs'].mean():<12.3f}"
          f"{agg_width['abs'].mean():<12.4f}{agg_rmse_n['abs'].mean():.4f}", flush=True)
    print(f"  {'持久化(逐视界)':<22}{agg_crps_h['persist'].mean():<12.4f}{agg_cover['persist'].mean():<12.3f}"
          f"{agg_width['persist'].mean():<12.4f}{agg_rmse_n['persist'].mean():.4f}", flush=True)

    print("\n===== 预警方向判别（增量法 p50 符号 vs 真实 Δ 符号）=====", flush=True)
    acc_dir = (n_up_correct + n_down_correct) / ntest_inc
    print(f"  样本: {ntest_inc}  真实上升 {n_up} ({n_up / ntest_inc:.1%}) 下降 {n_down} ({n_down / ntest_inc:.1%})"
          f"  预测上升 {n_pred_up} ({n_pred_up / ntest_inc:.1%})", flush=True)
    print(f"  方向命中率: {(n_up_correct + n_down_correct) / ntest_inc:.4f}  "
          f"（上升命中 {n_up_correct}/{n_up}={n_up_correct / max(n_up, 1):.3f}，"
          f"下降命中 {n_down_correct}/{n_down}={n_down_correct / max(n_down, 1):.3f}）", flush=True)

    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "n_windows": len(windows),
                     "y_sd": y_sd},
        "agg_crps_h": {k: agg_crps_h[k].mean(axis=0).tolist() for k in keys},
        "agg_crps_avg": {k: float(agg_crps_h[k].mean()) for k in keys},
        "agg_coverage": {k: float(agg_cover[k].mean()) for k in keys},
        "agg_width_norm": {k: float(agg_width[k].mean()) for k in keys},
        "agg_rmse_norm": {k: float(agg_rmse_n[k].mean()) for k in keys},
        "dir": {"n": ntest_inc, "n_up": int(n_up), "n_down": int(n_down),
                "n_pred_up": int(n_pred_up),
                "acc_up": float(n_up_correct / max(n_up, 1)),
                "acc_down": float(n_down_correct / max(n_down, 1)),
                "accuracy": float(acc_dir)},
        "windows": rows,
        "note": "增量法 Δ 口径 CRPS/覆盖/p50RMSE；绝对法 conc 口径；区间宽为归一化尺度（×y_sd 还原原始）",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
