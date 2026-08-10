#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T4: CRPS + 滚动窗口评估 —— 修正固定 70/15/15 切分的不公平评估协议

背景（评估协议 bug）：
  旧协议用固定 70/15/15 时序切分：把高波动段（2021-2024，std≈13.9）训练、
  低波动段（2025，std≈3.1）测试，导致模型 RMSE（3.44~3.64）反而不如平凡
  持久化（1.24）。这是评估协议不公平，不是模型没用。EFI 挑战正是因此用
  **CRPS + 滚动窗口**评估。

本脚本（4 件事）：
  1. 滚动窗口评估：训练 2 年、测试 3 个月、每 45 天推进一个窗口（重叠覆盖）；
     每个窗口独立训练模型，在窗口后段测试，聚合所有窗口测试结果。
  2. CRPS 指标：模型输出分位数（p10/p50/p90），对每个视界 h=1..8 步分别算
     CRPS（连续排序概率分数，proper scoring rule，越低越好）。
  3. 逐视界基线：
     - 持久化：EFI 口径——当前观测值当预测（分位数=点 → CRPS=MAE），
       但**按视界分别评估**（不再把所有视界混成一个数），看清 h 越大越难。
     - 气候学：训练段"同月浓度"的 p10/p50/p90 分位数（季节分布），同样逐视界。
  4. 公平对比：同一窗口、同一测试样本集合上，跑 模型 / 持久化 / 气候学；
     报告每个视界的 CRPS、模型相对持久化的提升%、相对气候学的技能分数。

协议细节（防泄漏）：
  - 每窗口训练段 = 前 2 年（[i0, te_start)），测试段 = 后 3 个月（[te_start, i1)）。
  - 模型训练窗口取"预测起点在训练段且目标在测试段之前"的窗口
    （k ∈ [i0, te_start - T - H)），测试窗口取"预测起点 ≥ te_start"的窗口
    （k ∈ [te_start - T, i1 - T - H)）；边界处丢失 H 步预测点（标准做法）。
  - 归一化与 M2/M4 标签阈值只用每窗口训练段拟合（防泄漏）。
  - 三个评估对象（模型/持久化/气候学）在同一批测试窗口上算 CRPS，公平对比。

用法（算力机 /data/RAMS/proj 下）：
  python3 scripts/explore/t4_crps_eval.py                      # 全量（每窗口 30 epoch）
  python3 scripts/explore/t4_crps_eval.py --smoke              # 冒烟：2 窗口 × 2 epoch
  python3 scripts/explore/t4_crps_eval.py --epochs 20 --max-windows 6   # 稀疏采样加速

保密：只输出统计量 / CRPS / 相对提升，绝不打印原始数据行。
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402
from rams.models.rams_net import RamsNet  # noqa: E402
from rams.training.trainer import Trainer  # noqa: E402

# ---- 时间步长：3h 网格，1 天 = 8 个时刻 ----
GRID_PER_DAY = 8
T, H = 24, 8            # 回看 3 天 / 预测 24h（8×3h 步，h=1..8）

# ---- 滚动窗口参数（天）----
TRAIN_DAYS = 730        # 每个窗口用 2 年训练
TEST_DAYS = 90          # 每窗口测试后 3 个月
STRIDE_DAYS = 45        # 每 45 天推进一个窗口（重叠覆盖，评估更密）

EPOCHS = 30
SEED = 0

# ---- M5/M3 整合特征（可选，默认关闭以贴近 M1 主线评估）----
USE_WIND_U = False      # M5：加 wind_u 短滞特征（风 u 分量）
SELECTED_DEPTHS = None  # M3：可选最优 5 层 [1.5, 5.0, 8.5, 9.5, 10.0]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS（闭合形式，允许越序修正）。

    通过分位点 (0.1,0.5,0.9) 与两个对称外推端点构造分段线性 CDF 的反函数
    （quantile function），再对 CRPS 积分给出闭合解。与数值梯形积分一致
    （冒烟已验证，~1e-11）。退化情形（三点重合）退化为 CRPS=MAE=|y-q50|。
    """
    q10 = np.asarray(q10, dtype=np.float64)
    q50 = np.asarray(q50, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    # 排序修正越序预测（分位数必须单调）
    qs = np.sort(np.stack([q10, q50, q90], axis=-1), axis=-1)
    q10, q50, q90 = qs[..., 0], qs[..., 1], qs[..., 2]
    # 外推端点：p80 = q10 - (q50-q10)/4（等距延伸），p20 = q90 + (q90-q50)/4
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
        p1 = np.where(np.abs(slope) < 1e-12, 1.0, slope)  # 极小斜率钳制
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
    """读取宽表（复用 TensorBuilder 逻辑，保证与正式管线一致）。"""
    loader = TensorBuilder(TensorConfig(T=T, H=H))
    wide = loader._load_wide(Path(parquet))
    return wide.sort_index()


def build_global_windows(wide, feat_cols):
    """一次构建全量窗口（原始尺度，未归一化），供各窗口切片复用。"""
    n = len(wide)
    temp_cols = [c for c in wide.columns if c.startswith("temp_")]
    conc_cols = [c for c in wide.columns if c.startswith("conc_")]
    assert conc_cols, "宽表缺少 conc_* 列"
    X_all = wide[feat_cols].values.astype(np.float32)
    y_all = wide[conc_cols[0]].values.astype(np.float32)   # 表层浓度（M1 目标）
    delta = wide["delta_T"].values
    n_w = n - T - H
    Xw = np.stack([X_all[i:i + T] for i in range(n_w)]).astype(np.float32)
    yw = np.stack([y_all[i + T:i + T + H] for i in range(n_w)]).astype(np.float32)
    strat_w = np.array([delta[i + T - 1] for i in range(n_w)])   # 原始 delta_T（未阈值化）
    warn_val = yw.max(axis=1)                                     # 预警信号 = 未来 24h 峰值
    return Xw, yw, strat_w, warn_val


def make_window_split(i0, i1, te_start):
    """按时间边界计算 (k_tr, k_va, k_te)：窗口索引切片（防泄漏）。"""
    k_tr_end = te_start - T - H          # 训练窗口：目标在测试段之前
    k_te_start = te_start - T            # 测试窗口：预测起点 ≥ te_start
    k_te_end = i1 - T - H
    k_tr = slice(max(i0, 0), k_tr_end)
    n_tr = max(k_tr_end - max(i0, 0), 0)
    n_va = int(n_tr * 0.15)
    k_va = slice(k_tr_end - n_va, k_tr_end)
    k_fit = slice(max(i0, 0), k_tr_end - n_va)
    k_te = slice(k_te_start, k_te_end)
    return k_fit, k_va, k_te, n_tr


def fit_normalize(Xw, yw, k_fit, feat_cols):
    """用训练窗口拟合归一化参数并变换训练/测试切片。"""
    Xf = Xw[k_fit].reshape(-1, len(feat_cols)).astype(np.float64)
    mu = Xf.mean(axis=0)
    sd = Xf.std(axis=0) + 1e-8
    y_mu = float(yw[k_fit].mean())
    y_sd = float(yw[k_fit].std()) + 1e-8

    def norm(X):
        Xn = X.astype(np.float64)
        Xn = (Xn - mu) / sd
        return Xn.astype(np.float32)

    return norm, y_mu, y_sd


def run_model_window(wide, Xw, yw, strat_w, warn_val, k_fit, k_va, k_te,
                     feat_cols, epochs, device, m1_only=False):
    """训练一个窗口的模型，返回测试窗口分位数预测（原始尺度）。

    m1_only=True 时只用 M1 分位数损失（w_m2=w_m4=0，关闭 M4 头），用于
    分离"辅助任务稀释 M1"这一混淆因素（项目默认 w=1/3/2）。
    """
    # 归一化（只用训练窗口拟合）
    norm, y_mu, y_sd = fit_normalize(Xw, yw, k_fit, feat_cols)

    X_tr = norm(Xw[k_fit]); y_tr = (yw[k_fit] - y_mu) / y_sd
    X_va = norm(Xw[k_va]);  y_va = (yw[k_va] - y_mu) / y_sd
    X_te = norm(Xw[k_te]);  y_te = (yw[k_te] - y_mu) / y_sd

    # M2 分层标签（训练窗口 delta_T 中位数阈值，防泄漏）
    thr = float(np.median(strat_w[k_fit]))
    s_tr = (strat_w[k_fit] > thr).astype(np.int64)
    s_va = (strat_w[k_va] > thr).astype(np.int64)
    s_te = (strat_w[k_te] > thr).astype(np.int64)

    # M4 预警标签（训练窗口未来 24h 峰值分位数阈值 p75/p90/p97，防泄漏）
    qs = np.quantile(warn_val[k_fit], [0.75, 0.90, 0.97])
    w_tr = np.searchsorted(qs, warn_val[k_fit]).astype(np.int64)
    w_va = np.searchsorted(qs, warn_val[k_va]).astype(np.int64)
    w_te = np.searchsorted(qs, warn_val[k_te]).astype(np.int64)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    use_m4 = not m1_only
    model = RamsNet(feat_dim=len(feat_cols), horizon=H, use_m4=use_m4)
    if m1_only:
        trainer = Trainer(model, device=device, w_m2=0.0, w_m4=0.0)
    else:
        trainer = Trainer(model, device=device)   # 项目默认 w_m1/w_m2/w_m4 = 1/3/2
    trainer.fit(X_tr, y_tr, s_tr, X_va, y_va, s_va,
                warn_tr=w_tr if use_m4 else None,
                warn_va=w_va if use_m4 else None,
                epochs=epochs, batch_size=128)

    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)  # (N,3,H) 归一化
    qp_raw = q.cpu().numpy().astype(np.float64) * y_sd + y_mu               # 还原原始尺度
    return qp_raw, len(X_tr), y_sd


def main():
    ap = argparse.ArgumentParser(description="T4 CRPS + 滚动窗口评估")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--m1-only", action="store_true",
                    help="只用 M1 分位数损失（关闭 M2/M4 辅助权重），分离'辅助任务稀释 M1'混淆")
    ap.add_argument("--with-conc", action="store_true",
                    help="把过去 3 天表层浓度（conc_0.5）加进输入特征（自回归信息），"
                         "与持久化公平比较（默认模型输入只有 temp+meteo，看不到浓度历史）")
    ap.add_argument("--smoke", action="store_true", help="冒烟：2 窗口 × 2 epoch")
    ap.add_argument("--out-json", default="docs/t4_crps_eval_results.json")
    args = ap.parse_args()

    t0 = time.time()
    mode = "M1-only" if args.m1_only else "多任务 w=1/3/2"
    print(f"== T4 CRPS + 滚动窗口评估（train {TRAIN_DAYS}d / test {TEST_DAYS}d / stride {STRIDE_DAYS}d）==", flush=True)
    print(f"   模型: GRU 分位数 p10/p50/p90 [{mode}"
          f"{'+conc自回归' if args.with_conc else '+无浓度特征'}]；"
          f"基线: 持久化(逐视界) + 气候学(同月分位数)", flush=True)

    # ---- [1] 读宽表，构造特征列 ----
    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    if USE_WIND_U:
        dir_rad = np.deg2rad(wide["wind_dir"].values)
        wide["wind_u"] = wide["wind_speed"].values * np.cos(dir_rad)
        feat_cols.append("wind_u")
    if SELECTED_DEPTHS is not None:
        keep = {f"temp_{d:.1f}" for d in SELECTED_DEPTHS}
        feat_cols = ([c for c in feat_cols if c.startswith("temp_") and c in keep]
                     + list(METEO_COLS))
    if args.with_conc:
        assert "conc_0.5" in wide.columns, "缺少 conc_0.5 列"
        feat_cols = feat_cols + ["conc_0.5"]      # 自回归输入（目标自身历史）
    feat_cols = [c for c in feat_cols if c in wide.columns]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象{'+wind_u' if USE_WIND_U else ''}"
          f"{'+conc_hist' if args.with_conc else ''}）", flush=True)

    # 原始目标浓度数组（基线直接用它，保持原始尺度）
    conc = wide["conc_0.5"].values.astype(np.float64)
    months = np.array([wide.index[k].month for k in range(n)])

    # ---- [2] 全量窗口 + 滚动窗口切分 ----
    Xw, yw, strat_w, warn_val = build_global_windows(wide, feat_cols)
    n_w = len(Xw)
    span_days = TRAIN_DAYS + TEST_DAYS
    span_ticks = span_days * GRID_PER_DAY
    windows = []
    for i0 in range(0, n - span_ticks + 1, STRIDE_DAYS * GRID_PER_DAY):
        i1 = i0 + span_ticks
        te_start = i0 + TRAIN_DAYS * GRID_PER_DAY
        windows.append((i0, i1, te_start))
    if args.smoke:
        windows = windows[:2]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个（训练 {TRAIN_DAYS}d + 测试 {TEST_DAYS}d，每 {STRIDE_DAYS}d 推进）", flush=True)

    # ---- [3] 逐窗口评估：模型 / 持久化 / 气候学（原始浓度尺度 CRPS）----
    N = len(windows)
    model_crps = np.zeros((N, H))
    persist_crps = np.zeros((N, H))
    clim_crps = np.zeros((N, H))
    win_info = []
    for wi, (i0, i1, te_start) in enumerate(windows):
        st, en = wide.index[i0], wide.index[i1 - 1]
        tr_std = float(np.std(conc[i0:te_start]))
        k_fit, k_va, k_te, n_tr = make_window_split(i0, i1, te_start)
        k_te = slice(max(k_te.start, 0), k_te.stop)
        print(f"  [3.{wi + 1}] 窗口 {wi + 1}/{N}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"训练段 std={tr_std:.2f}  n_train={n_tr}  n_test={k_te.stop - k_te.start}",
              flush=True)

        qp_raw, n_fit, y_sd = run_model_window(
            wide, Xw, yw, strat_w, warn_val, k_fit, k_va, k_te, feat_cols,
            args.epochs, args.device, m1_only=args.m1_only)

        # 训练段"同月"浓度 → 每月 p10/p50/p90（原始尺度，气候学基线）
        month_q = {}
        for m in range(1, 13):
            v = conc[i0:te_start][months[i0:te_start] == m]
            if len(v) >= 5:
                month_q[m] = (float(np.quantile(v, 0.1)), float(np.quantile(v, 0.5)),
                              float(np.quantile(v, 0.9)))
        fallback = (float(np.quantile(conc[i0:te_start], 0.1)),
                    float(np.quantile(conc[i0:te_start], 0.5)),
                    float(np.quantile(conc[i0:te_start], 0.9)))

        k_test = np.arange(k_te.start, k_te.stop)
        p = k_test + T                       # 预测起点（全局索引，预测目标 t+p..p+H-1）
        for h in range(1, H + 1):
            target = conc[p + h - 1]         # 原始观测（视界 h）
            # 模型 CRPS（原始尺度）
            mh = float(np.mean(crps_quantiles(qp_raw[:, 0, h - 1], qp_raw[:, 1, h - 1],
                                              qp_raw[:, 2, h - 1], target)))
            # 持久化（EFI 口径：当前观测当预测；逐视界评估 → CRPS=MAE）
            ph = float(np.mean(np.abs(target - conc[p - 1])))
            # 气候学：目标时刻所在月份的 p10/p50/p90
            tgt_mon = months[p + h - 1]
            qc = np.array([month_q.get(m, fallback) for m in tgt_mon])
            ch = float(np.mean(crps_quantiles(qc[:, 0], qc[:, 1], qc[:, 2], target)))
            model_crps[wi, h - 1], persist_crps[wi, h - 1], clim_crps[wi, h - 1] = mh, ph, ch

        win_info.append({"window": wi + 1, "start": str(st), "end": str(en),
                         "n_train_fit": int(n_fit), "n_test": int(len(k_test)),
                         "train_std": round(tr_std, 3), "y_sd": round(y_sd, 3)})
        print(f"        模型/持久/气候 CRPS (h=1..8): "
              f"{np.round(model_crps[wi], 2)} / {np.round(persist_crps[wi], 2)} / "
              f"{np.round(clim_crps[wi], 2)}", flush=True)

    # ---- [4] 聚合：跨窗口取均值 ----
    mc, pc, cc = model_crps.mean(axis=0), persist_crps.mean(axis=0), clim_crps.mean(axis=0)
    improv = (pc - mc) / np.maximum(pc, 1e-9) * 100.0     # 相对持久化提升 %
    skill = cc / np.maximum(mc, 1e-9)                      # 相对气候学技能 >1 即优于气候学

    # ---- [5] 输出 ----
    print("\n===== 逐窗口信息 =====", flush=True)
    for r in win_info:
        print(f"  {r}", flush=True)

    print("\n===== 逐视界 CRPS 对照表（原始浓度尺度，越低越好）=====", flush=True)
    header = (f"  {'视界 h':<8}{'持久化':>10}{'气候学':>10}{'模型':>10}"
              f"{'模型vs持久化':>14}{'模型vs气候学':>14}")
    print(header, flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    for h in range(H):
        print(f"  h={h + 1:<7}{pc[h]:>10.3f}{cc[h]:>10.3f}{mc[h]:>10.3f}"
              f"{improv[h]:>12.1f}%{skill[h]:>13.2f}×", flush=True)
    print("  " + "-" * (len(header) - 2), flush=True)
    print(f"  {'平均':<8}{pc.mean():>10.3f}{cc.mean():>10.3f}{mc.mean():>10.3f}"
          f"{improv.mean():>12.1f}%{skill.mean():>13.2f}×", flush=True)
    print("  注：原始尺度 = 归一化 × 训练段 y_sd。持久化用当前观测当预测（CRPS=MAE），逐视界评估。", flush=True)

    print("\n===== 模型在哪视界超越持久化 =====", flush=True)
    for h in range(H):
        tag = "超越" if improv[h] > 0 else "未超"
        print(f"  h={h + 1}: 模型 CRPS {mc[h]:.3f} vs 持久化 {pc[h]:.3f} → {improv[h]:+.1f}% [{tag}]", flush=True)
    n_beat = int(np.sum(improv > 0))
    print(f"  模型在 {n_beat}/{H} 个视界 CRPS 低于持久化。", flush=True)

    print("\n===== 结论 =====", flush=True)
    best_h = int(np.argmin(mc)) + 1
    print(f"  模型最低 CRPS 视界: h={best_h}（{mc[best_h - 1]:.3f}）", flush=True)
    if np.mean(improv) > 0:
        print(f"  模型平均优于持久化 {improv.mean():.1f}%，且优于气候学 {skill.mean():.2f}× → 有真实预测能力", flush=True)
    else:
        print(f"  模型平均不及持久化（{improv.mean():.1f}%）→ 提示模型目前只配当'修正后的气候学'", flush=True)
    print(f"  运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)

    # ---- [6] 存统计量（不含原始数据）----
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
                     "stride_days": STRIDE_DAYS, "T": T, "H": H,
                     "epochs": args.epochs, "n_windows": len(windows),
                     "mode": "m1_only" if args.m1_only else "multitask_w1_3_2",
                     "use_conc_feature": bool(args.with_conc),
                     "features": "temp_* + meteo" + (" + wind_u" if USE_WIND_U else "")
                                 + (" + conc_hist" if args.with_conc else "")},
        "scale": "raw concentration (conc_0.5)",
        "crps_per_horizon": {"model": mc.tolist(), "persistence": pc.tolist(),
                             "climatology": cc.tolist()},
        "model_vs_persistence_pct": improv.tolist(),
        "model_vs_climatology_skill": skill.tolist(),
        "windows": win_info,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[result] 统计量已写入 {out}", flush=True)
    print("冒烟通过", flush=True)


if __name__ == "__main__":
    main()
