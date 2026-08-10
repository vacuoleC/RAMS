# -*- coding: utf-8 -*-
"""T2 公开对比基线：RAMS 自家数据的"平凡基线"（可复现、秒级）

目的：为站位"单站点预测成果可发表"提供参照系——证明 RAMS 模型
(M1 GRU RMSE≈3.6) 显著优于 trivial baselines。三个平凡基线：
  1. 持久化 persistence：用最后观测浓度预测未来 24h（H=8 步不变）
  2. 气候学 climatology：用训练段"同月均值"预测（季节基线）
  3. 线性 AR(k)：用过去 k 个浓度自回归预测各步（可解释强基线）

评测协议与 train.py / tensor_builder.py 完全一致：
  - 目标 = 0.5m 层 total_conc，T=24 回看（3 天），H=8（24h）
  - 按窗口序号 70/15/15 时序切分（无泄漏）
  - 指标 = 测试段 RMSE（原始浓度单位），另给 MAE / 相对持久化改进

用法：python scripts/explore/t2_baseline_local.py
（纯 numpy/sklearn，CPU 秒级；不打印任何原始数据行，只出统计量）
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rams.data.tensor_builder import TensorBuilder, TensorConfig  # noqa: E402

T, H = 24, 8
PARQUET = Path("data/processed/standard.parquet")
SEED = 0


def rmse_mae(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    e = y_true - y_pred
    return float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e)))


def main() -> None:
    np.random.seed(SEED)
    print(f"[1] 读标准数据集并构建宽表: {PARQUET}", flush=True)
    builder = TensorBuilder(TensorConfig(T=T, H=H))
    wide = builder._load_wide(PARQUET)  # 与 train.py 同一透视逻辑
    target = "conc_0.5"  # 0.5m 表层总浓度（M1 预测目标）
    assert target in wide.columns, f"缺少目标列 {target}"

    conc = wide[target].values.astype(np.float64)
    n = len(conc)
    print(f"  宽表: {wide.shape}，目标列 {target}，T={T} H={H}", flush=True)

    # ---- 构建窗口：与 tensor_builder._make_windows 相同的滑动窗口 ----
    Xw, yw = [], []
    for i in range(n - T - H):
        Xw.append(conc[i:i + T])
        yw.append(conc[i + T:i + T + H])
    Xw = np.stack(Xw)  # (N, T) 历史浓度
    yw = np.stack(yw)  # (N, H) 未来 8 步
    N = len(Xw)
    n_tr, n_va = int(N * 0.7), int(N * 0.15)
    print(f"  窗口样本: {N} 个，切分 train {n_tr} / val {n_va} / test {N - n_tr - n_va}", flush=True)

    # 测试段（最后 15% 窗口）
    y_te = yw[n_tr + n_va:]
    x_last_te = Xw[n_tr + n_va:, -1]  # 每个测试窗口的最后观测浓度

    # ---- 基线 1：持久化 ----
    # 未来 24h 用最后观测值不变（保守但常很硬的时序基线）
    pred_persist = np.repeat(x_last_te[:, None], H, axis=1)
    r1, m1 = rmse_mae(y_te, pred_persist)

    # ---- 基线 2：季节气候学（同月均值，只从训练段拟合）----
    # 每个测试窗口取其"末时刻"（行 i+T-1）的月份，回填训练段同月浓度均值
    tr_wide = wide.iloc[:n_tr]
    monthly_mean = tr_wide[target].groupby(tr_wide.index.month).mean()  # {1..12: float}
    test_win_first = n_tr + n_va                 # 测试段第一个窗口
    te_months = np.array([wide.index[i + T - 1].month for i in range(test_win_first, N)])
    pred_clim = np.stack([monthly_mean[m] for m in te_months])
    pred_clim = np.repeat(pred_clim[:, None], H, axis=1)
    r2, m2 = rmse_mae(y_te, pred_clim)

    # ---- 基线 3：线性 AR(k) 自回归（用历史 k 个浓度 + 各步独立线性映射）----
    from sklearn.linear_model import LinearRegression
    k = 48  # 过去 6 天（96 个 3h 点）作特征
    X_ar = Xw[:, -k:]  # 每窗口取最后 k 个历史值
    X_ar_tr = X_ar[:n_tr]
    y_ar_tr = yw[:n_tr]
    X_ar_te = X_ar[n_tr + n_va:]
    # 每个预测步单独拟合线性模型（避免共享系数牺牲精度）
    pred_ar = np.zeros_like(y_te)
    for hh in range(H):
        lr = LinearRegression().fit(X_ar_tr, y_ar_tr[:, hh])
        pred_ar[:, hh] = lr.predict(X_ar_te)
    r3, m3 = rmse_mae(y_te, pred_ar)

    # ---- 输出统计量（不打印任何原始行）----
    print("\n===== 平凡基线结果（测试段，原始浓度单位）=====", flush=True)
    print(f"  持久化 persistence : RMSE={r1:.3f}  MAE={m1:.3f}", flush=True)
    print(f"  季节气候学 climatology : RMSE={r2:.3f}  MAE={m2:.3f}", flush=True)
    print(f"  线性 AR(k=48)     : RMSE={r3:.3f}  MAE={m3:.3f}", flush=True)
    print(f"\n  参照：RAMS M1 GRU 多任务模型（算力机 H100 实测）RMSE≈3.58~3.64", flush=True)
    print(f"  → GRU 相对持久化改进 : {100 * (1 - 3.64 / r1):.1f}%", flush=True)
    print(f"  → GRU 相对气候学改进 : {100 * (1 - 3.64 / r2):.1f}%", flush=True)
    print(f"  → GRU 相对 AR 改进    : {100 * (1 - 3.64 / r3):.1f}%", flush=True)

    # 持久化逐步 RMSE（展示"预测越远越难"）
    per_step = np.sqrt(np.mean((y_te - pred_persist) ** 2, axis=0))
    print(f"  持久化逐步 RMSE（步 1~8）: {np.round(per_step, 3)}", flush=True)
    print("\n冒烟通过", flush=True)


if __name__ == "__main__":
    main()
