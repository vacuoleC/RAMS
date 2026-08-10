# -*- coding: utf-8 -*-
"""E 方向 · 快速验证：corr(Δconc, Δstratification)

假设：分层状态**变化**（趋势/增量）与藻类浓度**增量** Δconc 存在相关性——
若分层趋势对增量无预测力，corr 应接近 0，物理链"分层趋势 → Δconc"在数据上不成立。

步骤：
  1. 从宽表读 delta_T（表层-底层温差）、thermo_grad（温跃层最大梯度）
     —— 与生产 tensor_builder._load_wide 同一口径（3h 网格、dropna）
  2. 分层趋势特征（3 天 = 8 时刻窗口内的变化）：
       trend_dT      = delta_T[t] - delta_T[t-LAG]    （Δ 分层强度，3 天差分）
       d_dT_3d       = 同上（与 trend_dT 相同，保留语义别名）
       slope_dT      = delta_T 在 3 天窗口内线性斜率（对时间回归，鲁棒于端点噪声）
       d_thermo_grad = thermo_grad 3 天差分
       trend_thermo  = thermo_grad 3 天差分（别名，供同句式调用）
  3. Δconc 目标（增量 abs_delta 协议）：Δc_h = conc_0.5[t+h] - conc_0.5[t]，
     h ∈ {1, 3, 8}（覆盖 3h / 9h / 24h 视界）
  4. 输出：跨全数据集 + 每窗口（训练段）的 Pearson / Spearman 相关；
     3-seed 用 Bootstrap（对样本重采样）估计相关性的 95% CI，跨种子差异。
     只输出聚合统计量，不打印原始数据行。

用法：D:/enviranment/Python313/python.exe exp/model_enhancement/e_strat_trend/corr_precheck.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

from rams.data.tensor_builder import TensorBuilder, TensorConfig  # noqa: E402

T, H = 24, 8
LAG = 8                      # 3 天（3h 网格）
HORIZONS = [1, 3, 8]         # 3h / 9h / 24h
TRAIN_DAYS, TEST_DAYS, STRIDE_DAYS = 730, 90, 45
GRID_PER_DAY = 8
SEEDS = [0, 1, 2]


def make_trend_features(wide: pd.DataFrame) -> pd.DataFrame:
    """构造分层趋势特征（对 delta_T / thermo_grad 做 3 天差分与线性斜率）。"""
    dT = wide["delta_T"].values.astype(np.float64)
    tg = wide["thermo_grad"].values.astype(np.float64)

    trend_dT = np.full(len(dT), np.nan)
    trend_dT[LAG:] = dT[LAG:] - dT[:-LAG]
    d_dT_3d = trend_dT.copy()

    # delta_T 在 3 天窗口内线性斜率（对时刻索引回归，t-LAG..t 共 LAG+1 点）
    slope_dT = np.full(len(dT), np.nan)
    x = np.arange(LAG + 1, dtype=np.float64)
    xm = x - x.mean()
    denom = (xm ** 2).sum()
    for i in range(LAG, len(dT)):
        y = dT[i - LAG:i + 1]
        slope_dT[i] = (xm * (y - y.mean())).sum() / denom

    d_tg = np.full(len(tg), np.nan)
    d_tg[LAG:] = tg[LAG:] - tg[:-LAG]
    trend_thermo = d_tg.copy()

    return pd.DataFrame({
        "trend_dT": trend_dT,
        "d_dT_3d": d_dT_3d,
        "slope_dT": slope_dT,
        "d_thermo_grad": d_tg,
        "trend_thermo": trend_thermo,
    }, index=wide.index)


def correlate(feat, dconc, method="pearson"):
    mask = ~(np.isnan(feat) | np.isnan(dconc))
    if mask.sum() < 30:
        return float("nan")
    return float(getattr(pd.Series(feat[mask]), "corr")(pd.Series(dconc[mask]), method=method))


def bootstrap_ci(feat, dconc, seed, n_boot=500):
    """Bootstrap 95% CI（对样本重采样），每个 seed 一个 RNG → 3-seed 差异。"""
    mask = ~(np.isnan(feat) | np.isnan(dconc))
    x = feat[mask]
    y = dconc[mask]
    n = len(x)
    if n < 30:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    rng_shift = np.random.default_rng(seed + 1000)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[b] = np.corrcoef(x[idx], y[idx])[0, 1]
    return (float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975)))


def main():
    loader = TensorBuilder(TensorConfig(T=T, H=H))
    wide = loader._load_wide(Path("data/processed/standard.parquet")).sort_index()
    n = len(wide)
    print(f"[数据] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格）", flush=True)

    ft = make_trend_features(wide)
    conc = wide["conc_0.5"].values.astype(np.float64)

    print("\n=== 全数据集相关 corr(分层趋势特征, Δconc) ===", flush=True)
    print(f"{'特征':<16}{'h':<4}{'Pearson':<10}{'Spearman':<10}"
          f"{'Bootstrap95CI(pear)':<24}{'样本数':<8}", flush=True)
    rows = []
    for feat_name in ["trend_dT", "slope_dT", "d_thermo_grad"]:
        for h in HORIZONS:
            dconc = conc[h:] - conc[:-h] if h > 0 else np.zeros_like(conc)
            # 对齐：Δconc[t] 对应时刻 t，分层趋势特征也取 t（趋势截至 t）
            dconc_full = np.full(n, np.nan)
            dconc_full[:-h] = conc[h:] - conc[:-h]
            pc = correlate(ft[feat_name].values, dconc_full, "pearson")
            sc = correlate(ft[feat_name].values, dconc_full, "spearman")
            mask = ~(np.isnan(ft[feat_name].values) | np.isnan(dconc_full))
            ciss = []
            for s in SEEDS:
                lo, hi = bootstrap_ci(ft[feat_name].values, dconc_full, s)
                ciss.append(f"seed{s}:({lo:+.3f},{hi:+.3f})")
            rows.append({"feat": feat_name, "h": h, "pearson": pc, "spearman": sc})
            print(f"{feat_name:<16}{h:<4}{pc:<10.4f}{sc:<10.4f}"
                  f"{'  '.join(ciss):<24}{int(mask.sum()):<8}", flush=True)

    # 每窗口训练段相关（对照滚动协议）
    print("\n=== 每窗口训练段相关（训练 730d 起点）===", flush=True)
    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + TRAIN_DAYS * GRID_PER_DAY))
    win_rows = []
    for wi, (i0, i1) in enumerate(windows):
        sl = slice(i0, i1)
        dconc_full = np.full(n, np.nan)
        dconc_full[:-8] = conc[8:] - conc[:-8]
        pc = correlate(ft["trend_dT"].values[sl], dconc_full[sl], "pearson")
        sc = correlate(ft["trend_dT"].values[sl], dconc_full[sl], "spearman")
        pc_tg = correlate(ft["d_thermo_grad"].values[sl], dconc_full[sl], "pearson")
        win_rows.append({"window": wi + 1, "pear_trend_dT": pc, "spear_trend_dT": sc,
                         "pear_d_thermo": pc_tg})
        print(f"  w{wi + 1:<3} pear(trend_dT)={pc:+.3f}  spear(trend_dT)={sc:+.3f}  "
              f"pear(d_thermo)={pc_tg:+.3f}", flush=True)

    # 摘要统计（无原始数据行）
    pcs = [r["pearson"] for r in rows]
    print(f"\n=== 摘要 ===", flush=True)
    print(f"全数据 |pearson| 范围: {min(abs(p) for p in pcs):.4f} ~ {max(abs(p) for p in pcs):.4f}",
          flush=True)
    pws = [r["pear_trend_dT"] for r in win_rows]
    pwt = [r["pear_d_thermo"] for r in win_rows]
    print(f"窗口 |pear(trend_dT)| 范围: {min(abs(p) for p in pws):.4f} ~ {max(abs(p) for p in pws):.4f}  "
          f"均值 {np.mean(pws):+.4f}", flush=True)
    print(f"窗口 |pear(d_thermo)| 范围: {min(abs(p) for p in pwt):.4f} ~ {max(abs(p) for p in pwt):.4f}  "
          f"均值 {np.mean(pwt):+.4f}", flush=True)

    out = Path("exp/model_enhancement/e_strat_trend/corr_precheck.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(__import__("json").dumps({
        "n_rows": n, "lags_3d": LAG, "horizons": HORIZONS, "seeds": SEEDS,
        "corr_rows": rows, "windows_train_corr": win_rows,
        "summary": {
            "pearson_abs_range": [min(abs(p) for p in pcs), max(abs(p) for p in pcs)],
            "window_trend_dT_pear_mean": float(np.mean(pws)),
            "window_d_thermo_pear_mean": float(np.mean(pwt)),
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
