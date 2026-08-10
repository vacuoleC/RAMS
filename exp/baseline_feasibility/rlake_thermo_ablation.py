# -*- coding: utf-8 -*-
"""对比基线：rLakeAnalyzer 温跃层深度 —— 协议内特征消融（lgb_q 主协议）

（mdl-baseline-compare · st-baseline-run 第 2 步，依赖 rlake_daily_thermo.py 产出的日级序列）

问题（冻结设计验收标准）：
  rLakeAnalyzer 温跃层深度是否在**正式协议（日级 + 滚动 730/90/45 + CRPS）**下
  相对现有分层代理 delta_T / thermo_grad 提供增量信息？

方法：以可解释性最强的**分位数口径基线 lgb_q**（探索已证相对持久化 +10.0%，是统计 ML
  中最强基线）为测试平台，做三组同协议消融（全部 17 窗口，CRPS conc 单位）：

  arm          | 特征集
  -------------|-------------------------------------------
  lgb_q        | 官方基线：过去 30 天 conc/temp 统计 + delta_T/thermo_grad 均值 + 气象（42 列）
  lgb_q+thermo | 官方特征 + thermo.depth_t 当前值与过去 7 天均值（+2 列）
  lgb_q-noStrat| 官方特征 − delta_T/thermo_grad（只删代理，不含 thermo）—— 阴性对照

  判定：lgb_q+thermo 相对 lgb_q 的 CRPS 增益 > 0 且可复现，说明 thermo.depth 在协议内
  有增量信息（非冗余）；lgb_q-noStrat 变差则说明分层代理本身有用。

与 run_ml_baselines.py 共享同一窗口构建（build_daily_window），保证与官方统计 ML
  基线严格同协议可比；唯一差异 = thermo 特征列的有无。

保密红线：只输出聚合统计量 / CRPS / 技能，不打印原始数据行。
  thermo.depth 日级序列来自 /tmp/rlake_daily.csv（算力机本地，只含 date + 派生深度）。

用法（算力机 sensecore）：
  PYTHONPATH=/data/RAMS/proj python3 exp/baseline_feasibility/rlake_thermo_ablation.py
  # 冒烟：加 --smoke（1 窗口 × 60 树）
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
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402

TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
N_WINDOWS = 17

STEP_H = 24
T = 30        # 回看 30 天
H = 7         # 未来 7 天
GRID = "24h"
LAG = 7

THERMO_CSV = "/tmp/rlake_daily.csv"
ARMS = ["lgb_q", "lgb_q+thermo", "lgb_q-noStrat"]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 run_ml_baselines 一致）。"""
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


def load_daily_wide(parquet):
    wide = TensorBuilder(TensorConfig())._load_wide(Path(parquet)).sort_index()
    return wide.resample(GRID).mean().dropna()


def build_daily_window(wide, thermo_series, start_ts, tr_ts, end_ts, use_thermo, use_strat):
    """切日级窗口，返回测试段样本。与 run_ml_baselines.build_daily_window 一致，
    另加 thermo 特征列（可选）与 strat 代理列（可选）。

    Returns:
      X_te (N, F) 测试特征 / y_te (N, H) / cur_te (N,) / n_tr / 特征列名
    """
    df = wide[(wide.index >= start_ts) & (wide.index < end_ts)]
    y_raw = df["conc_0.5"].values.astype(np.float64)
    n = len(df)
    n_tr_rows = int((df.index < tr_ts).sum())
    n_w = n - T - H
    n_tr = max(0, n_tr_rows - T - H + 1)
    days = [T - 1 + k for k in range(n_w)]
    # thermo 日级序列按窗口日期范围切片（与 df 行对齐）
    thermo = thermo_series.reindex(df.index)

    def window_stats(series, i):
        w = series[i - T + 1:i + 1]
        return [w[-1], w.mean(), w.std(), w[-1] - w[-7] if i >= 7 else 0.0]

    feat_rows, feat_names = [], []
    base_cols = ["conc_last", "conc_mean", "conc_std", "conc_slope"]
    for col in ["temp_0.5", "temp_5.0", "temp_10.0"]:
        base_cols += [f"{col}_last", f"{col}_mean", f"{col}_std", f"{col}_slope"]
    if use_strat:
        base_cols += ["delta_T_mean", "thermo_grad_mean"]
    for m in METEO_COLS:
        base_cols += [f"{m}_last", f"{m}_mean", f"{m}_std", f"{m}_slope"]
    feat_names = list(base_cols)
    if use_thermo:
        feat_names += ["thermo_cur", "thermo_mean7"]

    # thermo 日级序列按窗口对齐（thermo 已 reindex 到 df.index）
    for i in days:
        rows = []
        rows += window_stats(y_raw, i)
        for col in ["temp_0.5", "temp_5.0", "temp_10.0"]:
            rows += window_stats(df[col].values, i)
        if use_strat:
            rows += [df["delta_T"].values[i - T + 1:i + 1].mean(),
                     df["thermo_grad"].values[i - T + 1:i + 1].mean()]
        for m in METEO_COLS:
            rows += window_stats(df[m].values, i)
        if use_thermo:
            # thermo：当前日 + 过去 7 天均值（缺失补 0 —— thermo 只有 ~64% 可算日）
            td_cur = thermo.iloc[i]
            td7 = thermo.iloc[max(0, i - 6):i + 1].mean()
            rows += [0.0 if np.isnan(td_cur) else td_cur,
                     0.0 if np.isnan(td7) else td7]
        feat_rows.append(rows)
    X = np.array(feat_rows, dtype=np.float64)
    y_abs = np.stack([y_raw[i + 1:i + 1 + H] for i in days])
    cur = np.array([y_raw[i] for i in days])
    return X, y_abs, cur, n_tr, feat_names


def main():
    ap = argparse.ArgumentParser(description="rLakeAnalyzer 温跃层深度：协议内特征消融")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-json", default="exp/baseline_feasibility/rlake_thermo_ablation.json")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    t0 = time.time()
    print(f"== rLakeAnalyzer 温跃层深度：lgb_q 协议内特征消融（{len(arms)} arm）==", flush=True)
    print(f"   协议: 日级 1D 均值 | 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d"
          f" | 回看 T={T}d 视界 H={H}d | 目标 conc_0.5 未来 7 天", flush=True)

    wide = load_daily_wide(args.parquet)
    d0 = wide.index.min()
    print(f"[1] 日级宽表 {len(wide)} 天 {d0:%Y-%m-%d} → {wide.index.max():%Y-%m-%d}", flush=True)

    # thermo 日级序列
    thermo_full = pd.read_csv(THERMO_CSV, parse_dates=["date"]).set_index("date")["thermo_depth"]
    thermo_full = thermo_full.reindex(wide.index)  # 对齐日级宽表索引
    print(f"[1b] thermo.depth 日级序列 {thermo_full.notna().sum()}/{len(thermo_full)} 天可算", flush=True)

    windows = []
    for wi in range(N_WINDOWS):
        start_ts = d0 + pd.Timedelta(days=STRIDE_DAYS * wi)
        tr_ts = start_ts + pd.Timedelta(days=TRAIN_DAYS)
        end_ts = start_ts + pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)
        windows.append((start_ts, tr_ts, end_ts))
    if args.smoke:
        windows = windows[:1]
    print(f"[2] 滚动窗口 {len(windows)} 个", flush=True)

    Nw = len(windows)
    agg = {a: {"crps": np.zeros(Nw), "rmse": np.zeros(Nw), "cov": np.zeros(Nw),
               "crps_h": np.zeros((Nw, H))} for a in arms}

    def _lgb():
        import lightgbm as lgb
        if args.smoke:
            return lgb.LGBMRegressor(n_estimators=60, max_depth=4, learning_rate=0.1,
                                     n_jobs=-1, objective="regression", random_state=0,
                                     verbosity=-1)
        return lgb.LGBMRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                 subsample=0.8, colsample_bytree=0.8,
                                 n_jobs=-1, objective="regression", random_state=0,
                                 verbosity=-1)

    for wi, (start_ts, tr_ts, end_ts) in enumerate(windows):
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {start_ts:%Y-%m-%d} → {end_ts:%Y-%m-%d}", flush=True)
        # 每个 arm 一次窗口构建（列数不同）
        builds = {}
        for a in arms:
            use_th = a == "lgb_q+thermo"
            use_st = a != "lgb_q-noStrat"
            builds[a] = build_daily_window(wide, thermo_full, start_ts, tr_ts, end_ts,
                                           use_thermo=use_th, use_strat=use_st)
        n_tr = builds[arms[0]][3]
        for a in arms:
            X, y_abs, cur, n_tr_a, names = builds[a]
            X_tr, y_tr = X[:n_tr], y_abs[:n_tr]
            X_te, y_te = X[n_tr:], y_abs[n_tr:]
            Nte = len(X_te)
            # 分位数回归（alpha=0.1/0.5/0.9，逐视界）
            q = np.zeros((Nte, 3, H))
            for hi in range(H):
                for ai, alpha in enumerate((0.1, 0.5, 0.9)):
                    m = _lgb()
                    m.set_params(objective="quantile", alpha=alpha)
                    m.fit(X_tr, y_tr[:, hi])
                    q[:, ai, hi] = m.predict(X_te)
            q_conc = q.mean(axis=2)[:, :, None].repeat(H, axis=2)  # (N,3,H) 视界平均分位
            obs = y_te
            cov = float(np.mean((obs >= q_conc[:, 0]) & (obs <= q_conc[:, 2])))
            crps_h = [float(np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                   q_conc[:, 2, h], obs[:, h]))) for h in range(H)]
            crps = float(np.mean(crps_h))
            rmse = float(np.sqrt(np.mean((q_conc[:, 1] - obs) ** 2)))
            agg[a]["crps"][wi] = crps
            agg[a]["rmse"][wi] = rmse
            agg[a]["cov"][wi] = cov
            agg[a]["crps_h"][wi] = crps_h
            print(f"   [{a:<13}] CRPS={crps:.4f} RMSE={rmse:.3f} cov={cov:.3f} "
                  f"n_tr={n_tr} n_te={Nte} 特征={len(names)}", flush=True)

    print("\n===== 特征消融对照（17 窗口，CRPS 还原 conc 单位，越低越好）=====", flush=True)
    summary = {}
    base_crps = agg["lgb_q"]["crps"].mean()
    for a in arms:
        cp = agg[a]["crps"].mean()
        rel_base = (base_crps - cp) / base_crps * 100 if a != "lgb_q" else 0.0
        summary[a] = {
            "crps_mean": float(cp),
            "crps_vs_lgb_q_pct": float(rel_base),
            "rmse_mean": float(agg[a]["rmse"].mean()),
            "coverage_mean": float(agg[a]["cov"].mean()),
            "crps_h": agg[a]["crps_h"].mean(axis=0).tolist(),
            "crps_windows": agg[a]["crps"].tolist(),
        }
        print(f"  {a:<13} CRPS={cp:.4f} vs lgb_q {rel_base:+.2f}% RMSE={summary[a]['rmse_mean']:.3f} "
              f"cov={summary[a]['coverage_mean']:.3f}", flush=True)

    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {
            "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
            "n_windows": Nw, "target": "conc_0.5 未来 7 天 (conc 单位)",
            "note": "日级 1D 均值聚合；lgb_q 分位数回归（alpha=0.1/0.5/0.9）视界平均分位；"
                    "lgb_q+thermo 在官方特征上加 thermo.depth 当前值+7日均值；"
                    "lgb_q-noStrat 删 delta_T/thermo_grad 阴性对照",
        },
        "arms": summary,
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
