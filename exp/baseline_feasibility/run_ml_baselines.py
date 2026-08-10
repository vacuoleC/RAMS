# -*- coding: utf-8 -*-
"""对比基线可行性验证 —— 统计 ML 基线（可跑组）：正式协议下的 CRPS 对比

协议（与 L 探索一致，见 exp/model_enhancement/l_daily_scale/run_l.py）：
  - 日级采样（1D 均值聚合）
  - 滚动窗口：训练 730d / 测试 90d / 步长 45d（17 窗口）
  - 目标 = 未来 7 天表层浓度 conc_{t+h}（原始浓度单位）
  - 指标 = CRPS（分位数 p10/p50/p90 闭合形式）+ p50 RMSE + 覆盖率 + 相对持久化技能
  - 预测形式统一为分位数（p10/p50/p90）：
      持久化/线性AR/Ridge/MLP：确定性点 → 三点退化为同一点（退化 CRPS=MAE），
                            与 RamsNet 逐视界 CRPS 的退化口径一致（B7/D/L）。
      XGBoost/LightGBM：回归器 → 预测每视界的均值/峰值后作退化分布（同 Tick Tick 冠军口径），
                        另跑一次分位数回归（quantile alpha=0.1/0.5/0.9）作为校准口径。

基线定义（全部日级，滚动窗口）：
  1. persist      当前浓度 conc_t 当未来 7 天预测（退化分布）
  2. ar_ridge     Ridge：过去 N 天浓度 + 温度剖面均值 + 气象 → 各视界 conc_{t+h}
                 （绝对量回归，多输出）
  3. xgb_abs      XGBoost：过去 7 天窗口特征 → 未来 7 天各天 conc_{t+h}（多输出回归，退化分布）
  4. lgb_abs      LightGBM 同上
  5. xgb_peak     XGBoost：过去 7 天窗口特征 → 未来 7 天峰值（退化分布，任务口径：峰值）
  6. lgb_peak     LightGBM 同上
  7. xgb_q        XGBoost 分位数回归（alpha=0.1/0.5/0.9）→ 未来 7 天各天（校准口径）
  8. lgb_q        LightGBM 分位数回归同上

保密：只输出聚合统计量 / CRPS / 技能 / 相对提升，不打印原始数据行。
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

# 日级配置（与 L 探索 1D 尺度一致）
STEP_H = 24
T = 30        # 回看 30 天
H = 7         # 未来 7 天
LAG = 7       # 统计基线特征回看天数（线性/Ridge）
GRID = "24h"

BASELINES = ["persist", "ar_ridge", "xgb_abs", "lgb_abs", "xgb_peak", "lgb_peak",
             "xgb_q", "lgb_q"]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 B7/D/L 一致实现）。"""
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
    """3h 宽表 → 日级宽表（1D 均值聚合，与 L 一致）。"""
    loader = TensorBuilder(TensorConfig())
    wide = loader._load_wide(Path(parquet)).sort_index()
    wide = wide.resample(GRID).mean().dropna()
    return wide


def build_daily_window(wide, start_ts, tr_ts, end_ts):
    """切日级窗口，返回测试段样本。

    Returns:
      X_te (N, F) 测试特征（行 = 预测日 t，特征 = 过去 T 天统计/序列）
      y_te (N, H) 未来 7 天 conc_{t+h} 观测
      cur_te (N,)  当前 conc_t
      n_tr 训练样本数
    """
    df = wide[(wide.index >= start_ts) & (wide.index < end_ts)]
    y_raw = df["conc_0.5"].values.astype(np.float64)
    n = len(df)
    n_tr_rows = int((df.index < tr_ts).sum())
    # 与 L 协议对齐：窗口数 n_w = n - T - H，预测日 d = T-1+k
    #   cur = y[d]（当前浓度），y_abs = y[d+1 : d+1+H]（未来 7 天）
    n_w = n - T - H
    n_tr = max(0, n_tr_rows - T - H + 1)
    days = [T - 1 + k for k in range(n_w)]

    # 特征（每预测日一行）：过去 T 天浓度统计 + 水温剖面 + 气象（均值窗口统计）
    def window_stats(series, i):
        w = series[i - T + 1:i + 1]
        return [w[-1], w.mean(), w.std(), w[-1] - w[-7] if i >= 7 else 0.0]

    feat_rows = []
    for i in days:
        rows = []
        rows += window_stats(y_raw, i)
        for col in ["temp_0.5", "temp_5.0", "temp_10.0"]:
            if col in df.columns:
                rows += window_stats(df[col].values, i)
        rows += [df["delta_T"].values[i - T + 1:i + 1].mean(),
                 df["thermo_grad"].values[i - T + 1:i + 1].mean()]
        for m in METEO_COLS:
            if m in df.columns:
                rows += window_stats(df[m].values, i)
        feat_rows.append(rows)
    X = np.array(feat_rows, dtype=np.float64)
    y_abs = np.stack([y_raw[i + 1:i + 1 + H] for i in days])
    cur = np.array([y_raw[i] for i in days])

    return X, y_abs, cur, n_tr


def fit_predict(model_fn, X_tr, y_tr, X_te, model_kwargs=None):
    """fit 后 predict，返回 (N,H) 多输出（模型内做多输出处理）。"""
    m = model_fn(**model_kwargs) if model_kwargs else model_fn()
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def main():
    ap = argparse.ArgumentParser(description="对比基线可行性：统计 ML 基线（日级+滚动+CRPS）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--baselines", default=",".join(BASELINES))
    ap.add_argument("--max-windows", type=int, default=0, help="0=全部 17 窗口")
    ap.add_argument("--smoke", action="store_true", help="1 窗口 × 少量树")
    ap.add_argument("--out-json", default="exp/baseline_feasibility/ml_baselines_results.json")
    args = ap.parse_args()

    bls = [b.strip() for b in args.baselines.split(",") if b.strip()]
    t0 = time.time()
    print(f"== 对比基线：统计 ML 正式协议（{len(bls)} 基线）==", flush=True)
    print(f"   协议: 日级 1D 均值 | 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d"
          f" | 回看 T={T}d 视界 H={H}d | 目标 conc_0.5 未来 7 天", flush=True)

    wide = load_daily_wide(args.parquet)
    d0 = wide.index.min()
    print(f"[1] 日级宽表 {len(wide)} 天 {d0:%Y-%m-%d} → {wide.index.max():%Y-%m-%d}", flush=True)

    windows = []
    for wi in range(N_WINDOWS):
        start_ts = d0 + pd.Timedelta(days=STRIDE_DAYS * wi)
        tr_ts = start_ts + pd.Timedelta(days=TRAIN_DAYS)
        end_ts = start_ts + pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)
        windows.append((start_ts, tr_ts, end_ts))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个", flush=True)

    Nw = len(windows)
    agg = {b: {"crps": np.zeros(Nw), "crps_h": np.zeros((Nw, H)), "crps_p": np.zeros(Nw),
               "rmse": np.zeros(Nw), "cov": np.zeros(Nw)} for b in bls}

    # 轻量模型函数
    def _ridge():
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0)

    def _xgb():
        import xgboost as xgb
        if args.smoke:
            return xgb.XGBRegressor(n_estimators=60, max_depth=4, learning_rate=0.1,
                                    n_jobs=-1, objective="reg:squarederror", seed=0)
        return xgb.XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8,
                                n_jobs=-1, objective="reg:squarederror", seed=0)

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
        X, y_abs, cur, n_tr = build_daily_window(wide, start_ts, tr_ts, end_ts)
        X_tr, y_tr = X[:n_tr], y_abs[:n_tr]
        X_te, y_te, cur_te = X[n_tr:], y_abs[n_tr:], cur[n_tr:]
        Nte = len(X_te)
        print(f"     n_tr={n_tr} n_te={Nte} 特征 {X.shape[1]} 列", flush=True)

        preds = {}

        # 1. 持久化：conc_{t+h} = conc_t（退化分布）
        if "persist" in bls:
            preds["persist"] = np.repeat(cur_te[:, None], H, axis=1)

        # 2. 线性 AR / Ridge（过去 30 天 → 未来 7 天，多输出）
        if "ar_ridge" in bls:
            X_ar = X[:, :4]  # 用过去 30 天浓度统计子集（末值/均值/std/斜率）
            m = _ridge()
            m.fit(X_ar[:n_tr], y_tr)
            preds["ar_ridge"] = m.predict(X_ar[n_tr:])

        # 3-4. XGB/LGB 绝对量（多输出：未来 7 天各天）
        for tag, fn in (("xgb_abs", _xgb), ("lgb_abs", _lgb)):
            if tag not in bls:
                continue
            from sklearn.multioutput import MultiOutputRegressor
            m = MultiOutputRegressor(fn(), n_jobs=-1)
            m.fit(X_tr, y_tr)
            preds[tag] = m.predict(X_te)

        # 5-6. XGB/LGB 峰值（未来 7 天峰值）
        y_peak = y_tr.max(axis=1)
        for tag, fn in (("xgb_peak", _xgb), ("lgb_peak", _lgb)):
            if tag not in bls:
                continue
            m = fn()
            m.fit(X_tr, y_peak)
            p = m.predict(X_te)[:, None]
            preds[tag] = np.repeat(p, H, axis=1)

        # 7-8. XGB/LGB 分位数回归（alpha=0.1/0.5/0.9，逐视界各天）
        for tag, fn in (("xgb_q", _xgb), ("lgb_q", _lgb)):
            if tag not in bls:
                continue
            q = np.zeros((Nte, 3, H))
            for hi in range(H):
                for ai, alpha in enumerate((0.1, 0.5, 0.9)):
                    m = fn()
                    if tag.startswith("xgb"):
                        m.set_params(objective="reg:quantileerror", quantile_alpha=alpha)
                    else:
                        m.set_params(objective="quantile", alpha=alpha)
                    m.fit(X_tr, y_tr[:, hi])
                    q[:, ai, hi] = m.predict(X_te)
            preds[tag] = q.mean(axis=2)  # (N,3) 视界平均分位

        # 评估：每基线
        for b in bls:
            p = preds[b]
            # 统一为 (N,3,H)
            if p.ndim == 3:                      # (N,3,H) 逐视界分位（q 基线）
                q_conc = p
            elif p.shape[1] == 3:                # (N,3) 视界平均分位 → 各视界同分位
                q_conc = p[:, :, None].repeat(H, axis=2)
            else:                                # (N,H) 退化分布
                q_conc = np.stack([p, p, p], axis=1)
            q_conc = q_conc.astype(np.float64)

            obs = y_te
            cov = float(np.mean((obs >= q_conc[:, 0]) & (obs <= q_conc[:, 2])))
            crps_h = [float(np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                   q_conc[:, 2, h], obs[:, h])))
                      for h in range(H)]
            crps = float(np.mean(crps_h))
            # 持久化 CRPS（同窗口同测试段）
            if b != "persist":
                q_p = np.repeat(cur_te[:, None], H, axis=1)
                q_p = np.stack([q_p, q_p, q_p], axis=1)
                crps_p = float(np.mean([
                    np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 1, h], q_p[:, 2, h],
                                           obs[:, h])) for h in range(H)]))
            else:
                crps_p = crps
            rmse = float(np.sqrt(np.mean((q_conc[:, 1] - obs) ** 2)))
            agg[b]["crps"][wi] = crps
            agg[b]["crps_h"][wi] = crps_h
            agg[b]["crps_p"][wi] = crps_p
            agg[b]["rmse"][wi] = rmse
            agg[b]["cov"][wi] = cov
            skill = (crps_p - crps) / crps_p * 100 if crps_p else float("nan")
            print(f"   [{b:<9}] CRPS={crps:.4f} (persist {crps_p:.4f}, skill {skill:+.1f}%)  "
                  f"RMSE={rmse:.3f}  cov={cov:.3f}", flush=True)

    # ---- 聚合 ----
    print("\n===== 统计 ML 基线对照（17 窗口，CRPS 还原 conc 单位，越低越好）=====", flush=True)
    print(f"  {'基线':<10}{'CRPS':<10}{'持久化CRPS':<12}{'技能%':<10}{'RMSE':<10}{'覆盖':<8}", flush=True)
    summary = {}
    for b in bls:
        a = agg[b]
        cp = a["crps_p"].mean()
        rel = (cp - a["crps"].mean()) / cp * 100 if cp else float("nan")
        summary[b] = {
            "crps_mean": float(a["crps"].mean()),
            "crps_persist": float(cp),
            "skill_vs_persist_pct": float(rel),
            "rmse_mean": float(a["rmse"].mean()),
            "coverage_mean": float(a["cov"].mean()),
            "crps_h": a["crps_h"].mean(axis=0).tolist(),
            "crps_windows": a["crps"].tolist(),
        }
        print(f"  {b:<10}{a['crps'].mean():<10.4f}{cp:<12.4f}{rel:<+10.1f}"
              f"{a['rmse'].mean():<10.3f}{a['cov'].mean():<8.3f}", flush=True)

    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {
            "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
            "n_windows": Nw, "target": "conc_0.5 未来 7 天 (conc 单位)",
            "note": "日级 1D 均值聚合；预测=分位数(p10/p50/p90)；退化分布 CRPS=|y-q50|；"
                    "xgb_q/lgb_q 为分位数回归校准口径（视界平均分位）",
        },
        "baselines": summary,
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
