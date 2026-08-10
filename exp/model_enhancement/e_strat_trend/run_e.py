# -*- coding: utf-8 -*-
"""E 方向 · 分层状态趋势喂增量：基线 vs 加分层趋势特征（abs_delta 协议，3-seed）

背景：M5 证水温对藻类无因果中介（temp→conc 无 MCI 显著边）。本实验检验补充物理链
"分层状态变化 → 垂向混合 → 藻类分布变化"是否在**增量 Δ** 上浮现——把分层状态的
**趋势**（非当前值）作为特征喂增量预测，看对 Δ 是否有预测力。

快速验证（corr_precheck.py）已做：corr(Δconc, 分层趋势特征) 全部 ≈ 0
（|pearson|≤0.12、|spearman|≤0.08，Bootstrap95CI 几乎全含 0）。本脚本做**模型级确认**：
非线性/多元模式可能超出线性相关，用神经网络对比确认分层趋势特征是否有任何增益。

对照（同一 abs_delta 协议，17 窗口，与 B7 基线完全一致）：
  - base : 增量 abs_delta 目标 + 现输入（temp_* + 气象 + conc_t）       —— B7/B1/B2 基线
  - trend: 增量 abs_delta 目标 + 现输入 + 分层趋势特征（+3 维）          —— 本方向变体

分层趋势特征（3 天 = 8 时刻）：
  - trend_dT      = delta_T[t] - delta_T[t-8]      （表层-底层温差 3 天差分）
  - slope_dT      = delta_T 在 3 天窗内线性斜率     （鲁棒于端点噪声）
  - d_thermo_grad = thermo_grad[t] - thermo_grad[t-8]（温跃层最大梯度 3 天差分）

协议：滚动窗口 训练 730d / 测试 90d / 步长 45d（17 窗口），每窗口每变体 3 seed
独立训练同一 RamsNet（GRU + p10/p50/p90 分位数头 + M2/M4 多任务，与 B7 一致）。
目标归一化：窗口训练段 std 拟合 scale（防泄漏）。
评估：逐视界 CRPS（还原 conc 单位）+ p50 RMSE + 区间覆盖，跨窗口×seed 均值。

保密：只输出聚合统计量，不打印原始数据行。

用法：
  # 本地冒烟（CPU）
  D:/enviranment/Python313/python.exe exp/model_enhancement/e_strat_trend/run_e.py --smoke --device cpu
  # 算力机全量（3 seed × 2 变体 × 17 窗口 × 30 epoch）
  python3 exp/model_enhancement/e_strat_trend/run_e.py --seeds 3 --epochs 30 --device cuda
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
BASE_SEED = 0

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）
TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
GRID_PER_DAY = 8

# 分层趋势窗口（3 天 = 8 时刻）
LAG = 8

BASE_FEAT_KIND = "temp_* + meteo + conc_t"
TREND_FEATS = ["trend_dT", "slope_dT", "d_thermo_grad"]
VARIANTS = ["base", "trend"]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 B7/B2 一致）。"""
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


def add_trend_features(wide: pd.DataFrame) -> pd.DataFrame:
    """在宽表上追加 3 天分层趋势特征列，并丢弃前 LAG 时刻（趋势未定义的冷启动段）。

    两变体统一用裁剪后的宽表，保证时间轴与窗口完全一致（公平对照）。
    返回 wide 的拷贝，不修改原表。
    """
    dT = wide["delta_T"].values.astype(np.float64)
    tg = wide["thermo_grad"].values.astype(np.float64)

    trend_dT = np.full(len(wide), np.nan)
    trend_dT[LAG:] = dT[LAG:] - dT[:-LAG]

    slope_dT = np.full(len(wide), np.nan)
    x = np.arange(LAG + 1, dtype=np.float64)
    xm = x - x.mean()
    denom = (xm ** 2).sum()
    for i in range(LAG, len(wide)):
        y = dT[i - LAG:i + 1]
        slope_dT[i] = (xm * (y - y.mean())).sum() / denom

    d_thermo = np.full(len(wide), np.nan)
    d_thermo[LAG:] = tg[LAG:] - tg[:-LAG]

    out = wide.copy()
    out["trend_dT"] = trend_dT
    out["slope_dT"] = slope_dT
    out["d_thermo_grad"] = d_thermo
    return out.iloc[LAG:]


def build_window(wide, i0, i1, base_feat_cols, trend_feat_cols, strat_col="delta_T"):
    """返回窗口 [i0,i1) 的样本。feat 标准化只用窗口训练段（防泄漏）。

    Returns:
      X_base  (n_w, T, F_base)  基线特征窗口（标准化）
      X_trend (n_w, T, F_trend) 基线+分层趋势特征窗口（标准化）
      cur_raw (n_w,) conc_t 原始尺度
      y_abs   (n_w, H) conc_{t+h} 原始尺度
      strat_w, warn_w  M2/M4 标签
    """
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))

    def _norm(cols):
        Xtr = df[cols].values[:n_tr].astype(np.float32)
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0) + 1e-8
        return ((df[cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    Xb = _norm(base_feat_cols)
    Xt = _norm(trend_feat_cols)

    y_raw = df["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    X_base = np.stack([Xb[i:i + T] for i in range(n_w)]).astype(np.float32)
    X_trend = np.stack([Xt[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    # M2 分层标签（B2/B7 复用）
    delta = df[strat_col].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（B2/B7 复用）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return X_base, X_trend, cur_raw, y_abs, strat_w, warn_w


def train_model(Xw, yw, strat_w, warn_w, n_tr, epochs, seed, device):
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


def back_to_conc(cur_raw, q, scale):
    """abs_delta 还原：conc = conc_t + q*scale。"""
    return cur_raw[:, None, None] + q * scale


def main():
    ap = argparse.ArgumentParser(description="E 方向：分层趋势喂增量（基线 vs 加分层趋势）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch × 1 seed")
    ap.add_argument("--out-json", default="exp/model_enhancement/e_strat_trend/results.json")
    args = ap.parse_args()

    seeds = list(range(BASE_SEED, BASE_SEED + args.seeds))
    t0 = time.time()
    print(f"== E 分层状态趋势喂增量（基线 vs 加分层趋势，3-seed）==", flush=True)
    print(f"   变体: {VARIANTS}（分层趋势特征: {TREND_FEATS}，{LAG} 时刻=3 天）", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d；"
          f"GRU + p10/p50/p90 分位数头 + M2/M4 多任务；目标 abs_delta {H}h；seed×{args.seeds}",
          flush=True)

    wide_all = load_wide(args.parquet)
    wide = add_trend_features(wide_all)          # 两变体统一用裁剪后宽表
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（裁剪前 LAG 冷启动段）", flush=True)

    base_feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    base_feat_cols = [c for c in base_feat_cols if c in wide.columns]
    if "conc_0.5" not in base_feat_cols:
        base_feat_cols = base_feat_cols + ["conc_0.5"]
    trend_feat_cols = base_feat_cols + [c for c in TREND_FEATS if c in wide.columns]
    print(f"   基线特征 {len(base_feat_cols)} 个（temp_* + 气象 + conc_t）", flush=True)
    print(f"   变体特征 {len(trend_feat_cols)} 个（+ 分层趋势 {len(TREND_FEATS)} 维）", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
        args.seeds = 1
        args.epochs = 2
        seeds = [BASE_SEED]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    Nw = len(windows)
    print(f"[2] 滚动窗口 {Nw} 个 × 变体 {len(VARIANTS)} × seed {len(seeds)}", flush=True)

    # 聚合器
    agg = {v: {"crps_h": np.zeros((Nw, len(seeds), H)), "rmse_h": np.zeros((Nw, len(seeds), H)),
               "cov": np.zeros((Nw, len(seeds))), "crps": np.zeros((Nw, len(seeds))),
               "rmse": np.zeros((Nw, len(seeds))), "crps_p_h": np.zeros((Nw, len(seeds), H))}
          for v in VARIANTS}
    per_window_cur = []

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}", flush=True)

        X_base, X_trend, cur_raw, y_abs, strat_w, warn_w = build_window(
            wide, i0, i1, base_feat_cols, trend_feat_cols)
        n_win = len(X_base)
        n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
        te_sl = slice(n_tr, n_win)

        X_dict = {"base": X_base, "trend": X_trend}
        cur_te = cur_raw[te_sl]
        y_te = y_abs[te_sl]
        Nte = len(Xte if (Xte := X_base[te_sl]) is not None else X_base[te_sl])
        per_window_cur.append({"window": wi + 1, "start": str(st), "end": str(en),
                               "n_test": Nte,
                               "cur_med": float(np.median(cur_te)),
                               "cur_p10": float(np.quantile(cur_te, 0.10)),
                               "cur_p90": float(np.quantile(cur_te, 0.90))})

        # abs_delta 目标（归一化尺度）
        raw = y_abs - cur_raw[:, None]
        scale = float(np.std(raw[:n_tr])) + 1e-8
        y_norm = (raw / scale).astype(np.float32)

        # 持久化（目标=0 → conc_{t+h}=conc_t）
        q_p = cur_te[:, None, None] + np.zeros((Nte, 3, H))
        obs = y_te
        crps_p_h = [float(np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 1, h],
                                                 q_p[:, 2, h], obs[:, h])))
                    for h in range(H)]
        print(f"    持久化 CRPS/视界: {np.round(crps_p_h, 3)}", flush=True)

        for v in VARIANTS:
            for si, sd in enumerate(seeds):
                Xte = X_dict[v][te_sl]
                model = train_model(X_dict[v], y_norm, strat_w, warn_w, n_tr,
                                    args.epochs, sd, args.device)
                q_norm = predict_quantiles(model, Xte, args.device)
                q_conc = back_to_conc(cur_te, q_norm, scale)

                crps_h = np.array([np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                          q_conc[:, 2, h], obs[:, h]))
                                   for h in range(H)])
                rmse_h = np.sqrt(np.mean((q_conc[:, 1] - obs) ** 2, axis=0))
                cov = float(np.mean((obs >= q_conc[:, 0]) & (obs <= q_conc[:, 2])))
                agg[v]["crps_h"][wi, si] = crps_h
                agg[v]["rmse_h"][wi, si] = rmse_h
                agg[v]["crps"][wi, si] = float(crps_h.mean())
                agg[v]["rmse"][wi, si] = float(rmse_h.mean())
                agg[v]["cov"][wi, si] = cov
                agg[v]["crps_p_h"][wi, si] = crps_p_h
            print(f"    {v}: CRPS/视界={np.round(agg[v]['crps_h'][wi].mean(0), 3)}  "
                  f"RMSE/视界={np.round(agg[v]['rmse_h'][wi].mean(0), 3)}  "
                  f"覆盖={agg[v]['cov'][wi].mean():.3f}（跨 seed 均值）", flush=True)

    # ---- 聚合输出 ----
    print("\n===== 逐视界 CRPS（还原 conc 单位，跨窗口×seed 均值）=====", flush=True)
    print(f"  {'h':<4}{'base':<10}{'trend':<10}{'Δ(trend-base)':<14}{'持久化':<10}", flush=True)
    for h in range(H):
        b = agg["base"]["crps_h"].mean(axis=(0, 1))[h]
        t = agg["trend"]["crps_h"].mean(axis=(0, 1))[h]
        p = agg["base"]["crps_p_h"].mean(axis=(0, 1))[h]
        print(f"  h={h + 1:<3}{b:<10.4f}{t:<10.4f}{t - b:<+14.4f}{p:<10.4f}", flush=True)

    print("\n===== 逐视界 p50 RMSE（还原 conc 单位）=====", flush=True)
    print(f"  {'h':<4}{'base':<10}{'trend':<10}{'Δ(trend-base)':<14}", flush=True)
    for h in range(H):
        b = agg["base"]["rmse_h"].mean(axis=(0, 1))[h]
        t = agg["trend"]["rmse_h"].mean(axis=(0, 1))[h]
        print(f"  h={h + 1:<3}{b:<10.4f}{t:<10.4f}{t - b:<+14.4f}", flush=True)

    print("\n===== 汇总 =====", flush=True)
    for v in VARIANTS:
        a = agg[v]
        cp = a["crps_p_h"].mean()
        rel = (cp - a["crps"].mean()) / cp if cp else 0
        print(f"  {v:<6}: CRPS={a['crps'].mean():.4f}（vs 持久化 {cp:.4f} → {rel * 100:+.1f}%）"
              f"  RMSE={a['rmse'].mean():.3f}  覆盖={a['cov'].mean():.3f}", flush=True)

    b_c = agg["base"]["crps"].mean()
    t_c = agg["trend"]["crps"].mean()
    b_r = agg["base"]["rmse"].mean()
    t_r = agg["trend"]["rmse"].mean()
    print(f"  结论: CRPS Δ(trend-base)={t_c - b_c:+.4f}（{(t_c - b_c) / b_c * 100:+.2f}%）  "
          f"RMSE Δ(trend-base)={t_r - b_r:+.4f}（{(t_r - b_r) / b_r * 100:+.2f}%）", flush=True)

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "seeds": seeds,
                     "n_windows": Nw, "lag_3d": LAG,
                     "base_feat_dim": len(base_feat_cols),
                     "trend_feat_dim": len(trend_feat_cols),
                     "trend_feats": TREND_FEATS,
                     "note": "target=abs_delta，全 conc 单位评估"},
        "variants": {},
        "windows_conc": per_window_cur,
    }
    for v in VARIANTS:
        a = agg[v]
        res["variants"][v] = {
            "crps_mean": float(a["crps"].mean()),
            "crps_h": a["crps_h"].mean(axis=(0, 1)).tolist(),
            "crps_h_std": a["crps_h"].mean(axis=1).std(axis=0).tolist(),
            "crps_persist_h": a["crps_p_h"].mean(axis=(0, 1)).tolist(),
            "rmse_mean": float(a["rmse"].mean()),
            "rmse_h": a["rmse_h"].mean(axis=(0, 1)).tolist(),
            "coverage_mean": float(a["cov"].mean()),
        }
    res["delta"] = {
        "crps_delta_mean": float(t_c - b_c),
        "crps_delta_pct": float((t_c - b_c) / b_c * 100),
        "rmse_delta_mean": float(t_r - b_r),
        "rmse_delta_pct": float((t_r - b_r) / b_r * 100),
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
