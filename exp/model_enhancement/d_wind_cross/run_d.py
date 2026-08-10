# -*- coding: utf-8 -*-
"""D: wind_u 交叉特征在增量目标下重测（wind_u 短滞 / wind_u×conc 交叉）

假设：绝对浓度目标里自回归占主导、气象信号被淹没；增量目标（Δ=conc_{t+h}-conc_t）去掉
自回归后，"风大时浓度如何偏离"的弱信号可能浮现，wind_u 比 T1 绝对口径（-2.1%）更有用。

协议完全复用 B7 的 abs_delta 增量基线（训练 730d / 测试 90d / 步长 45d，17 窗口），
RamsNet 分位数模型（GRU + p10/p50/p90 + M2/M4 多任务），目标构造/还原/评估与 B7 一致。

4 特征变体（唯一差异 = 特征通道，其余完全一致）：
  base  : 现输入 = 全剖面水温 temp_*（20）+ 气象 METEO_COLS（6）+ conc_t（1）= 27 通道
  wind_u: base + wind_u 短滞通道（28 通道）
  cross : base + wind_u×conc_t 交叉通道（28 通道）
  both  : base + wind_u + cross（29 通道）

wind_u = wind_speed × cos(deg2rad(wind_dir))（与 T1/M5 一致）；cross = wind_u × conc_0.5
每时刻通道；T=24（3 天）窗口天然覆盖 0~2 天短滞，GRU 自适应滞后结构。

3 seed：每窗口每变体独立训练 3 次（seed 0/1/2），报告均值±std。

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
SEEDS = [0, 1, 2]

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）
TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
GRID_PER_DAY = 8

VARIANTS = ["base", "wind_u", "cross", "both"]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 B7 一致实现）。"""
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


def add_wind_features(df, variant):
    """追加 wind_u 与 wind_u×conc 交叉通道（与 T1/M5 一致的口径）。

    wind_u = wind_speed * cos(deg2rad(wind_dir))；cross = wind_u * conc_0.5（每时刻）。
    返回新 df（copy，不污染原表）。
    """
    out = df.copy()
    dir_rad = np.deg2rad(out["wind_dir"].values)
    out["wind_u"] = out["wind_speed"].values * np.cos(dir_rad)
    if variant in ("cross", "both"):
        out["wind_cross"] = out["wind_u"].values * out["conc_0.5"].values
    return out


def base_feat_cols(wide):
    """现输入特征列：全剖面水温 + 气象 + conc_t（B7 口径，27 列）。"""
    cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    cols = [c for c in cols if c in wide.columns]
    if "conc_0.5" not in cols:
        cols = cols + ["conc_0.5"]
    return cols


def build_window(wide, i0, i1, variant, strat_col="delta_T"):
    """返回窗口 [i0,i1) 的标准化特征（含变体通道）、原始浓度与增量目标。

    Returns:
      Xw       (n_w, T, F) 标准化特征窗口（训练段统计）
      cur_raw  (n_w,) conc_t 原始尺度
      y_abs    (n_w, H) conc_{t+h} 原始尺度
      strat_w, warn_w  M2/M4 标签（复用 B7）
    """
    df = add_wind_features(wide, variant)
    feat_cols = base_feat_cols(df)
    if variant in ("wind_u", "both"):
        feat_cols = feat_cols + ["wind_u"]
    if variant in ("cross", "both"):
        feat_cols = feat_cols + ["wind_cross"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    sl = df.iloc[i0:i1]
    n = len(sl)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))

    # 特征标准化（只用训练段）
    Xtr = sl[feat_cols].values[:n_tr].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((sl[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    y_raw = sl["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    # M2 分层标签（B7 复用）
    delta = sl[strat_col].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（B7 复用）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, cur_raw, y_abs, strat_w, warn_w, feat_cols


def make_abs_delta_targets(cur_raw, y_abs):
    """增量 abs_delta 目标（与 B7 一致）：Δ = conc_{t+h} - conc_t。
    Returns: raw (N,H) 原始尺度, kind 还原类型"""
    return y_abs - cur_raw[:, None], "add"


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
    """abs_delta 还原：conc = conc_t + Δ。q: (N,3,H) 归一化 → (N,3,H) conc 单位。"""
    qc = q * scale
    return cur_raw[:, None, None] + qc  # (N,1,1)+(N,3,H)


def persistent_conc(cur_raw, scale):
    """abs_delta 持久化：目标=0 → conc_{t+h}=conc_t。"""
    q0 = np.zeros((len(cur_raw), 3, H), dtype=np.float64)
    return back_to_conc(cur_raw, q0, scale)


def main():
    ap = argparse.ArgumentParser(description="D: wind_u 交叉特征在增量目标下重测（4 变体 × 3 seed）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch × 1 seed")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out-json", default="exp/model_enhancement/d_wind_cross/results.json")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    print(f"== D: wind_u 交叉特征在增量目标下重测（4 变体 × {len(seeds)} seed）==", flush=True)
    print(f"   变体: {variants}", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d；"
          f"GRU 骨干 + p10/p50/p90 分位数头 + M2/M4 多任务；增量 abs_delta 目标；"
          f"{T}h 回看 / {H}×3h 视界", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    print(f"[1b] 基准特征 {len(base_feat_cols(wide))} 列（temp_* + 气象 + conc_t）", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个", flush=True)

    Nw = len(windows)
    # 聚合器：每变体每 seed 每窗口
    # crps_h / rmse_h: (Nw, n_seed, H)；crps / rmse: (Nw, n_seed)
    agg = {v: {
        "cov": np.zeros((Nw, len(seeds))), "crps_h": np.zeros((Nw, len(seeds), H)),
        "crps": np.zeros((Nw, len(seeds))), "rmse": np.zeros((Nw, len(seeds))),
        "crps_p": np.zeros((Nw, len(seeds))),
    } for v in variants}
    per_window_cur = []

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        win_feat_dim = {}
        win_Nte = None
        for v in variants:
            Xw, cur_raw, y_abs, strat_w, warn_w, feat_cols = build_window(
                wide, i0, i1, v)
            n_win = len(Xw)
            n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
            te_sl = slice(n_tr, n_win)

            Xte = Xw[te_sl]
            cur_te = cur_raw[te_sl]
            y_te = y_abs[te_sl]                     # (N,H) 原始 conc 观测
            Nte = len(Xte)
            win_feat_dim[v] = len(feat_cols)
            win_Nte = Nte

            raw, kind = make_abs_delta_targets(cur_raw, y_abs)
            # 窗口训练段拟合尺度（防泄漏），归一化目标
            scale = float(np.std(raw[:n_tr])) + 1e-8
            y_norm = (raw / scale).astype(np.float32)

            for si, seed in enumerate(seeds):
                model = train_model(Xw, y_norm, strat_w, warn_w, n_tr,
                                    args.epochs, seed, args.device)
                q_norm = predict_quantiles(model, Xte, args.device)      # (N,3,H) 归一化
                q_conc = back_to_conc(cur_te, q_norm, scale)             # (N,3,H) conc 单位
                obs = y_te

                # 区间覆盖率（conc 单位）
                cov = float(np.mean((obs >= q_conc[:, 0]) & (obs <= q_conc[:, 2])))
                # CRPS（conc 单位）
                crps_h = [float(np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                       q_conc[:, 2, h], obs[:, h])))
                          for h in range(H)]
                crps_avg = float(np.mean(crps_h))
                # 持久化（conc 单位）
                q_p = persistent_conc(cur_te, scale)
                crps_p = float(np.mean([
                    np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 1, h],
                                           q_p[:, 2, h], obs[:, h])) for h in range(H)]))
                # p50 RMSE（conc 单位）
                rmse = float(np.sqrt(np.mean((q_conc[:, 1] - obs) ** 2)))

                agg[v]["cov"][wi, si] = cov
                agg[v]["crps_h"][wi, si] = crps_h
                agg[v]["crps"][wi, si] = crps_avg
                agg[v]["rmse"][wi, si] = rmse
                agg[v]["crps_p"][wi, si] = crps_p

            print(f"   [{v}] feat={len(feat_cols)}  "
                  f"CRPS={agg[v]['crps'][wi].mean():.4f}±{agg[v]['crps'][wi].std():.4f} "
                  f"(持久化 {agg[v]['crps_p'][wi].mean():.4f})  "
                  f"p50RMSE={agg[v]['rmse'][wi].mean():.3f}±{agg[v]['rmse'][wi].std():.3f}  "
                  f"覆盖={agg[v]['cov'][wi].mean():.3f}", flush=True)

        per_window_cur.append({"window": wi + 1, "start": str(st), "end": str(en),
                               "n_test": win_Nte, "feat_dim": win_feat_dim})

    # ---- 聚合输出 ----
    print("\n===== 4 变体对照（全部还原 conc 单位，3 seed 均值）=====", flush=True)
    print(f"  {'变体':<8}{'CRPS':<10}{'持久化CRPS':<12}{'CRPS相对技能':<14}"
          f"{'p50RMSE':<10}{'覆盖':<8}{'feat':<5}", flush=True)
    for v in variants:
        a = agg[v]
        cp = a["crps_p"].mean()
        rel = (cp - a["crps"].mean()) / cp if cp else 0
        print(f"  {v:<8}{a['crps'].mean():<10.4f}{cp:<12.4f}{rel * 100:<+14.1f}"
              f"{a['rmse'].mean():<10.3f}{a['cov'].mean():<8.3f}"
              f"{per_window_cur[0]['feat_dim'].get(v, '?'):<5}", flush=True)

    print("\n===== 每变体 CRPS/RMSE vs base（相对变化，<0 变体更优）=====", flush=True)
    for v in variants:
        if v == "base":
            continue
        base_crps, base_rmse = agg["base"]["crps"].mean(), agg["base"]["rmse"].mean()
        dcrps = (agg[v]["crps"].mean() - base_crps) / base_crps * 100
        drmse = (agg[v]["rmse"].mean() - base_rmse) / base_rmse * 100
        print(f"  {v:<8}: CRPS {dcrps:+.2f}%  RMSE {drmse:+.2f}%  "
              f"(vs base CRPS={base_crps:.4f} RMSE={base_rmse:.3f})", flush=True)

    # 逐窗口 CRPS 差（vs base，看是否全窗口一致或只在特定窗口）
    print("\n===== 逐窗口 CRPS（3 seed 均值，还原 conc 单位）=====", flush=True)
    hdr = "  w" + "".join(f"{v:>10}" for v in variants)
    print(hdr, flush=True)
    for wi in range(Nw):
        row = f"{wi + 1:>3}"
        for v in variants:
            row += f"{agg[v]['crps'][wi].mean():>10.4f}"
        print(row, flush=True)

    print("\n===== 每视界 CRPS（3 seed 均值）=====", flush=True)
    hdr = "  h" + "".join(f"{v:>10}" for v in variants)
    print(hdr, flush=True)
    for h in range(H):
        row = f"{h + 1:>3}"
        for v in variants:
            row += f"{agg[v]['crps_h'][..., h].mean():>10.4f}"
        print(row, flush=True)

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "n_windows": Nw, "seeds": seeds,
                     "target": "abs_delta (Δ=conc_{t+h}-conc_t)", "evaluate": "还原 conc 单位"},
        "variants": {},
        "windows": per_window_cur,
    }
    for v in variants:
        a = agg[v]
        res["variants"][v] = {
            "feat_dim": per_window_cur[0]["feat_dim"].get(v),
            "crps_mean": float(a["crps"].mean()),
            "crps_std": float(a["crps"].mean(axis=1).std()),          # 窗口间 std（seed 均值后）
            "crps_h": a["crps_h"].mean(axis=0).mean(axis=0).tolist(),  # (H,) 3-seed 均值
            "crps_persist": float(a["crps_p"].mean()),
            "rmse_mean": float(a["rmse"].mean()),
            "rmse_std": float(a["rmse"].mean(axis=1).std()),
            "coverage_mean": float(a["cov"].mean()),
            "crps_rel_vs_base": (float(a["crps"].mean()) - float(agg["base"]["crps"].mean()))
                                / float(agg["base"]["crps"].mean()) * 100.0,
            "rmse_rel_vs_base": (float(a["rmse"].mean()) - float(agg["base"]["rmse"].mean()))
                                / float(agg["base"]["rmse"].mean()) * 100.0,
            "crps_windows": a["crps"].mean(axis=1).tolist(),          # (Nw,) 3-seed 均值
        }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
