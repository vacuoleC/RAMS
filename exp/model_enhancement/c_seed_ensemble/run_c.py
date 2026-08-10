# -*- coding: utf-8 -*-
"""C 探索：跨 seed 集成（Cross-Seed Ensemble）

核心假设（文献共识）：单模型 ≈ 小集成；对**同一增量分位数模型**（abs_delta 变体，
B7 结论中 CRPS/RMSE 最优）用不同随机种子训练多个副本，再对分位数预测取平均
（p10/p50/p90 各自平均），通常能降方差、稳定 CRPS/覆盖率。

同一滚动窗口协议（复用 B1/B2/B7：训练 730d / 测试 90d / 步长 45d，17 窗口），
每个窗口独立训练 N_SEED 个同架构模型（RamsNet：GRU 骨干 + p10/p50/p90 分位数头
+ M2/M4 多任务），仅随机种子不同；对照：
  - ensemble : p10/p50/p90 各自跨 seed 平均（最简单稳妥）
  - best_single : 单 seed 中该窗口测试段 CRPS 最好的一个（Oracle，上界参考）
  - avg_single : 单 seed CRPS 的窗口均值（平凡"平均单模型"水平）
  - worst_single : 单 seed 中最差一个（下界参考）

评估（全部还原到原始浓度单位，abs_delta 口径）：
  a. 逐视界 CRPS：ensemble vs 单 seed
  b. 逐视界 RMSE：p50 还原 conc 单位
  c. 区间覆盖率：真实目标落在 [p10,p90]（重点看窗口间方差是否更稳）
  d. 关键问题：集成降了多少 CRPS/RMSE？收益 <2% 则诚实判定"不值得"

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
SEEDS = [0, 1, 2, 3, 4]      # 5 个不同 seed（同协议）
N_SEED = len(SEEDS)

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）
TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
GRID_PER_DAY = 8


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 T4/B2/B7 一致实现）。"""
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


def build_window(wide, i0, i1, feat_cols):
    """返回窗口 [i0,i1) 的标准化特征、原始浓度、abs_delta 目标。"""
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))

    # 特征标准化（只用训练段）
    Xtr = df[feat_cols].values[:n_tr].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((df[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    y_raw = df["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    # M2 分层标签（B2/B7 复用）
    delta = df["delta_T"].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（B2/B7 复用）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, cur_raw, y_abs, strat_w, warn_w


def make_targets_abs_delta(cur_raw, y_abs):
    """abs_delta 目标（B7 最优口径）：Δ = conc_{t+h} - conc_t。"""
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


def main():
    ap = argparse.ArgumentParser(description="C 跨 seed 集成（abs_delta 分位数）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch × 2 seed")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out-json", default="exp/model_enhancement/c_seed_ensemble/results.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    print(f"== C 跨 seed 集成（abs_delta 分位数，{len(seeds)} seed）==", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d；"
          f"GRU 骨干 + p10/p50/p90 分位数头 + M2/M4 多任务；目标 {H}h", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    if "conc_0.5" not in feat_cols:
        feat_cols = feat_cols + ["conc_0.5"]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象 + conc_t）", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个 × {len(seeds)} seed", flush=True)

    Nw = len(windows)
    agg = {
        "ens_cov": np.zeros(Nw), "ens_crps_h": np.zeros((Nw, H)),
        "ens_crps": np.zeros(Nw), "ens_rmse": np.zeros(Nw),
        "best_crps": np.zeros(Nw), "avg_crps": np.zeros(Nw), "worst_crps": np.zeros(Nw),
        "best_rmse": np.zeros(Nw), "avg_rmse": np.zeros(Nw),
        "best_cov": np.zeros(Nw), "avg_cov": np.zeros(Nw),
        # 每 seed 记录（用于单 seed 分布分析）
        "seed_crps": np.zeros((Nw, len(seeds))),
        "seed_crps_h": np.zeros((Nw, len(seeds), H)),
        "seed_rmse": np.zeros((Nw, len(seeds))),
        "seed_cov": np.zeros((Nw, len(seeds))),
    }

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        Xw, cur_raw, y_abs, strat_w, warn_w = build_window(wide, i0, i1, feat_cols)
        n_win = len(Xw)
        n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
        te_sl = slice(n_tr, n_win)

        Xte = Xw[te_sl]
        cur_te = cur_raw[te_sl]
        y_te = y_abs[te_sl]                     # (N,H) 原始 conc 观测
        Nte = len(Xte)

        # abs_delta 目标 + 尺度（窗口训练段拟合，防泄漏）
        raw = make_targets_abs_delta(cur_raw, y_abs)
        scale = float(np.std(raw[:n_tr])) + 1e-8
        y_norm = (raw / scale).astype(np.float32)

        # 训练 len(seeds) 个模型，各自预测
        qs_all = np.zeros((len(seeds), Nte, 3, H), dtype=np.float64)  # (S,N,3,H) conc 单位
        for si, seed in enumerate(seeds):
            model = train_model(Xw, y_norm, strat_w, warn_w, n_tr, args.epochs,
                                args.device, seed)
            q_norm = predict_quantiles(model, Xte, args.device)
            q_conc = back_to_conc_abs(cur_te, q_norm, scale)
            qs_all[si] = q_conc
            print(f"    seed{seed} 训练+预测完成", flush=True)

        # ---- 每 seed 指标 ----
        obs = y_te
        for si in range(len(seeds)):
            q = qs_all[si]
            crps_h = [float(np.mean(crps_quantiles(q[:, 0, h], q[:, 1, h], q[:, 2, h], obs[:, h])))
                      for h in range(H)]
            crps = float(np.mean(crps_h))
            rmse = float(np.sqrt(np.mean((q[:, 1] - obs) ** 2)))
            cov = float(np.mean((obs >= q[:, 0]) & (obs <= q[:, 2])))
            agg["seed_crps"][wi, si] = crps
            agg["seed_crps_h"][wi, si] = crps_h
            agg["seed_rmse"][wi, si] = rmse
            agg["seed_cov"][wi, si] = cov

        # ---- 集成：分位数跨 seed 平均 ----
        q_ens = qs_all.mean(axis=0)             # (N,3,H)
        ens_crps_h = [float(np.mean(crps_quantiles(q_ens[:, 0, h], q_ens[:, 1, h],
                                                   q_ens[:, 2, h], obs[:, h])))
                      for h in range(H)]
        agg["ens_crps_h"][wi] = ens_crps_h
        agg["ens_crps"][wi] = float(np.mean(ens_crps_h))
        agg["ens_rmse"][wi] = float(np.sqrt(np.mean((q_ens[:, 1] - obs) ** 2)))
        agg["ens_cov"][wi] = float(np.mean((obs >= q_ens[:, 0]) & (obs <= q_ens[:, 2])))

        # ---- 单 seed 对照 ----
        agg["best_crps"][wi] = float(agg["seed_crps"][wi].min())
        agg["avg_crps"][wi] = float(agg["seed_crps"][wi].mean())
        agg["worst_crps"][wi] = float(agg["seed_crps"][wi].max())
        agg["best_rmse"][wi] = float(agg["seed_rmse"][wi].min())
        agg["avg_rmse"][wi] = float(agg["seed_rmse"][wi].mean())
        agg["best_cov"][wi] = float(agg["seed_cov"][wi].max())
        agg["avg_cov"][wi] = float(agg["seed_cov"][wi].mean())

        print(f"        集成: 覆盖={agg['ens_cov'][wi]:.3f}  CRPS={agg['ens_crps'][wi]:.4f}  "
              f"p50RMSE={agg['ens_rmse'][wi]:.3f}", flush=True)
        print(f"        单seed: CRPS best={agg['best_crps'][wi]:.4f} avg={agg['avg_crps'][wi]:.4f} "
              f"worst={agg['worst_crps'][wi]:.4f} | 覆盖 best={agg['best_cov'][wi]:.3f} "
              f"avg={agg['avg_cov'][wi]:.3f}", flush=True)
        print(f"        每seed CRPS: {['%.4f' % v for v in agg['seed_crps'][wi]]}", flush=True)

    # ---- 聚合输出 ----
    print("\n===== 跨 seed 集成 vs 单 seed（abs_delta 口径，17 窗口）=====", flush=True)
    print(f"  {'方法':<12}{'CRPS':<9}{'p50RMSE':<10}{'覆盖':<8}{'覆盖std':<9}", flush=True)
    ens_crps = float(agg["ens_crps"].mean())
    ens_rmse = float(agg["ens_rmse"].mean())
    ens_cov = float(agg["ens_cov"].mean())
    ens_cov_std = float(np.std(agg["ens_cov"]))
    avg_crps = float(agg["avg_crps"].mean())
    avg_rmse = float(agg["avg_rmse"].mean())
    avg_cov = float(agg["avg_cov"].mean())
    best_crps = float(agg["best_crps"].mean())
    worst_crps = float(agg["worst_crps"].mean())
    print(f"  {'ensemble':<12}{ens_crps:<9.4f}{ens_rmse:<10.3f}{ens_cov:<8.3f}{ens_cov_std:<9.3f}", flush=True)
    print(f"  {'avg_single':<12}{avg_crps:<9.4f}{avg_rmse:<10.3f}{avg_cov:<8.3f}"
          f"{float(np.std(agg['avg_cov'])):<9.3f}", flush=True)
    print(f"  {'best_single':<12}{best_crps:<9.4f}{agg['best_rmse'].mean():<10.3f}"
          f"{agg['best_cov'].mean():<8.3f}{float(np.std(agg['best_cov'])):<9.3f}", flush=True)
    print(f"  {'worst_single':<12}{worst_crps:<9.4f}{'-':<10}", flush=True)

    print("\n===== 集成收益（相对 avg_single）=====", flush=True)
    print(f"  CRPS: {ens_crps:.4f} vs {avg_crps:.4f} → {(avg_crps - ens_crps) / avg_crps * 100:+.2f}%", flush=True)
    print(f"  RMSE: {ens_rmse:.3f} vs {avg_rmse:.3f} → {(avg_rmse - ens_rmse) / avg_rmse * 100:+.2f}%", flush=True)
    print(f"  覆盖: {ens_cov:.3f} (std {ens_cov_std:.3f}) vs {avg_cov:.3f} (std {float(np.std(agg['avg_cov'])):.3f})", flush=True)

    print("\n===== 逐视界 CRPS（集成 vs avg_single）=====", flush=True)
    avg_crps_h = agg["seed_crps_h"].mean(axis=0).mean(axis=0)
    ens_crps_h = agg["ens_crps_h"].mean(axis=0)
    for h in range(H):
        rel = (avg_crps_h[h] - ens_crps_h[h]) / avg_crps_h[h] * 100
        print(f"  h{h + 1}: 集成 {ens_crps_h[h]:.4f} vs 单seed {avg_crps_h[h]:.4f} "
              f"→ {rel:+.2f}%", flush=True)

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "n_windows": Nw,
                     "seeds": seeds, "ensemble": "p10/p50/p90 各自跨 seed 平均",
                     "variant": "abs_delta"},
        "ensemble": {
            "crps_mean": ens_crps, "crps_h": agg["ens_crps_h"].mean(axis=0).tolist(),
            "crps_windows": agg["ens_crps"].tolist(),
            "rmse_mean": ens_rmse, "cov_mean": ens_cov, "cov_windows": agg["ens_cov"].tolist(),
            "cov_std": ens_cov_std,
        },
        "avg_single": {
            "crps_mean": avg_crps, "rmse_mean": avg_rmse, "cov_mean": avg_cov,
            "cov_std": float(np.std(agg["avg_cov"])),
            "seed_crps_windows": agg["seed_crps"].tolist(),
            "seed_rmse_windows": agg["seed_rmse"].tolist(),
            "seed_cov_windows": agg["seed_cov"].tolist(),
        },
        "best_single": {
            "crps_mean": best_crps, "rmse_mean": float(agg["best_rmse"].mean()),
            "cov_mean": float(agg["best_cov"].mean()), "cov_std": float(np.std(agg["best_cov"])),
        },
        "worst_single": {"crps_mean": worst_crps},
        "gains": {
            "crps_vs_avg": (avg_crps - ens_crps) / avg_crps,
            "rmse_vs_avg": (avg_rmse - ens_rmse) / avg_rmse,
            "cov_vs_avg": ens_cov - avg_cov,
            "cov_std_vs_avg": ens_cov_std - float(np.std(agg["avg_cov"])),
        },
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
