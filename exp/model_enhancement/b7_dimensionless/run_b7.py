# -*- coding: utf-8 -*-
"""B7 探索：去量纲化目标对比（绝对Δ vs 相对变化 vs 对数差 vs 绝对log）

核心假设：藻类浓度是**乘性过程**（指数生长），绝对 Δ = conc_{t+h}-conc_t 的尺度随季节
水平走（高浓度段 Δ 天然大），导致 B2 窗口间方差大；相对/对数口径在跨季节可比，
可能修复 B2 的两个弱点：窗口间覆盖率方差、p50 方向判别。

同一滚动窗口协议（复用 B1/B2：训练 730d / 测试 90d / 步长 45d，17 窗口）下对比 4 变体，
全部用分位数头 p10/p50/p90 + GRU 骨干（RamsNet，M1+M2+M4 多任务，冻结）+
输入含 conc_t 浓度历史 + 全剖面水温 + 气象：

  1. abs_delta : target = conc_{t+h} - conc_t                  （B1/B2 基线）
  2. rel_delta : target = (conc_{t+h} - conc_t) / max(conc_t, eps_den)
  3. log_ratio : target = log1p(conc_{t+h}) - log1p(conc_t)     （评估还原 conc_t×exp(pred)-1）
  4. abs_log   : target = log1p(conc_{t+h})                     （对照，只"对数"不"增量"）

数据：conc_0.5（表层）p5≈0.97、含 242 个零值（min=0）→ 对数一律用 log1p 保证安全；
相对变化分母用 max(conc_t, eps_den) 抑制低浓度放大噪声。记录低浓度段表现。

评估（同一滚动窗口协议，全部还原到原始浓度单位）：
  a. 区间覆盖率：真实目标落在 [p10,p90]（还原到 conc 单位）→ 重点看窗口间方差
     （高波动 vs 低波动窗口的覆盖率差）
  b. 每视界 CRPS：各自口径内 vs 各自持久化（0 变化 / 0 相对 / 0 对数 / 逐视界 log 持久化）
  c. p50 RMSE：还原到原始浓度单位
  d. p50 方向判别：sign(pred) vs sign(true) 命中率（看去量纲化是否改善 B2 的 0.540）

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

# 相对变化分母下限：浓度 < eps_den 时相对变化噪声放大；用 conc_t 每窗口分位数决定
# （默认取窗口训练段 conc 的 p10 作分母下限，低浓度段抑制）
EPS_DEN_Q = 0.10

VARIANTS = ["abs_delta", "rel_delta", "log_ratio", "abs_log"]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 T4/B2 一致实现）。"""
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


def build_features(df, feat_cols):
    """特征列：全剖面水温 temp_* + 气象 METEO_COLS（训练段均值/方差标准化，防泄漏）。
    输入已含 conc_t（作为一列输入特征），这里把整列都标准化。返回 (X, col_mu, col_sd)。"""
    X = df[feat_cols].values.astype(np.float32)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-8
    X = (X - mu) / sd
    return X.astype(np.float32)


def build_window(wide, i0, i1, feat_cols, strat_col="delta_T"):
    """返回窗口 [i0,i1) 的原始浓度与增量，用于构造 4 变体目标。

    Returns:
      Xw       (n_w, T, F) 标准化特征窗口（训练段统计）
      cur_raw  (n_w,) conc_t 原始尺度
      y_abs    (n_w, H) conc_{t+h} 原始尺度
      strat_w, warn_w  M2/M4 标签（复用 B2）
    """
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

    # M2 分层标签（B2 复用）
    delta = df[strat_col].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（B2 复用）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, cur_raw, y_abs, strat_w, warn_w


def make_targets(variant, cur_raw, y_abs, eps_den):
    """构造 4 变体训练目标（归一化尺度，窗口训练段拟合 scale 防泄漏）。
    Returns: y_norm (N,H) 归一化目标, scale (float) 还原乘子, scale_kind 说明
    """
    N, H_ = y_abs.shape
    if variant == "abs_delta":
        raw = y_abs - cur_raw[:, None]
        kind = "add"
    elif variant == "rel_delta":
        denom = np.maximum(cur_raw[:, None], eps_den)
        raw = (y_abs - cur_raw[:, None]) / denom
        kind = "mul"
    elif variant == "log_ratio":
        raw = np.log1p(y_abs) - np.log1p(cur_raw[:, None])
        kind = "mul"
    elif variant == "abs_log":
        raw = np.log1p(y_abs)
        kind = "mul"
    else:
        raise ValueError(variant)
    return raw, kind


def train_model(Xw, yw, strat_w, warn_w, n_tr, epochs, device):
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
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def back_to_conc(variant, cur_raw, q, scale, eps_den):
    """把 (N,3,H) 分位数预测还原到原始 conc 尺度（与 make_targets 逆映射一致）。"""
    qc = q * scale
    if variant == "abs_delta":
        return cur_raw[:, None, None] + qc  # (N,1,1)+(N,3,H) broadcast
    elif variant == "rel_delta":
        # target = (y - cur) / max(cur, eps_den)  →  y = cur + max(cur, eps_den) * qc
        denom = np.maximum(cur_raw, eps_den)[:, None, None]
        return cur_raw[:, None, None] + denom * qc
    elif variant == "log_ratio":
        # log1p conc = log1p(conc_t) + qc  →  conc = expm1(log1p(conc_t) + qc)
        return np.expm1(np.log1p(cur_raw)[:, None, None] + qc)
    elif variant == "abs_log":
        return np.expm1(qc)
    raise ValueError(variant)


def persistent_conc(variant, cur_raw, scale, eps_den):
    """各自口径的持久化（平凡基线），还原到 conc 尺度。
    abs_delta/rel_delta/log_ratio：预测"零变化"（目标=0）→ conc_{t+h}=conc_t；
    abs_log：无"增量"，持久化 = 逐视界当前值 conc_t，即 log1p 尺度 = log1p(conc_t)。"""
    N = len(cur_raw)
    if variant in ("abs_delta", "rel_delta", "log_ratio"):
        q0 = np.zeros((N, 3, H), dtype=np.float64)
        return back_to_conc(variant, cur_raw, q0, scale, eps_den)
    elif variant == "abs_log":
        # 还原到 conc 尺度直接 = conc_t（逐视界），不经过 back_to_conc
        c = cur_raw[:, None, None] + np.zeros((N, 3, H), dtype=np.float64)
        return c
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser(description="B7 去量纲化目标对比（4 变体）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out-json", default="exp/model_enhancement/b7_dimensionless/results.json")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    t0 = time.time()
    print(f"== B7 去量纲化目标对比（4 变体）==", flush=True)
    print(f"   变体: {variants}", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d；"
          f"GRU 骨干 + p10/p50/p90 分位数头 + M2/M4 多任务；目标 {T}h", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    # 输入含 conc_t 浓度历史（B7 要求）：加入表层浓度作为特征
    if "conc_0.5" not in feat_cols:
        feat_cols = feat_cols + ["conc_0.5"]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象 + conc_t）", flush=True)

    # 全数据集训练段（前 2 年）拟合 conc 分位数，用于相对变化分母下限
    y_all = wide["conc_0.5"].values.astype(np.float64)
    n_tr_global = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS + STRIDE_DAYS))
    eps_den = float(np.quantile(y_all[:n_tr_global], EPS_DEN_Q))
    print(f"   [数据] conc_0.5 前2年 p10={eps_den:.3f}（相对变化分母下限，低浓度抑制）", flush=True)
    print(f"   [数据] conc_0.5 min={y_all.min():.2f} 零值={(y_all==0).sum()} 个 → 对数用 log1p 安全", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个", flush=True)

    # 聚合器：每变体每窗口
    Nw = len(windows)
    agg = {v: {
        "cov": np.zeros(Nw), "crps_h": np.zeros((Nw, H)), "crps": np.zeros(Nw),
        "rmse": np.zeros(Nw), "dir_acc": np.zeros(Nw),
        "low_cov": np.zeros(Nw), "crps_p": np.zeros(Nw),
    } for v in variants}
    per_window_cur = []
    rows = []

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

        # 低浓度段标记（测试段 conc_t < eps_den 的样本，用于"低浓度噪声"分析）
        low_mask = cur_te < eps_den
        n_low = int(low_mask.sum())
        per_window_cur.append({"window": wi + 1, "start": str(st), "end": str(en),
                               "n_test": Nte, "n_low": n_low,
                               "cur_p10": float(np.quantile(cur_te, 0.10)),
                               "cur_med": float(np.median(cur_te)),
                               "cur_p90": float(np.quantile(cur_te, 0.90)),
                               "y_p10": float(np.quantile(y_te, 0.10)),
                               "y_med": float(np.median(y_te)),
                               "y_p90": float(np.quantile(y_te, 0.90))})

        for v in variants:
            print(f"    —— 变体 {v} ——", flush=True)
            raw, kind = make_targets(v, cur_raw, y_abs, eps_den)
            # 窗口训练段拟合尺度（防泄漏），归一化目标
            scale = float(np.std(raw[:n_tr])) + 1e-8
            y_norm = (raw / scale).astype(np.float32)

            model = train_model(Xw, y_norm, strat_w, warn_w, n_tr, args.epochs, args.device)
            q_norm = predict_quantiles(model, Xte, args.device)      # (N,3,H) 归一化
            q_conc = back_to_conc(v, cur_te, q_norm, scale, eps_den)  # (N,3,H) conc 单位

            # 观测（conc 单位）
            obs = y_te

            # 区间覆盖率（conc 单位）——重点
            cov = float(np.mean((obs >= q_conc[:, 0]) & (obs <= q_conc[:, 2])))
            # 低浓度段覆盖率
            if n_low > 0:
                agg[v]["low_cov"][wi] = float(np.mean(
                    (obs[low_mask] >= q_conc[low_mask, 0])
                    & (obs[low_mask] <= q_conc[low_mask, 2])))
            else:
                agg[v]["low_cov"][wi] = float("nan")

            # CRPS（conc 单位，还原后）
            crps_h = [float(np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                   q_conc[:, 2, h], obs[:, h])))
                      for h in range(H)]
            crps_avg = float(np.mean(crps_h))

            # 持久化（各自口径，还原到 conc 单位）
            q_p = persistent_conc(v, cur_te, scale, eps_den)
            crps_p_h = [float(np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 1, h],
                                                     q_p[:, 2, h], obs[:, h])))
                        for h in range(H)]
            crps_p = float(np.mean(crps_p_h))

            # p50 RMSE（conc 单位）
            rmse = float(np.sqrt(np.mean((q_conc[:, 1] - obs) ** 2)))

            # 方向判别：p50 增量符号 vs 真实增量符号（conc 单位）
            pred_delta = q_conc[:, 1] - cur_te[:, None]
            true_delta = obs - cur_te[:, None]
            dir_acc = float(np.mean(np.sign(pred_delta) == np.sign(true_delta)))

            # 记录
            agg[v]["cov"][wi] = cov
            agg[v]["crps_h"][wi] = crps_h
            agg[v]["crps"][wi] = crps_avg
            agg[v]["rmse"][wi] = rmse
            agg[v]["dir_acc"][wi] = dir_acc
            agg[v]["crps_p"][wi] = crps_p

            low_cov_this = agg[v]["low_cov"][wi]
            print(f"        覆盖={cov:.3f}  CRPS={crps_avg:.4f} (持久化 {crps_p:.4f})  "
                  f"p50RMSE={rmse:.3f}  方向={dir_acc:.3f}  "
                  f"低浓度覆盖={low_cov_this if n_low else float('nan'):.3f}",
                  flush=True)

    # ---- 聚合输出 ----
    print("\n===== 逐窗口 conc 分布（高/低波动窗口识别用）=====", flush=True)
    print(pd.DataFrame(per_window_cur).to_string(index=False), flush=True)

    print("\n===== 4 变体对照（全部还原 conc 单位）=====", flush=True)
    print(f"  {'变体':<12}{'覆盖':<8}{'覆盖窗口std':<12}{'CRPS':<9}{'持久化CRPS':<12}"
          f"{'p50RMSE':<9}{'方向命中':<9}{'低浓度覆盖':<10}", flush=True)
    for v in variants:
        a = agg[v]
        cov_std = float(np.std(a["cov"]))
        low_cov_mean = float(np.nanmean(a["low_cov"]))
        print(f"  {v:<12}{a['cov'].mean():<8.3f}{cov_std:<12.3f}{a['crps'].mean():<9.4f}"
              f"{a['crps_p'].mean():<12.4f}{a['rmse'].mean():<9.3f}"
              f"{a['dir_acc'].mean():<9.3f}{low_cov_mean:<10.3f}", flush=True)

    print("\n===== 每变体 CRPS vs 各自持久化（%相对技能，>0 模型更优）=====", flush=True)
    for v in variants:
        a = agg[v]
        cp = a["crps_p"].mean()
        rel = (cp - a["crps"].mean()) / cp if cp else 0
        print(f"  {v:<12}: 模型CRPS {a['crps'].mean():.4f} vs 持久化 {cp:.4f} "
              f"→ 相对技能 {rel * 100:+.1f}%", flush=True)

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "n_windows": Nw,
                     "eps_den": eps_den, "log_transform": "log1p (conc_0.5 含 242 零值)"},
        "variants": {},
        "windows_conc": per_window_cur,
    }
    for v in variants:
        a = agg[v]
        res["variants"][v] = {
            "coverage_mean": float(a["cov"].mean()),
            "coverage_windows": a["cov"].tolist(),
            "coverage_std": float(np.std(a["cov"])),
            "coverage_low_mean": float(np.nanmean(a.get("low_cov", [np.nan]))),
            "crps_mean": float(a["crps"].mean()),
            "crps_h": a["crps_h"].mean(axis=0).tolist(),
            "crps_persist": float(a["crps_p"].mean()),
            "rmse_conc_mean": float(a["rmse"].mean()),
            "dir_acc_mean": float(a["dir_acc"].mean()),
            "dir_acc_windows": a["dir_acc"].tolist(),
        }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
