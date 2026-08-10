# -*- coding: utf-8 -*-
"""F 探索：共形校准（split conformal）让增量分位数区间的覆盖率有统计保证

背景：B2/B7 增量分位数模型（Δ=conc_{t+h}-conc_t 的 p10/p50/p90）覆盖率 0.79-0.82，
接近目标 80% 但"碰运气"——不是统计保证。本探索验证：共形校准能否把覆盖率
稳定保证到目标水平（α=0.2→80%、α=0.1→90%），以及区间宽度/CRPS 的成本。

方法（split conformal，最简单、可证有限样本保证）：
  每个滚动窗口内：训练 730d 拟合分位数模型 → 紧邻的 30d 作校准集算 conformal
  scores → 90d 测试集用校准后的分位数做区间。共形在**归一化 Δ 空间**做
  （与目标尺度一致、affine 可逆），再还原 conc 单位评估。

4 个方法（目标 α，每窗口每视界独立校准，n_cal=240/视界）：
  1. raw     ：模型原始 p10/p90，不校准（B2/B7 基线）
  2. abs50   ：split conformal on median，score=|y-q50|（绝对残差），双侧 ±Q
  3. symabs  ：split conformal on interval，score=max(q10-y, y-q90)，双侧 ±Q
  4. cqr     ：CQR（Romano 2019）——下/上分开校准，score_low=max(q10-y,0)，
               score_up=max(y-q90,0)，[q10-Q_low, q90+Q_up]

有限样本校正：Q = 校准集 score 的 ⌈(n+1)(1-α)⌉ 阶统计量（比 naive 分位数更紧的保证）。

评估（全部还原 conc 单位，滚动窗口协议）：
  a. 覆盖率（raw vs 各校准方法，α=0.2 与 0.1）——重点看窗口间稳定性（std）
  b. 区间宽度变化（校准膨胀了多少，Δ 单位）
  c. CRPS 变化（3 分位数闭合形式；校准只调外沿、保 p50；强制 q10≤p50≤q90 排序）
  d. 逐视界覆盖率（h=1..8，看是否逐视界达标）

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

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）：训练 730d / 校准 30d / 测试 90d / 步长 45d
TRAIN_DAYS = 730
CAL_DAYS = 30
TEST_DAYS = 90
STRIDE_DAYS = 45
GRID_PER_DAY = 8

ALPHAS = [0.2, 0.1]     # 目标不覆盖率（1-α = 80% / 90% 覆盖率）
METHODS = ["raw", "abs50", "symabs", "cqr"]


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


def finite_q(level, n_cal):
    """有限样本校正的阶统计量索引（1-indexed）：⌈(n+1)(1-α)⌉。

    Split conformal 保证：P(覆盖率 ≥ 1-α) ≥ 1-δ 在交换性假设下成立（有限样本）。
    """
    k = int(np.ceil((n_cal + 1) * level))
    return max(1, min(k, n_cal))


def load_wide(parquet):
    loader = TensorBuilder(TensorConfig(T=T, H=H))
    wide = loader._load_wide(Path(parquet))
    return wide.sort_index()


def build_window(wide, i0, i1, feat_cols, strat_col="delta_T"):
    """窗口 [i0,i1)（=训练730d+校准30d+测试90d）：特征/原始浓度/增量/标签。

    Returns:
      Xw       (n_w, T, F) 标准化特征窗口（训练段统计）
      cur_raw  (n_w,) conc_t 原始尺度
      y_abs    (n_w, H) conc_{t+h} 原始尺度
      seg      (n_w,) 0=训练 / 1=校准 / 2=测试 段标记
      strat_w, warn_w  M2/M4 标签（复用 B2）
    """
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = TRAIN_DAYS * GRID_PER_DAY
    n_cal = CAL_DAYS * GRID_PER_DAY

    Xtr = df[feat_cols].values[:n_tr].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((df[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    y_raw = df["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    # 段标记：样本 i 所在行归属
    seg = np.zeros(n_w, dtype=np.int64)
    for i in range(n_w):
        row = i + T - 1
        if row < n_tr:
            seg[i] = 0
        elif row < n_tr + n_cal:
            seg[i] = 1
        else:
            seg[i] = 2

    # M2 分层标签（B2 复用）
    delta = df[strat_col].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（B2 复用）
    warn_val = y_abs.max(axis=1)
    qs = np.quantile(warn_val[:n_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, cur_raw, y_abs, seg, strat_w, warn_w


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


def predict_quantiles(model, X, device):
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def calibrate(q_cal, y_cal, alpha):
    """在归一化 Δ 空间计算 conformal scores 并返回各方法每视界的校准量。

    Args:
      q_cal (N,3,H) 校准集分位数（归一化 Δ）
      y_cal (N,H)   校准集观测（归一化 Δ）
      alpha (float) 目标不覆盖率
    Returns: dict method -> (H,) Q 调整量（abs50/symabs 单侧；cqr 为 (Q_low, Q_up)）
    """
    q10, q50, q90 = q_cal[:, 0], q_cal[:, 1], q_cal[:, 2]
    n_cal = len(y_cal)

    out = {}
    # abs50: |y - q50|
    sc_abs50 = np.abs(y_cal - q50)
    out["abs50"] = np.zeros(H)
    for h in range(H):
        k = finite_q(1 - alpha, n_cal)
        out["abs50"][h] = np.sort(sc_abs50[:, h])[k - 1]

    # symabs: max(q10 - y, y - q90)
    sc_sym = np.maximum(q10 - y_cal, y_cal - q90)
    out["symabs"] = np.zeros(H)
    for h in range(H):
        k = finite_q(1 - alpha, n_cal)
        out["symabs"][h] = np.sort(sc_sym[:, h])[k - 1]

    # cqr: 下/上分开
    sc_low = np.maximum(q10 - y_cal, 0.0)
    sc_up = np.maximum(y_cal - q90, 0.0)
    out["cqr"] = np.zeros((2, H))
    for h in range(H):
        k = finite_q(1 - alpha, n_cal)
        out["cqr"][0, h] = np.sort(sc_low[:, h])[k - 1]
        out["cqr"][1, h] = np.sort(sc_up[:, h])[k - 1]
    return out


def adjust(method, q, Qs, alpha):
    """把 (N,3,H) 原始分位数（归一化 Δ）按方法调整，返回调整后 q10/q90（归一化 Δ）。

    保 p50；强制 q10≤p50≤q90 排序（供 CRPS 闭合形式）。
    """
    q10, q50, q90 = q[:, 0], q[:, 1], q[:, 2]
    if method == "raw":
        q10a, q90a = q10, q90
    elif method == "abs50":
        q10a = q50 - Qs[None, :]
        q90a = q50 + Qs[None, :]
    elif method == "symabs":
        q10a = q10 - Qs[None, :]
        q90a = q90 + Qs[None, :]
    elif method == "cqr":
        q10a = q10 - Qs[0, None, :]
        q90a = q90 + Qs[1, None, :]
    else:
        raise ValueError(method)
    q10a = np.minimum(q10a, q50)
    q90a = np.maximum(q90a, q50)
    return q10a, q50, q90a


def main():
    ap = argparse.ArgumentParser(description="F 共形校准（split conformal）覆盖率保证探索")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch")
    ap.add_argument("--out-json", default="exp/model_enhancement/f_conformal/results.json")
    args = ap.parse_args()

    t0 = time.time()
    print("== F 共形校准（split conformal）：让增量分位数区间覆盖率有统计保证 ==", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 校准 {CAL_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d；"
          f"GRU 骨干 + p10/p50/p90 分位数头 + M2/M4 多任务；目标 Δ={H}h", flush=True)
    print(f"   目标 α: {ALPHAS}（覆盖率 {[1 - a for a in ALPHAS]}）；方法: {METHODS}", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    if "conc_0.5" not in feat_cols:
        feat_cols = feat_cols + ["conc_0.5"]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象 + conc_t）", flush=True)

    days = TRAIN_DAYS + CAL_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    Nw = len(windows)
    print(f"[2] 滚动窗口 {Nw} 个（训练{CAL_DAYS + TEST_DAYS + TRAIN_DAYS}d + 步长{STRIDE_DAYS}d）", flush=True)

    # 聚合器：每 α × 每方法 × 每窗口
    def make_agg():
        return {
            "cov": np.zeros(Nw), "width": np.zeros(Nw), "crps": np.zeros(Nw),
            "cov_h": np.zeros((Nw, H)), "width_h": np.zeros((Nw, H)),
        }

    agg = {a: {m: make_agg() for m in METHODS} for a in ALPHAS}
    per_window = []
    rows = []

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        Xw, cur_raw, y_abs, seg, strat_w, warn_w = build_window(wide, i0, i1, feat_cols)
        n_tr = TRAIN_DAYS * GRID_PER_DAY - T          # 训练样本数（去掉 T 回看剪裁）
        n_cal = CAL_DAYS * GRID_PER_DAY               # 校准样本数
        n_te = int((seg == 2).sum())

        cal_sl = slice(n_tr, n_tr + n_cal)
        te_sl = slice(n_tr + n_cal, None)

        # Δ 目标（归一化）：scale 用训练段 Δ 的 std（防泄漏）
        delta_raw = y_abs - cur_raw[:, None]
        scale = float(np.std(delta_raw[:n_tr])) + 1e-8
        y_norm = (delta_raw / scale).astype(np.float32)

        model = train_model(Xw, y_norm, strat_w, warn_w, n_tr, args.epochs, args.device)

        # 校准集 + 测试集分位数（归一化 Δ）
        q_cal = predict_quantiles(model, Xw[cal_sl], args.device)   # (n_cal,3,H)
        q_te = predict_quantiles(model, Xw[te_sl], args.device)     # (n_te,3,H)
        y_cal_norm = y_norm[cal_sl]                                 # (n_cal,H)
        y_te_norm = y_norm[te_sl]                                   # (n_te,H)
        cur_te = cur_raw[te_sl]

        # 诊断：校准段自身的 raw 覆盖率（解释 conformal 膨胀机制——校准段欠/过覆盖）
        cal_cov = float(np.mean((y_cal_norm >= q_cal[:, 0]) & (y_cal_norm <= q_cal[:, 2])))
        cal_width = float(np.mean((q_cal[:, 2] - q_cal[:, 0]).mean(axis=0))) * scale

        # 校准（归一化 Δ 空间，每 α）
        cals = {a: calibrate(q_cal, y_cal_norm, a) for a in ALPHAS}

        # 测试集观测（conc 单位）
        obs = y_abs[te_sl]                                          # (n_te,H)
        n_test = len(obs)

        row = {"window": wi + 1, "start": str(st), "end": str(en),
               "n_train": int(n_tr), "n_cal": int(n_cal), "n_test": n_test,
               "sd_inc": round(scale, 3),
               "cal_cov_raw": round(cal_cov, 3), "cal_width_raw": round(cal_width, 2),
               "cur_med": round(float(np.median(cur_te)), 3),
               "y_med": round(float(np.median(obs)), 3)}

        for a in ALPHAS:
            for m in METHODS:
                if m == "raw":
                    q10a, q50, q90a = q_te[:, 0], q_te[:, 1], q_te[:, 2]
                else:
                    q10a, q50, q90a = adjust(m, q_te, cals[a][m], a)
                # 还原 conc 单位：cur + Δnorm * scale
                q10c = cur_te[:, None] + q10a * scale
                q90c = cur_te[:, None] + q90a * scale
                q50c = cur_te[:, None] + q50 * scale

                cov_h = np.mean((obs >= q10c) & (obs <= q90c), axis=0)   # (H,)
                cov = float(np.mean(cov_h))
                width_h = np.mean(q90c - q10c, axis=0)                    # (H,) conc 单位（Δ 尺度）
                crps_h = np.array([float(np.mean(crps_quantiles(
                    q10c[:, h], q50c[:, h], q90c[:, h], obs[:, h]))) for h in range(H)])

                agg[a][m]["cov"][wi] = cov
                agg[a][m]["cov_h"][wi] = cov_h
                agg[a][m]["width"][wi] = float(np.mean(width_h))
                agg[a][m]["width_h"][wi] = width_h
                agg[a][m]["crps"][wi] = float(np.mean(crps_h))
                row[f"{m}_cov_a{a}"] = round(cov, 4)
                row[f"{m}_w_a{a}"] = round(float(np.mean(width_h)), 3)
                row[f"{m}_crps_a{a}"] = round(float(np.mean(crps_h)), 4)

        raw_cov_a2 = row.get("raw_cov_a0.2", float("nan"))
        print(f"        校准段 raw覆盖={cal_cov:.3f}（宽 {cal_width:.2f}）| 测试段 raw覆盖={raw_cov_a2:.3f}  "
              f"raw宽={row.get('raw_w_a0.2', float('nan')):.2f}  rawCRPS={row.get('raw_crps_a0.2', float('nan')):.4f}",
              flush=True)
        for m in ("abs50", "symabs", "cqr"):
            print(f"        α=0.2 {m:<6}覆盖={row[f'{m}_cov_a0.2']:.3f} 宽={row[f'{m}_w_a0.2']:.2f}  "
                  f"CRPS={row[f'{m}_crps_a0.2']:.4f}  | α=0.1 覆盖={row[f'{m}_cov_a0.1']:.3f}  "
                  f"宽={row[f'{m}_w_a0.1']:.2f}", flush=True)

        rows.append(row)

    # ---- 聚合输出 ----
    print("\n===== 覆盖率均值（跨窗口）/ 窗口 std / 区间宽 / CRPS =====", flush=True)
    for a in ALPHAS:
        print(f"\n  α={a}（目标覆盖率 {1 - a}）", flush=True)
        hdr = f"  {'方法':<8}{'覆盖':<8}{'覆盖std':<10}{'区间宽':<9}{'CRPS':<9}{'CRPS vs raw'}"
        print(hdr, flush=True)
        for m in METHODS:
            c = agg[a][m]["cov"]
            cov_std = float(np.std(c))
            w = float(np.mean(agg[a][m]["width"]))
            cr = float(np.mean(agg[a][m]["crps"]))
            rel = (agg[a]["raw"]["crps"].mean() - cr) / agg[a]["raw"]["crps"].mean() * 100
            print(f"  {m:<8}{c.mean():<8.3f}{cov_std:<10.3f}{w:<9.2f}{cr:<9.4f}{rel:+.1f}%", flush=True)

    print("\n===== 逐视界覆盖率（α=0.2）=====", flush=True)
    print("  h:" + "".join(f"{h + 1:>9}" for h in range(H)), flush=True)
    for m in METHODS:
        row_cov = agg[0.2][m]["cov_h"].mean(axis=0)
        print(f"  {m:<3}" + "".join(f"{v:>9.3f}" for v in row_cov), flush=True)

    print("\n===== 逐窗口覆盖率（α=0.2，raw vs 各方法）=====", flush=True)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    # ---- 输出 JSON ----
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "cal_days": CAL_DAYS, "test_days": TEST_DAYS,
                     "stride_days": STRIDE_DAYS, "T": T, "H": H, "epochs": args.epochs,
                     "n_windows": Nw, "alphas": ALPHAS, "methods": METHODS,
                     "finite_sample": "Q = ⌈(n+1)(1-α)⌉ 阶统计量"},
        "alphas": {},
        "windows": rows,
    }
    for a in ALPHAS:
        res["alphas"][f"a{a}"] = {}
        for m in METHODS:
            c = agg[a][m]
            res["alphas"][f"a{a}"][m] = {
                "coverage_mean": float(c["cov"].mean()),
                "coverage_std": float(np.std(c["cov"])),
                "coverage_windows": c["cov"].tolist(),
                "coverage_h": c["cov_h"].mean(axis=0).tolist(),
                "width_mean": float(c["width"].mean()),
                "width_h": c["width_h"].mean(axis=0).tolist(),
                "crps_mean": float(c["crps"].mean()),
            }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
