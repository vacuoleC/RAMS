# -*- coding: utf-8 -*-
"""L 探索：采样尺度 vs 藻类过程尺度（3h / 12h / 1D 日级）

背景（已证实）：
  - 当前模型 3h 采样：T=24（3 天回看）→ H=8（24h 视界）
  - M5 证明藻类过程是天级到周级：翻倍 1-3 天、降水滞后 13-20 天、风 20-29 天
  - 假设：3h 对天级过程是过采样——相邻 3h 高度相关（3h 网格 lag1 自相关≈0.90），
    T=24 窗口只装 3 天，装不下天级过程完整周期；H=8 只覆盖不到一次翻倍
  - 假设：降采样到日级 + 拉长视界，可能更匹配过程尺度，气象信号可能重新浮现
    （D 证伪 wind_u 是在 3h 尺度 + 24h 视界）

设计（同一增量 abs_delta 协议，唯一差异 = 采样尺度 / 回看 / 视界）：
  尺度     网格   聚合        T(回看)   H(视界)   视界实际时长
  3h 基线  3h     原网格       24        8         24h    （复现 B7/D base）
  12h      12h    均值         24        4         48h    （M5 用过的 12h 网格）
  1D       1D     均值         30        7         168h   （主候选：7 天预警提前量）
  1D_wind  1D     均值         30        7         168h   + wind_u 通道（气象信号是否在日级浮现）

公平：各尺度滚动窗口按实际天对齐——同一起始日历日，训练 730 天 / 测试 90 天 / 步长 45 天，
17 窗口。同尺度内按"预测末端日期 < 训练截止日期"切分训练/测试（训练量按实际天数一致）。
3 seed（0/1/2）。

评估（全部还原 conc 单位，同一日历测试段）：
  a. 每视界 CRPS（p10/p50/p90 闭合形式，与 B7/D 一致）
  b. 每视界 p50 RMSE；区间覆盖率 [p10,p90]
  c. 各自持久化技能 skill = (CRPS_persist - CRPS_model)/CRPS_persist
     —— 判断哪个尺度模型相对持久化最强（核心问题 3）
  d. 相同实际提前量的跨尺度对照（24h 提前：3h h=8 / 12h h=2 / 1D h=1）
  e. wind_u 是否在日级浮现（1D vs 1D_wind）

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

EPOCHS = 30
SEEDS = [0, 1, 2]

# 滚动窗口参数（天）——所有尺度按实际天数对齐
TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
N_WINDOWS = 17

# 各尺度配置：网格小时 / 回看 T / 视界 H
SCALES = {
    "3h":   {"step_h": 3,  "T": 24, "H": 8,  "wind": False},
    "12h":  {"step_h": 12, "T": 24, "H": 4,  "wind": False},
    "1D":   {"step_h": 24, "T": 30, "H": 7,  "wind": False},
    "1D_wind": {"step_h": 24, "T": 30, "H": 7, "wind": True},
}
ORDER = ["3h", "12h", "1D", "1D_wind"]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 B7/D 一致实现）。"""
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


def load_wide_resampled(parquet, step_h, add_wind):
    """3h 宽表 → 可选加 wind_u（3h 网格上逐时刻算，与 D/M5 一致）→ 可选降采样（均值）。

    step_h=3 时返回原 3h 网格；>3 时 resample(f'{step_h}h').mean().dropna()。
    wind_u = wind_speed * cos(deg2rad(wind_dir))，在 3h 上先算再聚合（保持矢量均值，同 M5）。
    """
    loader = TensorBuilder(TensorConfig())
    wide = loader._load_wide(Path(parquet)).sort_index()
    if add_wind:
        dir_rad = np.deg2rad(wide["wind_dir"].values)
        wide["wind_u"] = wide["wind_speed"].values * np.cos(dir_rad)
    if step_h != 3:
        wide = wide.resample(f"{step_h}h").mean().dropna()
    return wide


def base_feat_cols(wide, add_wind):
    """现输入特征列：全剖面水温 + 气象 + conc_t（B7/D 口径），可加 wind_u。"""
    cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    cols = [c for c in cols if c in wide.columns]
    if "conc_0.5" not in cols:
        cols = cols + ["conc_0.5"]
    if add_wind and "wind_u" in wide.columns:
        cols = cols + ["wind_u"]
    return cols


def build_window(wide, start_ts, tr_ts, end_ts, T, H, feat_cols, strat_col="delta_T"):
    """构造一个窗口（按实际日期切片），训练/测试按"预测末端日期"切分（天数一致）。

    Returns:
      Xw       (n_w, T, F) 标准化特征窗口（训练段统计）
      cur_raw  (n_w,) conc_t 原始尺度
      y_abs    (n_w, H) conc_{t+h} 原始尺度
      strat_w, warn_w  M2/M4 标签
      n_tr    训练样本数（预测末端 < tr_ts）
    """
    df = wide[(wide.index >= start_ts) & (wide.index < end_ts)]
    idx = df.index
    n = len(df)
    n_tr_rows = int((idx < tr_ts).sum())          # 训练段行数（按日期）

    # 特征标准化（只用训练段行）
    Xtr = df[feat_cols].values[:n_tr_rows].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((df[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    y_raw = df["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    # 训练样本数：预测末端索引 i+T+H-1 < n_tr_rows（在训练日期内）
    n_tr = max(0, n_tr_rows - T - H + 1)

    # M2 分层标签（复用 B7/D）
    delta = df[strat_col].values
    thr = float(np.median(delta[:n_tr_rows]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（复用 B7/D）：训练样本的 y_abs 峰值定阈值
    warn_val = y_abs.max(axis=1)
    qs = np.quantile(warn_val[:n_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, cur_raw, y_abs, strat_w, warn_w, n_tr


def make_abs_delta_targets(cur_raw, y_abs):
    """增量 abs_delta 目标（与 B7/D 一致）：Δ = conc_{t+h} - conc_t。"""
    return y_abs - cur_raw[:, None], "add"


def train_model(Xw, yw, strat_w, warn_w, n_tr, epochs, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RamsNet(feat_dim=Xw.shape[2], horizon=yw.shape[1], use_m4=True)
    trainer = Trainer(model, device=device)
    trainer.fit(Xw[:n_tr], yw[:n_tr], strat_w[:n_tr],
                Xw[n_tr:], yw[n_tr:], strat_w[n_tr:],
                warn_tr=warn_w[:n_tr], warn_va=warn_w[n_tr:],
                epochs=epochs, batch_size=128)
    return model


def predict_quantiles(model, X_te, device):
    model.eval()
    H = model.horizon
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def back_to_conc(cur_raw, q, scale):
    """abs_delta 还原：conc = conc_t + Δ。q: (N,3,H) 归一化 → (N,3,H) conc 单位。"""
    qc = q * scale
    return cur_raw[:, None, None] + qc  # (N,1,1)+(N,3,H)


def persistent_conc(cur_raw, scale, H):
    """abs_delta 持久化：目标=0 → conc_{t+h}=conc_t。"""
    N = len(cur_raw)
    q0 = np.zeros((N, 3, H), dtype=np.float64)
    return back_to_conc(cur_raw, q0, scale)


def main():
    ap = argparse.ArgumentParser(description="L: 采样尺度对比（3h/12h/1D，abs_delta 增量协议）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 2 epoch × 1 seed")
    ap.add_argument("--scales", default=",".join(ORDER))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--out-json", default="exp/model_enhancement/l_daily_scale/results.json")
    args = ap.parse_args()

    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    print(f"== L: 采样尺度对比（{len(scales)} 尺度 × {len(seeds)} seed）==", flush=True)
    for s in scales:
        c = SCALES[s]
        print(f"   {s:<8} step={c['step_h']}h  T={c['T']}  H={c['H']}  "
              f"视界={c['H'] * c['step_h']}h  wind={'Y' if c['wind'] else 'N'}", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d，{N_WINDOWS} 窗口"
          f"（按日历日对齐）；增量 abs_delta；RamsNet 分位数 + M2/M4 多任务", flush=True)

    # 加载 3h 原始宽表，确定窗口锚点
    wide_raw = load_wide_resampled(args.parquet, 3, False)
    d0 = wide_raw.index.min()
    print(f"[1] 3h 宽表 {len(wide_raw)} 时刻 {d0:%Y-%m-%d} → {wide_raw.index.max():%Y-%m-%d}", flush=True)

    # 每尺度加载（缓存）
    wide_cache = {}
    for s in scales:
        c = SCALES[s]
        w = load_wide_resampled(args.parquet, c["step_h"], c["wind"])
        wide_cache[s] = w
        print(f"    [{s}] {len(w)} 时刻，特征 {len(base_feat_cols(w, c['wind']))} 列"
              f"{'（含 wind_u）' if c['wind'] else ''}", flush=True)

    # 窗口（按日历日对齐，17 窗口）
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
    print(f"[2] 滚动窗口 {len(windows)} 个（起始 {windows[0][0]:%Y-%m-%d}，"
          f"训练截止 {windows[0][1]:%Y-%m-%d}，测试结束 {windows[0][2]:%Y-%m-%d}）", flush=True)

    Nw = len(windows)
    # 聚合器：每尺度每 seed 每窗口
    agg = {s: {
        "cov": np.zeros((Nw, len(seeds))),
        "crps_h": np.zeros((Nw, len(seeds), SCALES[s]["H"])),
        "crps": np.zeros((Nw, len(seeds))),
        "crps_p": np.zeros((Nw, len(seeds))),
        "rmse": np.zeros((Nw, len(seeds))),
        "rmse_h": np.zeros((Nw, len(seeds), SCALES[s]["H"])),
    } for s in scales}
    per_window = []  # 每窗口元信息

    for wi, (start_ts, tr_ts, end_ts) in enumerate(windows):
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {start_ts:%Y-%m-%d} → {end_ts:%Y-%m-%d}  "
              f"(训练 {start_ts:%Y-%m-%d}→{tr_ts:%Y-%m-%d}，测试 {tr_ts:%Y-%m-%d}→{end_ts:%Y-%m-%d})",
              flush=True)

        win_info = {"window": wi + 1, "start": str(start_ts), "tr": str(tr_ts),
                    "end": str(end_ts), "n_test": {}}

        for s in scales:
            c = SCALES[s]
            w = wide_cache[s]
            T, H = c["T"], c["H"]
            feat_cols = base_feat_cols(w, c["wind"])
            Xw, cur_raw, y_abs, strat_w, warn_w, n_tr = build_window(
                w, start_ts, tr_ts, end_ts, T, H, feat_cols)
            te_sl = slice(n_tr, len(Xw))

            Xte = Xw[te_sl]
            cur_te = cur_raw[te_sl]
            y_te = y_abs[te_sl]                     # (N,H) 原始 conc 观测
            Nte = len(Xte)
            win_info["n_test"][s] = int(Nte)

            raw, kind = make_abs_delta_targets(cur_raw, y_abs)
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
                q_p = persistent_conc(cur_te, scale, H)
                crps_p = float(np.mean([
                    np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 1, h],
                                           q_p[:, 2, h], obs[:, h])) for h in range(H)]))
                # p50 RMSE（conc 单位）
                rmse = float(np.sqrt(np.mean((q_conc[:, 1] - obs) ** 2)))
                rmse_h = [float(np.sqrt(np.mean((q_conc[:, 1, h] - obs[:, h]) ** 2)))
                          for h in range(H)]

                agg[s]["cov"][wi, si] = cov
                agg[s]["crps_h"][wi, si] = crps_h
                agg[s]["crps"][wi, si] = crps_avg
                agg[s]["crps_p"][wi, si] = crps_p
                agg[s]["rmse"][wi, si] = rmse
                agg[s]["rmse_h"][wi, si] = rmse_h

            skill = ((agg[s]["crps_p"][wi].mean() - agg[s]["crps"][wi].mean())
                     / agg[s]["crps_p"][wi].mean())
            print(f"   [{s}] T={T} H={H} n_test={Nte}  "
                  f"CRPS={agg[s]['crps'][wi].mean():.4f}±{agg[s]['crps'][wi].std():.4f} "
                  f"(持久化 {agg[s]['crps_p'][wi].mean():.4f}, 技能 {skill * 100:+.1f}%)  "
                  f"p50RMSE={agg[s]['rmse'][wi].mean():.3f}  覆盖={agg[s]['cov'][wi].mean():.3f}",
                  flush=True)

        per_window.append(win_info)

    # ---- 聚合输出 ----
    print("\n===== 各尺度对照（3 seed 均值，全部还原 conc 单位）=====", flush=True)
    print(f"  {'尺度':<9}{'视界':<7}{'CRPS':<10}{'持久化CRPS':<12}{'技能%':<10}"
          f"{'p50RMSE':<10}{'覆盖':<8}", flush=True)
    for s in scales:
        c = SCALES[s]
        a = agg[s]
        cp = a["crps_p"].mean()
        rel = (cp - a["crps"].mean()) / cp * 100 if cp else float("nan")
        print(f"  {s:<9}{c['H'] * c['step_h']}h   {a['crps'].mean():<10.4f}{cp:<12.4f}"
              f"{rel:<+10.1f}{a['rmse'].mean():<10.3f}{a['cov'].mean():<8.3f}", flush=True)

    print("\n===== 相同实际提前量对照（24h 提前：3h h8 / 12h h2 / 1D h1）=====", flush=True)
    lead_map = {"3h": 8, "12h": 2, "1D": 1, "1D_wind": 1}
    print(f"  {'尺度':<9}{'h':<5}{'CRPS@24h':<12}{'RMSE@24h':<12}{'覆盖@24h':<10}", flush=True)
    for s in scales:
        if s not in lead_map or lead_map[s] - 1 >= SCALES[s]["H"]:
            continue
        h = lead_map[s] - 1
        a = agg[s]
        crps24 = float(a["crps_h"][..., h].mean())
        rmse24 = float(a["rmse_h"][..., h].mean())
        print(f"  {s:<9}{lead_map[s]:<5}{crps24:<12.4f}{rmse24:<12.3f}", flush=True)

    print("\n===== 每视界 CRPS（3 seed 均值，还原 conc 单位）=====", flush=True)
    for s in scales:
        c = SCALES[s]
        a = agg[s]
        row = f"  {s:<9} 视界时长的步数 H={c['H']}: "
        row += " ".join(f"h{i + 1}({c['step_h'] * (i + 1)}h)={a['crps_h'][..., i].mean():.4f}"
                        for i in range(c["H"]))
        print(row, flush=True)

    print("\n===== 逐窗口 CRPS（3 seed 均值）=====", flush=True)
    hdr = "  w" + "".join(f"{s:>12}" for s in scales)
    print(hdr, flush=True)
    for wi in range(Nw):
        row = f"{wi + 1:>3}"
        for s in scales:
            row += f"{agg[s]['crps'][wi].mean():>12.4f}"
        print(row, flush=True)

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {
            "train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
            "n_windows": Nw, "seeds": seeds, "epochs": args.epochs,
            "target": "abs_delta (Δ=conc_{t+h}-conc_t)", "evaluate": "还原 conc 单位",
            "note": "窗口按日历日对齐，各尺度训练量/测试量按实际天数一致；"
                    "1D/12h 聚合用均值；wind_u 在 3h 上逐时刻算后聚合（同 M5）",
        },
        "scales": {},
        "windows": per_window,
    }
    for s in scales:
        c = SCALES[s]
        a = agg[s]
        cp = float(a["crps_p"].mean())
        res["scales"][s] = {
            "step_h": c["step_h"], "T": c["T"], "H": c["H"],
            "horizon_hours": c["H"] * c["step_h"], "wind": c["wind"],
            "crps_mean": float(a["crps"].mean()),
            "crps_std_windows": float(a["crps"].mean(axis=1).std()),
            "crps_h": a["crps_h"].mean(axis=0).mean(axis=0).tolist(),
            "crps_persist": cp,
            "skill_vs_persist_pct": (cp - float(a["crps"].mean())) / cp * 100.0,
            "rmse_mean": float(a["rmse"].mean()),
            "rmse_h": a["rmse_h"].mean(axis=0).mean(axis=0).tolist(),
            "coverage_mean": float(a["cov"].mean()),
            "crps_windows": a["crps"].mean(axis=1).tolist(),
        }
        if s in lead_map and lead_map[s] - 1 < c["H"]:
            h = lead_map[s] - 1
            res["scales"][s]["crps_at_24h"] = float(a["crps_h"][..., h].mean())
            res["scales"][s]["rmse_at_24h"] = float(a["rmse_h"][..., h].mean())
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
