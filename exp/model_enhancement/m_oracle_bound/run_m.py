# -*- coding: utf-8 -*-
"""M 方向：oracle 信息上限探针（藻类浓度的可预测上限在哪？我们接近了吗？）

世界观验证（核心问题）：
  A（接近上限）：数据可预测信息有限，继续优化收益递减
  B（没接近）  ：数据有信号但方法没提取到
  C（中间态）  ：平稳期接近上限，暴发期有空间但被评估掩盖

方法：oracle 上限探针——给模型输入"未来 N 天的真实 conc 值"（正常推理不可能有），
看 CRPS 能降到多少。该数字是数据在当前特征集下的可预测上限（诊断工具，非部署模型）。

设计（关键防退化决策，详见 rethinking.md）：
  协议复用 A/B7：训练 730d/测试 90d/步长 45d，17 窗口，3 seed，增量 abs_delta 目标
  （归一化窗口训练段 scale 防泄漏，还原 conc 单位评估）。
  **探针视界 H=64（8 天，3h 网格）**：若 H=8（1 天），oracle N≥1 天会完整覆盖目标窗 →
  所有 oracle 档 CRPS≈0（退化，无信息量）。H=64 使 oracle 只覆盖前 N 天，后 8−N 天是
  "诚实"预报区，N=1/3/7 才有可区分的读数，且能测"轨迹信息随视界的衰减"。

oracle 注入（保证信息上限被充分使用，避免低估）：
  1. 未来 N 天真实 conc（标准化）作为额外输入通道，重复到全部 T 时刻（骨干 GRU 可见轨迹）；
  2. 同一 oracle 通道末时刻值直接拼接到 M1 头输入（hidden ⊕ oracle）——已知段（h≤M）的
     "复制"变为近线性操作，模型能近乎完美利用 oracle → 收紧信息上限估计。

arm 定义（全部同一 17 窗口协议，3 seed；base/ar_only 用 A/B7 同款 RamsNet 保证可比锚定）：
  base8     H=8  全特征（temp_*+气象+conc 历史）无 oracle   —— 锚定公开基线 CRPS≈0.86
  base      H=64 全特征 无 oracle                            —— 当前模型扩展到 8 天
  ar_only   H=64 仅 conc 历史（无 temp/气象）               —— 强自回归上限（隔离 AR 信息）
  oracle_1  H=64 全特征 + 未来 1 天真实 conc（8 步，通道+头注入）
  oracle_3  H=64 全特征 + 未来 3 天真实 conc（24 步，通道+头注入）
  oracle_7  H=64 全特征 + 未来 7 天真实 conc（56 步，通道+头注入）

评估（全部还原 conc 单位，逐视界 + 按天聚合 d=1..8）：
  a. 每视界 CRPS（分位数闭合形式，与 A/B7 一致）+ p50 RMSE + 覆盖率
  b. 关键数字（世界观判定）：
     - **1-day-ahead 阶梯**（同难度、不同已知轨迹长度）：base_d1（已知 0 天）
       vs oracle_1_d2（已知 1 天）/ oracle_3_d4（已知 3 天）/ oracle_7_d8（已知 7 天）
       → 每多知道 1 天真实轨迹，1 天前预报能降多少 = 轨迹信号量
     - base_d8 vs oracle_7_d8（固定目标第 8 天，已知轨迹 0 vs 7 天）→ 不可桥接固有噪声
     - base vs ar_only（temp/气象是否贡献 = 非自回归信号量）
     - 逐窗口 base−oracle 差距 vs 浓度波动（World C 检验）
  c. oracle 启动持久化基线（h≤M 真实=完美，h>M 保持最后已知值）：oracle 臂下界参照
3 seed 报告均值±std。保密：只输出聚合统计量，不打印原始数据行。
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
import torch.nn as nn  # noqa: E402

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402
from rams.models.rams_net import RamsNet, SharedGRU, M2Head, M4Head  # noqa: E402
from rams.training.trainer import Trainer  # noqa: E402

T = 24                # 回看 3 天（3h 网格）
H8 = 8                # 锚定臂视界（1 天，与 A/B7 一致）
H64 = 64              # 探针臂视界（8 天）
EPOCHS = 30
SEEDS = [0, 1, 2]

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）
TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
GRID_PER_DAY = 8

# 多任务权重（与 A/B7 口径一致：w=1/3/2）
W_M1, W_M2, W_M4 = 1.0, 3.0, 2.0


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
    loader = TensorBuilder(TensorConfig(T=T, H=H8))
    wide = loader._load_wide(Path(parquet))
    return wide.sort_index()


def build_window(wide, i0, i1, feat_cols, oracle_M, H):
    """窗口 [i0,i1)：特征（训练段标准化）+ oracle 通道 + 原始 conc + 增量目标 + M2/M4 标签。

    oracle_M>0 时，把未来 M 步真实 conc（训练段 conc 统计标准化）作为额外特征通道
    追加在特征末尾，并重复到全部 T 时刻（骨干可见轨迹；头注入取末时刻值）。
    Returns:
      Xw       (n_w, T, F+oracle_M) 标准化特征（训练段统计，防泄漏）
      y_abs    (n_w, H)  conc_{t+1..t+H} 原始尺度
      cur_raw  (n_w,)    conc_t 原始尺度
      strat_w, warn_w    M2/M4 多任务标签
    """
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))

    Xtr = df[feat_cols].values[:n_tr].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((df[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    y_raw = df["conc_0.5"].values.astype(np.float64)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    if oracle_M > 0:
        c_mu = float(np.mean(y_raw[:n_tr]))
        c_sd = float(np.std(y_raw[:n_tr])) + 1e-8
        o = np.stack([y_raw[i + T:i + T + oracle_M] for i in range(n_w)])  # (n_w, M)
        o = ((o - c_mu) / c_sd).astype(np.float32)
        extra = np.repeat(o[:, None, :], T, axis=1)  # (n_w, T, M)
        Xw = np.concatenate([Xw, extra], axis=2)

    # M2 分层标签（A/B7 复用：delta_T 训练段中位数阈值）
    delta = df["delta_T"].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（A/B7 复用：未来 H 步浓度峰值 + 训练段分位数阈值）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, y_abs, cur_raw, strat_w, warn_w


class OracleRamsNet(nn.Module):
    """探针专用：RamsNet + oracle 头注入（oracle_M>0 时用，仅诊断，非部署）。

    骨干 GRU 与 RamsNet 相同（可见全部通道含 oracle 轨迹）；M1 头输入 = hidden ⊕
    oracle 末时刻值（近线性可复制已知段 → 收紧信息上限估计）；M2/M4 复用 M2Head/M4Head。
    forward 返回 (m1, m2, m4)，接口与 RamsNet 一致（Trainer 可直接用）。
    """

    def __init__(self, feat_dim, horizon, oracle_M, hidden=64, n_layers=1,
                 dropout=0.0, n_classes=2, n_levels=4, use_m4=True):
        super().__init__()
        self.oracle_M = oracle_M
        self.horizon = horizon
        self.quantile = True
        self.use_m4 = use_m4
        self.backbone = SharedGRU(feat_dim, hidden, n_layers, dropout)
        self.m1 = nn.Sequential(
            nn.Linear(hidden + oracle_M, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * 3),
        )
        self.m2 = M2Head(hidden, n_classes)
        self.m4 = M4Head(hidden, n_levels) if use_m4 else None

    def forward(self, x):
        # x: (B, T, F+M)，最后 M 列为 oracle 通道
        h = self.backbone(x)
        oracle_last = x[:, -1, -self.oracle_M:]      # (B, M)
        h_flat = torch.cat([h, oracle_last], dim=1)
        m1 = self.m1(h_flat)
        m2 = self.m2(h)
        m4 = self.m4(h) if self.m4 is not None else None
        return m1, m2, m4

    def predict_mean(self, m1_out):
        """与 RamsNet 接口一致（Trainer 验证用）：取中位数通道。"""
        H = self.horizon
        return m1_out[:, H:2 * H]

    def predict_interval(self, m1_out):
        """与 RamsNet 接口一致：p10/p90 区间。"""
        H = self.horizon
        return m1_out[:, :H], m1_out[:, 2 * H:]


def train_model(Xw, y_norm, strat_w, warn_w, n_tr, H, oracle_M, epochs, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if oracle_M > 0:
        model = OracleRamsNet(feat_dim=Xw.shape[2], horizon=H, oracle_M=oracle_M,
                              use_m4=True)
    else:
        model = RamsNet(feat_dim=Xw.shape[2], horizon=H, use_m4=True)
    trainer = Trainer(model, device=device, w_m1=W_M1, w_m2=W_M2, w_m4=W_M4)
    trainer.fit(Xw[:n_tr], y_norm[:n_tr], strat_w[:n_tr],
                Xw[n_tr:], y_norm[n_tr:], strat_w[n_tr:],
                warn_tr=warn_w[:n_tr], warn_va=warn_w[n_tr:],
                epochs=epochs, batch_size=128)
    return model


def predict_quantiles(model, X_te, device):
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
        q = torch.stack([m1[:, :model.horizon], m1[:, model.horizon:2 * model.horizon],
                         m1[:, 2 * model.horizon:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def day_agg(crps_h, H):
    """把逐视界 CRPS（H 步）按天聚合（1 天 = 8 步）。返回 (ndays,) + 平均。"""
    nd = H // GRID_PER_DAY
    day = np.array([np.mean(crps_h[GRID_PER_DAY * d:GRID_PER_DAY * (d + 1)])
                    for d in range(nd)])
    return day, float(np.mean(day))


def main():
    ap = argparse.ArgumentParser(description="M 方向：oracle 信息上限探针（3-seed）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 1 seed × 2 epoch")
    ap.add_argument("--out-json", default="exp/model_enhancement/m_oracle_bound/results.json")
    args = ap.parse_args()

    # arm 配置：H（视界）、feat（特征子集）、oracle_M（未来步数）
    ARMS = {
        "base8":    {"H": H8,  "feat": "full", "M": 0},
        "base":     {"H": H64, "feat": "full", "M": 0},
        "ar_only":  {"H": H64, "feat": "conc", "M": 0},
        "oracle_1": {"H": H64, "feat": "full", "M": 8},
        "oracle_3": {"H": H64, "feat": "full", "M": 24},
        "oracle_7": {"H": H64, "feat": "full", "M": 56},
    }

    t0 = time.time()
    print("== M 方向：oracle 信息上限探针（3-seed，同一 17 窗口协议）==", flush=True)
    print(f"   H8={H8}（锚定基线） H64={H64}（探针 8 天）；seeds={SEEDS} epochs={args.epochs}", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    feat_full = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_full = [c for c in feat_full if c in wide.columns]
    if "conc_0.5" not in feat_full:
        feat_full = feat_full + ["conc_0.5"]
    feat_conc = ["conc_0.5"]
    print(f"   全特征 {len(feat_full)} 列；conc 特征 {len(feat_conc)} 列", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    SEEDS_ = list(SEEDS)
    if args.smoke:
        windows = windows[:1]
        args.epochs = min(args.epochs, 2)
        SEEDS_ = [0]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    Nw = len(windows)
    print(f"[2] 滚动窗口 {Nw} 个（训练 {TRAIN_DAYS}d + 测试 {TEST_DAYS}d）；seed={SEEDS_}", flush=True)

    # 聚合器：arm × seed → 逐窗口数组
    agg = {a: {s: {
        "crps_h": np.zeros((Nw, ARMS[a]["H"])), "rmse_h": np.zeros((Nw, ARMS[a]["H"])),
        "cover_h": np.zeros((Nw, ARMS[a]["H"])), "crps": np.zeros(Nw),
        "day_crps": np.zeros((Nw, ARMS[a]["H"] // GRID_PER_DAY)),
        "crps_p": np.zeros(Nw),       # 标准持久化（conc_t）
        "crps_op": np.zeros(Nw),      # oracle 启动持久化（h≤M 真实，h>M 最后已知）
    } for s in SEEDS_} for a in ARMS}
    win_stats = []    # 逐窗口 conc 分布（World C 分析）

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        win_info = {"window": wi + 1, "start": str(st), "end": str(en)}
        for a in ARMS:
            cfg = ARMS[a]
            feat = feat_full if cfg["feat"] == "full" else feat_conc
            H = cfg["H"]; M = cfg["M"]
            Xw, y_abs, cur_raw, strat_w, warn_w = build_window(
                wide, i0, i1, feat, M, H)
            n_win = len(Xw)
            n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
            te_sl = slice(n_tr, n_win)
            Xte = Xw[te_sl]; cur_te = cur_raw[te_sl]; obs = y_abs[te_sl]
            Nte = len(Xte)

            # 增量目标（窗口训练段 scale 防泄漏）
            delta_raw = y_abs - cur_raw[:, None]
            sd_inc = float(np.std(delta_raw[:n_tr])) + 1e-8
            y_inc = (delta_raw / sd_inc).astype(np.float32)

            # 标准持久化（Δ≡0 → conc_{t+h}=conc_t）
            q_p = np.repeat(cur_te[:, None, None], H, axis=2).astype(np.float64)
            crps_p_h = np.array([np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 0, h],
                                                        q_p[:, 0, h], obs[:, h]))
                                 for h in range(H)])
            crps_p = float(crps_p_h.mean())
            # oracle 启动持久化（h≤M 真实值=完美；h>M 保持最后已知 conc_{t+M}）
            crps_op = float("nan")
            if M > 0:
                q_op = np.empty_like(obs)
                q_op[:, :M] = obs[:, :M]
                q_op[:, M:] = obs[:, M - 1:M]
                crps_op = float(np.mean(
                    [np.mean(crps_quantiles(q_op[:, h], q_op[:, h], q_op[:, h], obs[:, h]))
                     for h in range(H)]))

            for s in SEEDS_:
                model = train_model(Xw, y_inc, strat_w, warn_w, n_tr, H, M,
                                    args.epochs, args.device, s)
                q_norm = predict_quantiles(model, Xte, args.device)        # (N,3,H)
                q_conc = cur_te[:, None, None] + q_norm * sd_inc           # conc 单位

                crps_h = np.array([np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                          q_conc[:, 2, h], obs[:, h]))
                                   for h in range(H)])
                rmse_h = np.array([np.sqrt(np.mean((q_conc[:, 1, h] - obs[:, h]) ** 2))
                                   for h in range(H)])
                cover_h = np.array([np.mean((obs[:, h] >= q_conc[:, 0, h])
                                            & (obs[:, h] <= q_conc[:, 2, h]))
                                    for h in range(H)])
                day_c, _ = day_agg(crps_h, H)

                g = agg[a][s]
                g["crps_h"][wi] = crps_h
                g["rmse_h"][wi] = rmse_h
                g["cover_h"][wi] = cover_h
                g["crps"][wi] = float(np.mean(crps_h))
                g["day_crps"][wi] = day_c
                g["crps_p"][wi] = crps_p
                g["crps_op"][wi] = crps_op

            # 窗口摘要（seed 均值）
            cm = float(np.mean([agg[a][s]["crps"][wi] for s in SEEDS_]))
            c1 = float(np.mean([agg[a][s]["day_crps"][wi][0] for s in SEEDS_]))
            op_str = f"{crps_op:.4f}" if M > 0 else "  n/a "
            print(f"    {a:<9} H={H:<3} M={M:<3} CRPS_all={cm:.4f}  CRPS_d1={c1:.4f}  "
                  f"(持久化 {crps_p:.4f}  orac_persist {op_str})",
                  flush=True)

            if a == "base":
                win_info["conc_p50"] = float(np.median(cur_te))
                win_info["conc_range"] = float(np.quantile(cur_te, 0.90)
                                               - np.quantile(cur_te, 0.10))
                win_info["conc_std"] = float(np.std(cur_te))
        win_stats.append(win_info)

    # ---- 聚合输出 ----
    def arm_mean_std(key, arm, s, h_idx=None):
        vals = [agg[arm][s][key][wi] for s in SEEDS_ for wi in range(Nw)]
        if h_idx is not None:
            vals = [v[h_idx] for v in vals if h_idx < len(v)]
        return float(np.mean(vals)), float(np.std(vals))

    print("\n===== 全窗口平均 CRPS（3-seed 均值±std，17 窗口）=====", flush=True)
    print(f"  {'arm':<10}{'CRPS_all':<20}{'CRPS_d1':<20}{'CRPS_d8':<20}{'RMSE_d1':<12}",
          flush=True)
    summary = {}
    for a in ARMS:
        H = ARMS[a]["H"]
        m_all, sd_all = arm_mean_std("crps", a, SEEDS_[0])
        m_d1, sd_d1 = arm_mean_std("day_crps", a, SEEDS_[0], 0)
        m_d8, sd_d8 = arm_mean_std("day_crps", a, SEEDS_[0], 7)
        rmse_d1, _ = arm_mean_std("rmse_h", a, SEEDS_[0], h_idx=0)
        crps_p = float(np.mean([agg[a][s]["crps_p"][wi] for s in SEEDS_ for wi in range(Nw)]))
        op_vals = [agg[a][s]["crps_op"][wi]
                   for s in SEEDS_ for wi in range(Nw)
                   if not np.isnan(agg[a][s]["crps_op"][wi])]
        crps_op = float(np.mean(op_vals)) if op_vals else float("nan")
        skill_p = (crps_p - m_all) / crps_p * 100 if crps_p else 0
        print(f"  {a:<10}{m_all:<10.4f}±{sd_all:<7.4f}{m_d1:<10.4f}±{sd_d1:<7.4f}"
              f"{m_d8:<10.4f}±{sd_d8:<7.4f}{rmse_d1:<12.4f}  persist={crps_p:.4f} "
              f"skill={skill_p:+.1f}%  opersist={crps_op:.4f}", flush=True)
        summary[a] = {
            "crps_all": round(m_all, 4), "crps_all_std": round(sd_all, 4),
            "crps_day1": round(m_d1, 4), "crps_day1_std": round(sd_d1, 4),
            "crps_day8": round(m_d8, 4), "crps_day8_std": round(sd_d8, 4),
            "rmse_day1_h1": round(rmse_d1, 4),
            "crps_persist": round(crps_p, 4),
            "skill_vs_persist_pct": round(skill_p, 2),
            "crps_oracle_persist": round(crps_op, 4),
        }

    print("\n===== 按天 CRPS（d=1..8，3-seed 均值，17 窗口平均）=====", flush=True)
    print(f"  {'arm':<10}" + "".join([f"{'d'+str(d+1):>10}" for d in range(8)]), flush=True)
    day_tbl = {}
    for a in ARMS:
        H = ARMS[a]["H"]
        nd = H // GRID_PER_DAY
        row = []
        for d in range(8):
            if d < nd:
                m, _ = arm_mean_std("day_crps", a, SEEDS_[0], d)
            else:
                m = float("nan")
            row.append(m)
        day_tbl[a] = row
        print(f"  {a:<10}" + "".join([f"{v:>10.4f}" for v in row]), flush=True)

    # ---- 世界观判定（关键数字）----
    print("\n===== 世界观判定 =====", flush=True)
    base_d1 = day_tbl["base"][0]
    base_d8 = day_tbl["base"][7]
    # 1-day-ahead 阶梯：已知 N 天真实轨迹后做 1 天前预报（同难度，不同已知量）
    stair = {
        "known0_base_d1": base_d1,
        "known1_oracle1_d2": day_tbl["oracle_1"][1],
        "known3_oracle3_d4": day_tbl["oracle_3"][3],
        "known7_oracle7_d8": day_tbl["oracle_7"][7],
    }
    print(f"  1-day-ahead 阶梯（同难度、不同已知轨迹长度）：", flush=True)
    for k, v in stair.items():
        rel = 100 * (base_d1 - v) / base_d1 if base_d1 else 0
        print(f"    {k:<24} CRPS={v:.4f}  vs base_d1 提升 {rel:+.1f}%", flush=True)
    # 固定目标第 8 天：已知轨迹 0 vs 7 天
    gap_d8 = base_d8 - day_tbl["oracle_7"][7]
    print(f"  固定目标第8天：base_d8={base_d8:.4f} vs oracle7_d8={day_tbl['oracle_7'][7]:.4f} "
          f"→ 差 {gap_d8:.4f}（= 未来7天未知的固有代价，不可桥接）", flush=True)
    # 特征贡献：base vs ar_only
    ar_d1 = day_tbl["ar_only"][0]
    feat_gain = 100 * (ar_d1 - base_d1) / ar_d1 if ar_d1 else 0
    print(f"  特征贡献：ar_only_d1={ar_d1:.4f} vs base_d1={base_d1:.4f} "
          f"→ temp/气象给 {feat_gain:+.1f}%（<0 说明 conc 历史自回归主导）", flush=True)
    # base vs ar_only 全量
    base_all = summary["base"]["crps_all"]
    ar_all = summary["ar_only"]["crps_all"]
    print(f"  特征贡献（全8天）：ar_only CRPS_all={ar_all:.4f} vs base {base_all:.4f} "
          f"→ {100*(ar_all-base_all)/base_all if base_all else 0:+.1f}%", flush=True)

    # World C：逐窗口 base 第1天 vs oracle7 第8天（同难度 1-day-ahead，不同已知量）gap
    print("\n===== World C 检验：逐窗口 gap（base_d1 − oracle7_d8）vs 浓度波动 =====", flush=True)
    print(f"  {'win':<6}{'conc_p50':<10}{'conc_range':<11}{'base_d1':<10}{'oracle7_d8':<12}"
          f"{'gap':<8}", flush=True)
    gaps = []; ranges = []
    for w in win_stats:
        wi = w["window"] - 1
        b1 = float(np.mean([agg["base"][s]["day_crps"][wi][0] for s in SEEDS_]))
        o8 = float(np.mean([agg["oracle_7"][s]["day_crps"][wi][7] for s in SEEDS_]))
        gap = b1 - o8
        gaps.append(gap); ranges.append(w["conc_range"])
        print(f"  {w['window']:<6}{w['conc_p50']:<10.2f}{w['conc_range']:<11.2f}"
              f"{b1:<10.4f}{o8:<12.4f}{gap:<8.4f}", flush=True)
    if len(gaps) > 2:
        corr_gap_range = float(np.corrcoef(gaps, ranges)[0, 1])
        corr_gap_p50 = float(np.corrcoef(gaps, [w["conc_p50"] for w in win_stats])[0, 1])
        corr_gap_std = float(np.corrcoef(gaps, [w["conc_std"] for w in win_stats])[0, 1])
        print(f"  corr(gap, 浓度范围) = {corr_gap_range:+.3f}   "
              f"corr(gap, p50) = {corr_gap_p50:+.3f}   "
              f"corr(gap, 浓度std) = {corr_gap_std:+.3f}", flush=True)
    else:
        corr_gap_range = corr_gap_p50 = corr_gap_std = float("nan")

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H8": H8, "H64": H64, "epochs": args.epochs, "n_windows": Nw,
                     "seeds": SEEDS_, "w": {"m1": W_M1, "m2": W_M2, "m4": W_M4},
                     "arch": "base/ar_only: RamsNet; oracle arms: OracleRamsNet(通道+头注入)",
                     "oracle_note": "oracle 通道=未来 N 天真实 conc（诊断探针，非部署，明确标注未来真实值）"},
        "arms": summary,
        "day_crps": {a: [round(v, 4) for v in day_tbl[a]] for a in ARMS},
        "worldview": {
            "staircase_1day": {k: round(v, 4) for k, v in stair.items()},
            "base_day1": round(base_d1, 4),
            "base_day8": round(base_d8, 4),
            "oracle7_day8": round(day_tbl["oracle_7"][7], 4),
            "gap_d8_unknown_cost": round(gap_d8, 4),
            "ar_only_day1": round(ar_d1, 4),
            "feat_gain_pct": round(feat_gain, 2),
            "corr_gap_range": round(corr_gap_range, 3),
            "corr_gap_p50": round(corr_gap_p50, 3),
            "corr_gap_std": round(corr_gap_std, 3),
        },
        "windows": win_stats,
    }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
