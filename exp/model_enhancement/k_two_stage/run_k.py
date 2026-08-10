# -*- coding: utf-8 -*-
"""K 方向探索：两阶段训练（解"精度 vs 校准"矛盾）

背景（A 方向已证实）：
  - 单任务增量（GRU+分位数头）：点精度好（CRPS 0.857 / RMSE 1.665）但区间严重欠校准（覆盖 0.672）
  - 多任务增量（RamsNet w=1/3/2）：区间校准达标（覆盖 0.806）但稀释点精度（CRPS 0.891 / RMSE 1.774）
  - 矛盾：精度好→校准差；校准好→精度差。未解。

假设：两阶段训练可能同时拿到两者——
  Stage1：单任务训 M1 增量（保精度）N epoch
  Stage2：冻结 backbone 只训多头（保校准）M epoch；或解冻全部但 M2/M4 loss 主导
  关键问题：两阶段能否拿到单任务的精度 + 多任务的校准？

实验设计（同一 17 窗口滚动协议，与 A 完全同口径，可横向对照）：
  arm0 single     = 增量 + 单任务 GRU 分位数头（A 基准，30 epoch，复现 A 0.857/1.665/0.672）
  arm1 multi      = 增量 + 多任务 RamsNet w=1/3/2（A 基准，30 epoch，复现 A 0.891/1.774/0.806）
  arm2 ts_freeze  = 两阶段：Stage1 单任务 M1 20ep → Stage2 冻结 backbone 只训 M1/M2/M4 头 10ep
  arm3 ts_full    = 两阶段：Stage1 单任务 M1 20ep → Stage2 解冻全部、多任务 loss 主导 10ep

评估（全部还原 conc 单位，同一测试段）：每视界 CRPS / RMSE / 覆盖率 + 辅助任务 acc。
3 seed（0/1/2）。保密：只输出聚合统计量。
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
from rams.models.rams_net import QUANTILES, RamsNet, SharedGRU  # noqa: E402
from rams.training.trainer import Trainer, MultiTaskLoss  # noqa: E402

T, H = 24, 8                 # 回看 3 天 / 预测 24h（8×3h 步）
EPOCHS = 30                  # single / multi 全程 epoch
EP1, EP2 = 20, 10            # 两阶段：Stage1 / Stage2 epoch
SEEDS = [0, 1, 2]

# ---- 滚动窗口参数（天，3h 网格：1 天 = 8 个时刻）----
TRAIN_DAYS = 730             # 每个窗口用 2 年训练
TEST_DAYS = 90               # 每窗口测试后 3 个月
STRIDE_DAYS = 45             # 每 45 天推进一个窗口
GRID_PER_DAY = 8

# 多任务权重（与冻结 Trainer/rams 口径一致：w=1/3/2）
W_M1, W_M2, W_M4 = 1.0, 3.0, 2.0

ARMS = ["single", "multi", "ts_freeze", "ts_full"]


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 T4/B2/B7/A 一致实现）。"""
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
    """窗口 [i0,i1)：归一化特征（训练段统计）、原始浓度、增量目标、M2/M4 标签。

    Returns:
      Xw        (n_w, T, F)  标准化特征窗口（训练段统计，防泄漏）
      y_abs     (n_w, H)     conc_{t+h} 原始尺度
      cur_raw   (n_w,)       conc_t 原始尺度
      delta_raw (n_w, H)     conc_{t+h} - conc_t 原始尺度
      strat_w, warn_w        M2/M4 多任务标签
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
    delta_raw = y_abs - cur_raw[:, None]

    # M2 分层标签（复用 B2/B7/A：delta_T 训练段中位数阈值）
    delta = df["delta_T"].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（复用 B2/B7/A：未来 24h 浓度峰值 + 训练段分位数阈值）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, y_abs, cur_raw, delta_raw, strat_w, warn_w


class IncrementQuantileNet(nn.Module):
    """单任务增量分位数网（与 A/B1 同架构）：GRU 骨干 + 单头分位数回归 p10/p50/p90。"""

    def __init__(self, feat_dim: int, horizon: int, hidden: int = 64,
                 n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden, n_layers, dropout)
        self.horizon = horizon
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, horizon * 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))  # (B, 3H): [p10, p50, p90]


def quantile_loss(pred, target, qs=QUANTILES):
    """分位数损失（合并全部视界）。pred: (B,3H), target: (B,H)。"""
    H = target.shape[1]
    e = target.unsqueeze(1) - pred.reshape(-1, 3, H)
    losses = [torch.mean(torch.maximum(q * e[:, i], (q - 1) * e[:, i]))
              for i, q in enumerate(qs)]
    return torch.stack(losses).mean()


def train_single(Xw, y_norm, n_tr, epochs, device, seed):
    """单任务（A 基准）：增量分位数 GRU。返回模型。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = IncrementQuantileNet(feat_dim=Xw.shape[2], horizon=H)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(Xw[:n_tr], device=device)
    yt = torch.tensor(y_norm[:n_tr], dtype=torch.float32, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 128):
            idx = perm[s:s + 128]
            opt.zero_grad()
            loss = quantile_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


def train_multi(Xw, y_norm, strat_w, warn_w, n_tr, epochs, device, seed):
    """多任务（A 基准）：RamsNet w=1/3/2（= B7 abs_delta / A multi 完全同口径）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RamsNet(feat_dim=Xw.shape[2], horizon=H, use_m4=True)
    trainer = Trainer(model, device=device, w_m1=W_M1, w_m2=W_M2, w_m4=W_M4)
    trainer.fit(Xw[:n_tr], y_norm[:n_tr], strat_w[:n_tr],
                Xw[n_tr:], y_norm[n_tr:], strat_w[n_tr:],
                warn_tr=warn_w[:n_tr], warn_va=warn_w[n_tr:],
                epochs=epochs, batch_size=128)
    return model


def train_stage1_ramsonly(Xw, y_norm, n_tr, epochs, device, seed):
    """两阶段 Stage1：RamsNet 只用 M1 分位数 loss 训骨干+头（单任务口径，保精度）。

    与 train_single 同 loss/优化器/批大小/打乱；仅多 m2/m4 头（零梯度，不影响 M1）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = RamsNet(feat_dim=Xw.shape[2], horizon=H, use_m4=True)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(Xw[:n_tr], device=device)
    yt = torch.tensor(y_norm[:n_tr], dtype=torch.float32, device=device)
    n = len(Xt)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 128):
            idx = perm[s:s + 128]
            opt.zero_grad()
            m1, _, _ = model(Xt[idx])
            loss = quantile_loss(m1, yt[idx])
            loss.backward()
            opt.step()
    return model


def _m4_class_weights(warn_tr, n_levels, device):
    """M4 类别权重（逆频率，与冻结 Trainer 自动逻辑一致）。"""
    counts = np.bincount(warn_tr, minlength=n_levels)
    inv = 1.0 / (counts.astype(np.float64) + 1.0)
    w = torch.tensor(inv / inv.sum() * len(counts), dtype=torch.float32)
    return w.to(device)


def train_stage2(model, Xw, y_norm, strat_w, warn_w, n_tr, epochs, device,
                 freeze_backbone: bool):
    """两阶段 Stage2：多任务 loss（w=1/3/2）微调。

    freeze_backbone=True  → 冻结 backbone，只训 M1/M2/M4 头
    freeze_backbone=False → 解冻全部（M2/M4 loss 主导：w2+w4=5 > w1=1）
    M4 类别权重按训练段自动逆频率（与 Trainer 一致）。
    返回 model（已原地训练）。
    """
    model.train()
    n_levels = model.m4.mlp[-1].out_features
    crit = MultiTaskLoss(model.horizon, W_M1, W_M2, W_M4, True,
                         _m4_class_weights(warn_w[:n_tr], n_levels, device))
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-3)
    Xt = torch.tensor(Xw[:n_tr], device=device)
    yt = torch.tensor(y_norm[:n_tr], dtype=torch.float32, device=device)
    st = torch.tensor(strat_w[:n_tr], dtype=torch.long, device=device)
    wt = torch.tensor(warn_w[:n_tr], dtype=torch.long, device=device)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 128):
            idx = perm[s:s + 128]
            opt.zero_grad()
            m1, m2, m4 = model(Xt[idx])
            loss, _, _, _ = crit(m1, m2, yt[idx], st[idx], m4, wt[idx])
            loss.backward()
            opt.step()
    if freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = True
    return model


def predict_single(model, X_te, device):
    model.eval()
    with torch.no_grad():
        m1 = model(torch.tensor(X_te).to(device))
        q = m1.reshape(-1, 3, H)
    return q.cpu().numpy().astype(np.float64)


def predict_multi(model, X_te, device):
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1)
    return q.cpu().numpy().astype(np.float64)


def back_to_conc(cur_te, q_norm, scale):
    """增量分位数还原到 conc 单位：conc = cur + pred*scale。q_norm: (N,3,H)。"""
    return cur_te[:, None, None] + q_norm * scale


def main():
    ap = argparse.ArgumentParser(description="K 方向：两阶段训练（精度 vs 校准）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="single/multi 全程 epoch")
    ap.add_argument("--ep1", type=int, default=EP1, help="两阶段 Stage1 epoch")
    ap.add_argument("--ep2", type=int, default=EP2, help="两阶段 Stage2 epoch")
    ap.add_argument("--arms", nargs="+", default=ARMS, help="arm 子集")
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 1 seed × 2 epoch")
    ap.add_argument("--out-json", default="exp/model_enhancement/k_two_stage/results.json")
    args = ap.parse_args()

    arms = [a for a in args.arms if a in ARMS]
    t0 = time.time()
    print(f"== K 方向：两阶段训练（单任务精度 + 多任务校准，3-seed，17 窗口）==", flush=True)
    print(f"   arms={arms} seeds={SEEDS} single/multi epochs={args.epochs} "
          f"two-stage ep1+ep2={args.ep1}+{args.ep2}", flush=True)
    print(f"   ts_freeze = Stage1 单任务 M1 → Stage2 冻结 backbone 只训头", flush=True)
    print(f"   ts_full   = Stage1 单任务 M1 → Stage2 解冻全部、多任务 loss 主导", flush=True)

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
    Nw = len(windows)
    print(f"[2] 滚动窗口 {Nw} 个（训练 {TRAIN_DAYS}d + 测试 {TEST_DAYS}d）", flush=True)

    # 每 arm × seed：逐视界 CRPS/RMSE（平均窗口）、每窗口平均、覆盖率、相对技能
    agg = {a: {s: {
        "crps_h": np.zeros((Nw, H)), "rmse_h": np.zeros((Nw, H)),
        "crps": np.zeros(Nw), "rmse": np.zeros(Nw),
        "cover": np.zeros(Nw), "crps_p": np.zeros(Nw),
        "m2_acc": np.zeros(Nw), "m4_acc": np.zeros(Nw),
    } for s in SEEDS} for a in arms}
    persist_crps_h = np.zeros((Nw, H))   # 持久化逐视界 CRPS（每窗口）
    persist_rmse_h = np.zeros((Nw, H))   # 持久化逐视界 RMSE（每窗口）

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        Xw, y_abs, cur_raw, delta_raw, strat_w, warn_w = build_window(
            wide, i0, i1, feat_cols)
        n_win = len(Xw)
        n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
        te_sl = slice(n_tr, n_win)

        Xte = Xw[te_sl]
        cur_te = cur_raw[te_sl]
        obs = y_abs[te_sl]                    # (N,H) 原始 conc 观测
        Nte = len(Xte)

        # 增量目标：归一化用窗口训练段拟合 scale（防泄漏）
        sd_inc = float(np.std(delta_raw[:n_tr])) + 1e-8
        y_inc = (delta_raw / sd_inc).astype(np.float32)

        # 持久化基线（Δ≡0 → conc_{t+h}=conc_t，逐视界）
        q_p = np.repeat(cur_te[:, None, None], H, axis=2).astype(np.float64)  # (N,1,H)
        crps_p_h = np.array([np.mean(crps_quantiles(q_p[:, 0, h], q_p[:, 0, h],
                                                    q_p[:, 0, h], obs[:, h]))
                             for h in range(H)])
        persist_crps_h[wi] = crps_p_h
        persist_rmse_h[wi] = np.array([np.sqrt(np.mean((q_p[:, 0, h] - obs[:, h]) ** 2))
                                       for h in range(H)])
        crps_p = float(crps_p_h.mean())
        rmse_p = float(np.sqrt(np.mean((q_p[:, 0, :] - obs) ** 2)))

        for a in arms:
            for s in SEEDS:
                if a == "single":
                    model = train_single(Xw, y_inc, n_tr, args.epochs, args.device, s)
                    q_norm = predict_single(model, Xte, args.device)
                elif a == "multi":
                    model = train_multi(Xw, y_inc, strat_w, warn_w, n_tr,
                                        args.epochs, args.device, s)
                    q_norm = predict_multi(model, Xte, args.device)
                else:  # 两阶段
                    model = train_stage1_ramsonly(Xw, y_inc, n_tr, args.ep1,
                                                  args.device, s)
                    model = train_stage2(model, Xw, y_inc, strat_w, warn_w, n_tr,
                                         args.ep2, args.device,
                                         freeze_backbone=(a == "ts_freeze"))
                    q_norm = predict_multi(model, Xte, args.device)
                q_conc = back_to_conc(cur_te, q_norm, sd_inc)   # (N,3,H) conc 单位

                crps_h = np.array([np.mean(crps_quantiles(q_conc[:, 0, h], q_conc[:, 1, h],
                                                          q_conc[:, 2, h], obs[:, h]))
                                   for h in range(H)])
                rmse_h = np.array([np.sqrt(np.mean((q_conc[:, 1, h] - obs[:, h]) ** 2))
                                   for h in range(H)])
                cover = float(np.mean((obs >= q_conc[:, 0]) & (obs <= q_conc[:, 2])))

                g = agg[a][s]
                g["crps_h"][wi] = crps_h
                g["rmse_h"][wi] = rmse_h
                g["crps"][wi] = float(crps_h.mean())
                g["rmse"][wi] = float(np.sqrt(np.mean(rmse_h ** 2)))
                g["cover"][wi] = cover
                g["crps_p"][wi] = crps_p
                if a in ("multi", "ts_freeze", "ts_full"):
                    model.eval()
                    with torch.no_grad():
                        _, m2v, m4v = model(torch.tensor(Xte).to(args.device))
                    g["m2_acc"][wi] = float(
                        (m2v.argmax(1).cpu().numpy() == strat_w[te_sl]).mean())
                    g["m4_acc"][wi] = float(
                        (m4v.argmax(1).cpu().numpy() == warn_w[te_sl]).mean())

        # 窗口摘要（seed 均值）
        for a in arms:
            c = np.mean([agg[a][s]["crps"][wi] for s in SEEDS])
            r = np.mean([agg[a][s]["rmse"][wi] for s in SEEDS])
            cv = np.mean([agg[a][s]["cover"][wi] for s in SEEDS])
            print(f"    {a:<9} CRPS={c:.4f} (持久化 {crps_p:.4f})  RMSE={r:.3f} "
                  f"(持久化 {rmse_p:.3f})  覆盖={cv:.3f}  n_test={Nte}", flush=True)

    # ---- 聚合（3-seed 均值±std，窗口平均）----
    print("\n===== 逐视界 CRPS 对照（conc 单位，3-seed 均值，窗口平均）=====", flush=True)
    hdr = f"  {'h':<4}" + "".join(f"{a:<14}" for a in arms) + f"{'持久化':<10}"
    print(hdr, flush=True)
    for h in range(H):
        line = f"  {h:<4}"
        for a in arms:
            mu0, _ = _arm_stats(agg, a, "crps_h", h, SEEDS, Nw)
            line += f"{mu0:<14.4f}"
        line += f"{float(np.mean(persist_crps_h[:, h])):<10.4f}"
        print(line, flush=True)

    print("\n===== 逐视界 RMSE 对照（conc 单位，3-seed 均值）=====", flush=True)
    print(hdr, flush=True)
    for h in range(H):
        line = f"  {h:<4}"
        for a in arms:
            mu0, _ = _arm_stats(agg, a, "rmse_h", h, SEEDS, Nw)
            line += f"{mu0:<14.4f}"
        line += f"{float(np.mean(persist_rmse_h[:, h])):<10.4f}"
        print(line, flush=True)

    print("\n===== 平均对照（3-seed 均值±std）=====", flush=True)
    summary = {}
    for a in arms:
        crps_mean = _arm_scalar(agg, a, "crps", SEEDS, Nw)
        crps_std = _arm_scalar_std(agg, a, "crps", SEEDS, Nw)
        rmse_mean = _arm_scalar(agg, a, "rmse", SEEDS, Nw)
        rmse_std = _arm_scalar_std(agg, a, "rmse", SEEDS, Nw)
        cover_mean = _arm_scalar(agg, a, "cover", SEEDS, Nw)
        cover_std = _arm_scalar_std(agg, a, "cover", SEEDS, Nw)
        crps_p_mean = float(np.mean([agg[a][s]["crps_p"][wi]
                                     for s in SEEDS for wi in range(Nw)]))
        skill = (crps_p_mean - crps_mean) / crps_p_mean * 100
        print(f"  {a:<9} CRPS={crps_mean:.4f}±{crps_std:.4f}  持久化={crps_p_mean:.4f}  "
              f"技能={skill:+.1f}%   RMSE={rmse_mean:.3f}±{rmse_std:.3f}  "
              f"覆盖={cover_mean:.3f}±{cover_std:.3f}", flush=True)
        summary[a] = {
            "crps": round(float(crps_mean), 4), "crps_std": round(float(crps_std), 4),
            "rmse": round(float(rmse_mean), 4), "rmse_std": round(float(rmse_std), 4),
            "coverage": round(float(cover_mean), 4), "coverage_std": round(float(cover_std), 4),
            "crps_persist": round(float(crps_p_mean), 4),
            "skill_vs_persist_pct": round(float(skill), 2),
        }
        if a in ("multi", "ts_freeze", "ts_full"):
            m2m = _arm_scalar(agg, a, "m2_acc", SEEDS, Nw)
            m4m = _arm_scalar(agg, a, "m4_acc", SEEDS, Nw)
            summary[a]["m2_acc"] = round(float(m2m), 4)
            summary[a]["m4_acc"] = round(float(m4m), 4)

    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "ep1": args.ep1, "ep2": args.ep2,
                     "n_windows": Nw, "seeds": SEEDS, "w": {"m1": W_M1, "m2": W_M2, "m4": W_M4},
                     "single_arch": "IncrementQuantileNet (GRU+quantile head)",
                     "multi_arch": "RamsNet use_m4=True (M1 inc + M2 + M4)",
                     "ts": "Stage1 M1-only single-task → Stage2 multi-task (freeze/full)"},
        "arms": summary,
        "crps_h": {a: {str(h): {"mean": round(float(_arm_stats(agg, a, "crps_h", h, SEEDS, Nw)[0]), 4),
                                "std": round(float(_arm_stats(agg, a, "crps_h", h, SEEDS, Nw)[1]), 4)}
                       for h in range(H)} for a in arms},
        "rmse_h": {a: {str(h): {"mean": round(float(_arm_stats(agg, a, "rmse_h", h, SEEDS, Nw)[0]), 4),
                                "std": round(float(_arm_stats(agg, a, "rmse_h", h, SEEDS, Nw)[1]), 4)}
                       for h in range(H)} for a in arms},
        "persist_crps_h": [round(float(x), 4) for x in persist_crps_h.mean(axis=0)],
        "persist_rmse_h": [round(float(x), 4) for x in persist_rmse_h.mean(axis=0)],
        "windows": [],
    }
    # 逐窗口 seed 均值（审计用，不越权打印数据）
    for wi in range(Nw):
        row = {"window": wi + 1}
        for a in arms:
            row[f"{a}_crps"] = round(float(np.mean([agg[a][s]["crps"][wi] for s in SEEDS])), 4)
            row[f"{a}_rmse"] = round(float(np.mean([agg[a][s]["rmse"][wi] for s in SEEDS])), 4)
            row[f"{a}_cover"] = round(float(np.mean([agg[a][s]["cover"][wi] for s in SEEDS])), 4)
        res["windows"].append(row)
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


def _arm_stats(agg, arm, key, h, seeds, nw):
    """arm 逐视界指标在 (seed, window) 上的均值±std。key ∈ {crps_h, rmse_h}。"""
    vals = [agg[arm][s][key][wi, h] for s in seeds for wi in range(nw)]
    return float(np.mean(vals)), float(np.std(vals))


def _arm_scalar(agg, arm, key, seeds, nw):
    vals = [agg[arm][s][key][wi] for s in seeds for wi in range(nw)]
    return float(np.mean(vals))


def _arm_scalar_std(agg, arm, key, seeds, nw):
    vals = [agg[arm][s][key][wi] for s in seeds for wi in range(nw)]
    return float(np.std(vals))


if __name__ == "__main__":
    main()
