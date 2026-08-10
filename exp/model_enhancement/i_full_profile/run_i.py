# -*- coding: utf-8 -*-
"""I 探索：预测全剖面（表层单层 vs 全剖面 20 层 vs 关键层）

方向 I 核心问题：目前所有模型都预测 conc_0.5（表层单层），但数据有 20 层完整剖面。
M3 已验证垂向信息价值（20 层输入降误差 32%）。本实验验证**输出侧**是否该用全剖面——
预测全剖面是**帮助表层**（多任务共享：分层结构让 backbone 学更泛化表征）还是**拖累表层**
（输出头负担变重、层间噪声干扰）？

同一增量 abs_delta + 滚动窗口协议（复用 B1/B2/B7/C：训练 730d / 测试 90d / 步长 45d，
17 窗口）下对比 3 变体（同一 GRU 骨干 + 分位数头 + M2/M4 多任务，仅 M1 输出头不同）：

  1. surface  : 基线，M1 预测表层 conc_0.5（当前所有模型的做法）
  2. full     : M1 预测全剖面 20 层（Δ 按层定义：Δ_h(d) = conc_{t+h}(d) - conc_t(d)）
  3. key      : M1 预测关键层 {conc_0.5, conc_3.5, conc_7.0}（表层+中上层+中下层，3 层）

M1 输出头设计（RamsNet 冻结架构只支持单层 M1 头，故在 exp/ 下扩展，不碰 rams/ 冻结代码）：
  surface : M1Head(hidden, H)          → (B, 3H)      [复用冻结组件]
  full    : M1Head(hidden, H×20)       → (B, 3×20H)   [共享 backbone，同一 MLP 头出 20 层]
  key     : M1Head(hidden, H×3)        → (B, 3×3H)    [共享 backbone，同一 MLP 头出 3 层]

训练：冻结 Trainer/MultiTaskLoss 的 M1 分位数损失只支持单层目标（reshape (-1,3,H)），
多层目标下无法广播 → 本文件内复制冻结训练协议（分位数损失 + M2/M4 交叉熵 + 同权重
w_m1/w_m2/w_m4 + Adam lr=1e-3 + batch 128 + 30 epoch），仅把 M1 分位数损失泛化到多层
（reshape (-1, NL, 3, H)）。M4 类别权重沿用冻结 Trainer 的逆频率自动计算。

评估（全部还原 conc 单位；CRPS 分位数闭合形式与 T4/B2/B7/C 一致）：
  a. 表层（conc_0.5）逐视界 CRPS + p50 RMSE + 区间覆盖率 [p10,p90]（3 变体可比）
     → 关键问题：预测全剖面帮助还是拖累表层？
  b. 全剖面重建 RMSE：full 变体各层 p50 还原后 vs 观测（表层/中层/底层分别汇总 +
     全部 20 层加权平均）→ 全剖面重建质量
  c. key 变体在所选 3 层各自的重建 RMSE
  d. 对照：每变体各自"表层持久化"（零变化 = conc_{t+h}=conc_t）CRPS

保密：只输出聚合统计量 / CRPS / RMSE / 覆盖率，不打印原始数据行（继承 B7/C 约束）。
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
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402
from rams.models.rams_net import SharedGRU, M1Head, M2Head, M4Head, QUANTILES  # noqa: E402
# 注意：不 import 冻结 Trainer/MultiTaskLoss —— 其 M1 分位数损失只支持单层目标
# （reshape (-1,3,H) 在多层 y 下无法广播）。多层训练循环在本文件内复制冻结协议。

T, H = 24, 8
EPOCHS = 30
SEEDS = [0, 1, 2]          # 3 seed（任务要求）
N_SEED = len(SEEDS)

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）
TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
GRID_PER_DAY = 8

# 关键层：表层 + 中上层（3.5m，浓度均值峰值区）+ 中下层（7.0m）
KEY_DEPTHS = [0.5, 3.5, 7.0]
KEY_COLS = [f"conc_{d}" for d in KEY_DEPTHS]

VARIANTS = ["surface", "full", "key"]


class MLayerNet(nn.Module):
    """多任务网络（M1 多层输出 + M2 + M4）：在 rams/ 冻结架构外扩展 M1 头输出多层。

    与冻结 RamsNet 共享同一 backbone（SharedGRU）+ M1Head/M2Head/M4Head 组件（import 复用），
    仅把 M1 输出维度从 H 扩到 H×n_target_layer，即同一 MLP 头输出各层的分位数预测。
    forward 返回 (m1, m2, m4)，m1: (B, 3×n_layer×H) = [p10; p50; p90] 按层堆叠。
    """

    def __init__(self, feat_dim: int, horizon: int, n_layer: int, hidden: int = 64,
                 n_layers: int = 1, n_classes: int = 2, n_levels: int = 4,
                 use_m4: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden, n_layers, dropout)
        self.m1 = M1Head(hidden, horizon * n_layer, quantile=True)
        self.m2 = M2Head(hidden, n_classes)
        self.m4 = M4Head(hidden, n_levels) if use_m4 else None
        self.horizon = horizon
        self.n_layer = n_layer
        self.use_m4 = use_m4

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        m4 = self.m4(h) if self.m4 is not None else None
        return self.m1(h), self.m2(h), m4


def crps_quantiles(q10, q50, q90, y):
    """分位数预测（p10/p50/p90）的 CRPS 闭合形式（与 T4/B2/B7/C 一致实现）。"""
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
    """返回窗口 [i0,i1) 的标准化特征 + 全剖面原始浓度。

    Returns:
      Xw      (n_w, T, F) 标准化特征窗口（训练段统计）
      cur_raw (n_w, n_layer) 各层 conc_t 原始尺度（0.5m 在最前）
      y_abs   (n_w, n_layer, H) 各层 conc_{t+h} 原始尺度
      strat_w, warn_w  M2/M4 标签（复用 B2/B7/C）
    """
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr = int(n * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))

    # 特征标准化（只用训练段）
    Xtr = df[feat_cols].values[:n_tr].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((df[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    # 全剖面 20 层浓度（列序 = 深度升序，索引 0 = 表层）
    conc_cols = [c for c in df.columns if c.startswith("conc_")]
    y_mat = df[conc_cols].values.astype(np.float64)     # (n, 20)

    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_mat[i + T:i + T + H] for i in range(n_w)]).transpose(0, 2, 1)  # (n_w, 20, H)
    cur_raw = np.stack([y_mat[i + T - 1] for i in range(n_w)])           # (n_w, 20)

    # M2 分层标签（B2/B7/C 复用）
    delta = df["delta_T"].values
    thr = float(np.median(delta[:n_tr]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（B2/B7/C 复用；用表层浓度）
    warn_val = y_abs[:, 0, :].max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    return Xw, cur_raw, y_abs, strat_w, warn_w, conc_cols


def select_layers(cur_raw, y_abs, conc_cols, sel):
    """按层名列表抽取 (cur_raw, y_abs) 的子集。"""
    idx = [conc_cols.index(c) for c in sel]
    return cur_raw[:, idx], y_abs[:, idx, :]


def make_targets_abs_delta(cur_raw, y_abs):
    """abs_delta 目标：Δ_h(d) = conc_{t+h}(d) - conc_t(d)。cur/y 同 shape。"""
    return y_abs - cur_raw[:, :, None]


def _multi_task_loss(m1_out, m2_out, m4_out, y, strat_label, warn_label,
                     horizon, n_layer, use_m4, w_m1, w_m2, w_m4, warn_ce=None):
    """多层多任务损失（复制冻结 MultiTaskLoss 协议，M1 分位数损失泛化到 NL 层）。

    m1_out: (B, 3*NL*H)；y: (B, NL, H)。分位数损失在 NL 层与 H 视界上平均。
    """
    yb = y.to(m1_out.device)
    sb = strat_label.to(m1_out.device)
    qs = torch.tensor(QUANTILES, device=m1_out.device)
    m1_q = m1_out.reshape(-1, n_layer, 3, horizon)   # (B, NL, 3, H)
    # (B, NL, 1, H) vs (B, NL, 3, H) → (B, NL, 3, H)
    e = yb.unsqueeze(2) - m1_q
    losses = [torch.mean(torch.maximum(q * e[:, :, i, :], (q - 1) * e[:, :, i, :]))
              for i, q in enumerate(qs)]
    l1 = torch.stack(losses).mean()

    ce = nn.CrossEntropyLoss()
    l2 = ce(m2_out, sb)
    l4 = None
    if use_m4 and m4_out is not None and warn_label is not None:
        wb = warn_label.to(m1_out.device)
        l4 = (warn_ce if warn_ce is not None else ce)(m4_out, wb)
    total = w_m1 * l1 + w_m2 * l2
    if l4 is not None:
        total = total + w_m4 * l4
    return total, l1, l2, l4


def _fit_multilayer(X_tr, y_tr, strat_tr, warn_tr, X_va, y_va, strat_va, warn_va,
                    model, epochs, batch_size, seed):
    """多层训练（复制冻结 Trainer 协议：Adam lr=1e-3、shuffle=False、M4 逆频率权重）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = next(model.parameters()).device
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    use_m4 = model.use_m4
    warn_weights = None
    if use_m4 and warn_tr is not None:
        n_levels = model.m4.mlp[-1].out_features
        counts = np.bincount(warn_tr, minlength=n_levels)
        inv = 1.0 / (counts.astype(np.float64) + 1.0)
        warn_weights = torch.tensor(inv / inv.sum() * len(counts), dtype=torch.float32).to(dev)
    warn_ce = nn.CrossEntropyLoss(weight=warn_weights) if warn_weights is not None else None

    tensors = [torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(strat_tr)]
    if use_m4 and warn_tr is not None:
        tensors.append(torch.tensor(warn_tr))
    dl = DataLoader(TensorDataset(*tensors), batch_size=batch_size, shuffle=False)
    Xv = torch.tensor(X_va).to(dev)
    yv = torch.tensor(y_va).to(dev)
    sv = torch.tensor(strat_va).to(dev)
    wv = torch.tensor(warn_va).to(dev) if (use_m4 and warn_va is not None) else None

    NL, Hh = model.n_layer, model.horizon
    for ep in range(epochs):
        model.train()
        for batch in dl:
            xb = batch[0].to(dev)
            yb = batch[1].to(dev)
            sb = batch[2].to(dev)
            wb = batch[3].to(dev) if len(batch) > 3 else None
            opt.zero_grad()
            m1, m2, m4 = model(xb)
            loss, l1, l2, l4 = _multi_task_loss(m1, m2, m4, yb, sb, wb, Hh, NL,
                                                use_m4, 1.0, 3.0, 2.0, warn_ce)
            loss.backward()
            opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                m1v, m2v, m4v = model(Xv)
                pred = m1v.reshape(-1, NL, 3, Hh)[:, :, 1, :]      # (B, NL, H) p50
                val_rmse = float(torch.sqrt(torch.mean((pred - yv) ** 2)).item())
                val_acc = float((m2v.argmax(1) == sv).float().mean().item())
                extra = ""
                if m4v is not None and wv is not None:
                    val_wacc = float((m4v.argmax(1) == wv).float().mean().item())
                    extra = f" val_wacc={val_wacc:.4f}"
            print(f"  ep{ep} loss={loss.item():.4f} val_rmse={val_rmse:.4f} "
                  f"val_acc={val_acc:.4f}{extra}", flush=True)
    return model


def train_model(Xw, yw, strat_w, warn_w, n_tr, n_layer, epochs, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLayerNet(feat_dim=Xw.shape[2], horizon=H, n_layer=n_layer, use_m4=True).to(device)
    _fit_multilayer(Xw[:n_tr], yw[:n_tr], strat_w[:n_tr], warn_w[:n_tr],
                    Xw[n_tr:], yw[n_tr:], strat_w[n_tr:], warn_w[n_tr:],
                    model, epochs, batch_size=128, seed=seed)
    return model


def predict_quantiles(model, X_te, device):
    """(N, n_layer, 3, H) 归一化分位数预测，n_layer 按深度升序。

    M1 输出布局与 _multi_task_loss 一致：[n][layer][quantile][horizon]。
    """
    model.eval()
    with torch.no_grad():
        m1, _, _ = model(torch.tensor(X_te).to(device))
    m1 = m1.cpu().numpy().astype(np.float64)           # (N, 3*n_layer*H)
    NL, Hh = model.n_layer, model.horizon
    q = m1.reshape(-1, NL, 3, Hh)                        # (N, NL, 3, H)
    return q


def per_variant_metrics(variant, q_all, cur_te, y_te):
    """计算某变体的指标（全部 conc 单位）。

    q_all : (N_seed, N, n_layer, 3, H) 该变体各 seed 的分位数预测（conc 单位）
    cur_te: (N, n_layer)  各层 conc_t
    y_te  : (N, n_layer, H)  各层 conc_{t+h} 观测

    Returns: dict 含表层指标 + 各层重建 RMSE + 表层覆盖率/持久化 CRPS。
    """
    obs_surface = y_te[:, 0, :]            # (N, H)
    cur_surface = cur_te[:, 0]             # (N,)
    # 各 seed 指标
    seed_crps = []
    seed_crps_h = []
    seed_rmse = []
    seed_cov = []
    q_ens = q_all.mean(axis=0)             # (N, n_layer, 3, H) 分位数跨 seed 平均
    for si in range(q_all.shape[0]):
        q = q_all[si]
        qs0 = q[:, 0]                      # (N, 3, H) 表层
        crps_h = [float(np.mean(crps_quantiles(qs0[:, 0, h], qs0[:, 1, h], qs0[:, 2, h],
                                               obs_surface[:, h]))) for h in range(H)]
        crps = float(np.mean(crps_h))
        rmse = float(np.sqrt(np.mean((qs0[:, 1] - obs_surface) ** 2)))
        cov = float(np.mean((obs_surface >= qs0[:, 0]) & (obs_surface <= qs0[:, 2])))
        seed_crps.append(crps); seed_crps_h.append(crps_h)
        seed_rmse.append(rmse); seed_cov.append(cov)

    # 集成（主答案）：表层
    qs_ens = q_ens[:, 0]                   # (N, 3, H)
    ens_crps_h = [float(np.mean(crps_quantiles(qs_ens[:, 0, h], qs_ens[:, 1, h],
                                               qs_ens[:, 2, h], obs_surface[:, h])))
                  for h in range(H)]
    ens_crps = float(np.mean(ens_crps_h))
    ens_rmse = float(np.sqrt(np.mean((qs_ens[:, 1] - obs_surface) ** 2)))
    ens_cov = float(np.mean((obs_surface >= qs_ens[:, 0]) & (obs_surface <= qs_ens[:, 2])))

    # 表层持久化（零变化 = conc_{t+h}=conc_t，逐视界）
    q_pers = np.zeros((len(obs_surface), 3, H))
    q_pers[:, 1, :] = cur_surface[:, None]
    q_pers[:, 0, :] = cur_surface[:, None]
    q_pers[:, 2, :] = cur_surface[:, None]
    crps_pers = float(np.mean([
        np.mean(crps_quantiles(q_pers[:, 0, h], q_pers[:, 1, h], q_pers[:, 2, h],
                               obs_surface[:, h])) for h in range(H)]))

    # 各层重建 RMSE（集成 p50）：n_layer 全列
    p50 = q_ens[:, :, 1, :]                # (N, n_layer, H)
    layer_rmse = np.sqrt(np.mean((p50 - y_te) ** 2, axis=(0, 2)))   # (n_layer,)
    overall_rmse = float(np.sqrt(np.mean((p50 - y_te) ** 2)))       # 全层全视界
    surf_rmse_recon = float(layer_rmse[0])

    # 表层逐视界重建 RMSE（重建质量随视界）
    surf_h_rmse = np.sqrt(np.mean((p50[:, 0, :] - obs_surface) ** 2, axis=0)).tolist()

    return {
        "seed_crps": seed_crps, "seed_crps_h": seed_crps_h,
        "seed_rmse": seed_rmse, "seed_cov": seed_cov,
        "ens_crps": ens_crps, "ens_crps_h": ens_crps_h,
        "ens_rmse": ens_rmse, "ens_cov": ens_cov,
        "crps_persist_surface": crps_pers,
        "layer_rmse": layer_rmse.tolist(),
        "recon_rmse_overall": overall_rmse,
        "recon_rmse_surface": surf_rmse_recon,
        "recon_rmse_surface_h": surf_h_rmse,
    }


def main():
    ap = argparse.ArgumentParser(description="I 预测全剖面（表层 vs 全剖面 vs 关键层）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0, help="最多窗口数（0=全部）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 1 epoch × 1 seed")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--out-json", default="exp/model_enhancement/i_full_profile/results.json")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    smoke = args.smoke
    epochs = args.epochs
    if smoke:
        seeds = seeds[:1]
        epochs = 1
    t0 = time.time()
    print(f"== I 预测全剖面（{len(seeds)} seed × {len(variants)} 变体）==", flush=True)
    print(f"   协议: 训练 {TRAIN_DAYS}d / 测试 {TEST_DAYS}d / 步长 {STRIDE_DAYS}d；"
          f"GRU 骨干 + p10/p50/p90 分位数头 + M2/M4 多任务；目标 {H}h", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格，2021-03 → 2025-09）", flush=True)

    conc_cols = [c for c in wide.columns if c.startswith("conc_")]
    print(f"   剖面层 {len(conc_cols)} 层: {conc_cols[0]} … {conc_cols[-1]}；"
          f"关键层: {[c for c in KEY_COLS]}", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    if "conc_0.5" not in feat_cols:
        feat_cols = feat_cols + ["conc_0.5"]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象 + conc_t 表层）", flush=True)

    days = TRAIN_DAYS + TEST_DAYS
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个 × {len(seeds)} seed × {len(variants)} 变体", flush=True)

    Nw = len(windows)
    # 聚合器：每变体每窗口
    agg = {v: {
        "seed_crps": np.zeros((Nw, len(seeds))), "seed_crps_h": np.zeros((Nw, len(seeds), H)),
        "seed_rmse": np.zeros((Nw, len(seeds))), "seed_cov": np.zeros((Nw, len(seeds))),
        "ens_crps": np.zeros(Nw), "ens_crps_h": np.zeros((Nw, H)),
        "ens_rmse": np.zeros(Nw), "ens_cov": np.zeros(Nw),
        "crps_persist": np.zeros(Nw),
        "layer_rmse": np.zeros((Nw, len(conc_cols))),
        "recon_rmse_overall": np.zeros(Nw), "recon_rmse_surface": np.zeros(Nw),
        "recon_rmse_surface_h": np.zeros((Nw, H)),
    } for v in variants}
    per_window_meta = []

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        Xw, cur_full, y_full, strat_w, warn_w, conc_cols_ = build_window(wide, i0, i1, feat_cols)
        assert conc_cols_ == conc_cols
        n_win = len(Xw)
        n_tr = int(n_win * TRAIN_DAYS / (TRAIN_DAYS + TEST_DAYS))
        te_sl = slice(n_tr, n_win)

        Xte = Xw[te_sl]
        Nte = len(Xte)
        cur_te_full = cur_full[te_sl]      # (N, 20)
        y_te_full = y_full[te_sl]          # (N, 20, H)

        # 各变体观测/当前（不同 n_layer）
        layer_sets = {"surface": ["conc_0.5"], "full": conc_cols, "key": KEY_COLS}
        per_window_meta.append({"window": wi + 1, "start": str(st), "end": str(en),
                                "n_test": Nte})

        for v in variants:
            sel = layer_sets[v]
            NL = len(sel)
            cur_te, y_te = select_layers(cur_te_full, y_te_full, conc_cols, sel)
            # abs_delta 目标 + 尺度（窗口训练段拟合，防泄漏）
            cur_raw_w, y_abs_w = select_layers(cur_full, y_full, conc_cols, sel)
            raw = make_targets_abs_delta(cur_raw_w, y_abs_w)        # (n_w, NL, H)
            scale = float(np.std(raw[:n_tr])) + 1e-8
            y_norm = (raw / scale).astype(np.float32)

            qs_all = np.zeros((len(seeds), Nte, NL, 3, H), dtype=np.float64)
            for si, seed in enumerate(seeds):
                model = train_model(Xw, y_norm, strat_w, warn_w, n_tr, NL,
                                    epochs, args.device, seed)
                qs_all[si] = predict_quantiles(model, Xte, args.device)
            # 还原 conc 单位（abs_delta：conc = cur + Δ）
            qs_all = cur_te[None, :, :, None, None] + qs_all * scale
            # (1, N, NL, 1, 1) broadcast → (S, N, NL, 3, H)

            m = per_variant_metrics(v, qs_all, cur_te, y_te)
            agg[v]["seed_crps"][wi] = m["seed_crps"]
            agg[v]["seed_crps_h"][wi] = m["seed_crps_h"]
            agg[v]["seed_rmse"][wi] = m["seed_rmse"]
            agg[v]["seed_cov"][wi] = m["seed_cov"]
            agg[v]["ens_crps"][wi] = m["ens_crps"]
            agg[v]["ens_crps_h"][wi] = m["ens_crps_h"]
            agg[v]["ens_rmse"][wi] = m["ens_rmse"]
            agg[v]["ens_cov"][wi] = m["ens_cov"]
            agg[v]["crps_persist"][wi] = m["crps_persist_surface"]
            sel_idx = [conc_cols.index(c) for c in sel]
            agg[v]["layer_rmse"][wi, sel_idx] = m["layer_rmse"]
            agg[v]["recon_rmse_overall"][wi] = m["recon_rmse_overall"]
            agg[v]["recon_rmse_surface"][wi] = m["recon_rmse_surface"]
            agg[v]["recon_rmse_surface_h"][wi] = m["recon_rmse_surface_h"]

            print(f"    —— 变体 {v}（{NL} 层）——", flush=True)
            print(f"        [表层] ens CRPS={m['ens_crps']:.4f} (持久化 {m['crps_persist_surface']:.4f}, "
                  f"技能 {(m['crps_persist_surface'] - m['ens_crps']) / m['crps_persist_surface'] * 100:+.1f}%)  "
                  f"RMSE={m['ens_rmse']:.3f}  覆盖={m['ens_cov']:.3f}", flush=True)
            print(f"        [seed 范围] CRPS {min(m['seed_crps']):.4f}~{max(m['seed_crps']):.4f}  "
                  f"RMSE {min(m['seed_rmse']):.3f}~{max(m['seed_rmse']):.3f}", flush=True)
            if v in ("full", "key"):
                lay = m["layer_rmse"]
                print(f"        [重建 RMSE] 各层 {[f'{x:.2f}' for x in lay]}  全层 {m['recon_rmse_overall']:.3f}",
                      flush=True)

    # ---- 聚合输出 ----
    print("\n===== 3 变体对照（表层指标，全部还原 conc 单位，跨 seed 集成）=====", flush=True)
    print(f"  {'变体':<10}{'表层CRPS':<10}{'持久化CRPS':<12}{'相对技能':<10}{'表层RMSE':<10}"
          f"{'表层覆盖':<10}{'seedCRPS范围'}", flush=True)
    for v in variants:
        a = agg[v]
        cp = a["crps_persist"].mean()
        rel = (cp - a["ens_crps"].mean()) / cp * 100 if cp else 0
        seed_crps_all = a["seed_crps"].reshape(-1)
        print(f"  {v:<10}{a['ens_crps'].mean():<10.4f}{cp:<12.4f}{rel:<10.1f}"
              f"{a['ens_rmse'].mean():<10.3f}{a['ens_cov'].mean():<10.3f}"
              f"{seed_crps_all.min():.4f}~{seed_crps_all.max():.4f}", flush=True)

    print("\n===== 全剖面重建质量（full 变体）=====", flush=True)
    print("  表层/中层(5.0m)/底层(10.0m) 重建 RMSE + 全层平均：", flush=True)
    for dname, col in (("表层", "conc_0.5"), ("中层", "conc_5.0"), ("底层", "conc_10.0")):
        idx = conc_cols.index(col)
        val = np.mean([agg["full"]["layer_rmse"][wi, idx] for wi in range(Nw)])
        print(f"    {dname} ({col}): 重建 RMSE {val:.3f}", flush=True)
    print(f"    全层（20 层全视界加权）: 重建 RMSE {agg['full']['recon_rmse_overall'].mean():.3f}", flush=True)
    print(f"    表层重建 RMSE（还原 conc 单位）: {agg['full']['recon_rmse_surface'].mean():.3f}", flush=True)

    # 输出 JSON
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": TRAIN_DAYS, "test_days": TEST_DAYS, "stride_days": STRIDE_DAYS,
                     "T": T, "H": H, "epochs": args.epochs, "n_windows": Nw,
                     "seeds": seeds, "n_layer_all": len(conc_cols),
                     "key_layers": KEY_COLS},
        "variants": {},
        "windows_meta": per_window_meta,
    }
    for v in variants:
        a = agg[v]
        res["variants"][v] = {
            "n_layer": len(layer_sets[v]),
            "surface_crps_mean": float(a["ens_crps"].mean()),
            "surface_crps_h": a["ens_crps_h"].mean(axis=0).tolist(),
            "surface_crps_windows": a["ens_crps"].tolist(),
            "surface_crps_persist": float(a["crps_persist"].mean()),
            "surface_rmse_mean": float(a["ens_rmse"].mean()),
            "surface_cov_mean": float(a["ens_cov"].mean()),
            "surface_cov_std": float(np.std(a["ens_cov"])),
            "seed_crps_min": float(a["seed_crps"].min()),
            "seed_crps_max": float(a["seed_crps"].max()),
            "layer_rmse_mean": np.mean(a["layer_rmse"], axis=0).tolist(),
            "recon_rmse_overall": float(a["recon_rmse_overall"].mean()),
            "recon_rmse_surface": float(a["recon_rmse_surface"].mean()),
            "recon_rmse_surface_h": a["recon_rmse_surface_h"].mean(axis=0).tolist(),
        }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
