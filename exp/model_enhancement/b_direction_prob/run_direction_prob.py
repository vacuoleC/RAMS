# -*- coding: utf-8 -*-
"""B_direction_prob 探索：方向概率 P(Δ>0) 判别力实证

背景（B2/B7 已证）：增量分位数模型 p50 的**方向判别弱**（sign 命中 0.53≈随机）。
B7 排除"目标尺度"因素后，方向判别弱是 Δ 本身难判别；建议另配 P(Δ>0) 概率输出 + 阈值校准。
本实验回答三个问题：
  Q1 P(Δ>0) 是否比 p50 符号更有判别力？（ROC-AUC / 方向命中率）
  Q2 概率校准是否可靠？（Brier / ECE / 校准曲线）
  Q3 方向是否**本质上不可判别**？（若全部方法 AUC≈0.5 且校准可靠，= 数据限制而非方法缺陷）

方法（同一滚动窗口协议，同 B7：窗口 820d / 步长 45d，17 窗口；
每窗口内 训练 640d / 校准 90d / 测试 90d——校准段与测试段分离防泄漏）：
  - 模型：GRU 骨干 + M1 p10/p50/p90 分位数头（abs_delta 目标）+ M2 分层头 + M4 预警头
    + **方向头**（B,H 二元 logits，与 M1/M2/M4 联合训练，BCE）——多任务架构同 B7
  - P(Δ>0) 来源 4 种：
    1. p50 符号（B7 基线，非概率，方向 = sign(q50_delta)）
    2. 分位数反推-高斯：Φ(q50/σ)，σ=(q90-q10)/2.563
    3. 分位数反推-PCHIP：单调三次 CDF 插值求 F(0)（线性 CDF 作对照）
    4. 二分类方向头：sigmoid(dir_logits)
  - 阈值校准：校准段逐视界扫 τ 使方向命中最大 → 测试段应用（acc@τ_cal）
  - 概率校准：校准段 Isotonic → 测试段应用（Brier_cal / ECE_cal）
  - 评估（测试段，逐视界 h=1..8）：acc@0.5、acc@τ_cal、ROC-AUC、Brier（原始+校准）、
    ECE（10 bin 校准曲线）、平凡基线（常量预测器 max(up_rate, 1-up_rate)）

3 seed（0/1/2）× 17 窗口。
理论备忘：凡 P 是 q50 的单调函数（高斯/PCHIP/线性），其 ROC-AUC 与 acc@0.5 与 p50 符号
**恒等**（单调变换保序）——只有方向头可能真正改变判别力；AUC 是阈值无关的诚实判别指标。

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
import torch.nn as nn  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS  # noqa: E402
from rams.models.rams_net import SharedGRU, M1Head, M2Head, M4Head  # noqa: E402

T, H = 24, 8
EPOCHS = 30
QUANTILES = (0.1, 0.5, 0.9)

# 滚动窗口参数（天，3h 网格：1 天 = 8 时刻）
TRAIN_DAYS = 640     # 实际训练（B7 的 730d 中抽出 90d 作校准段）
CAL_DAYS = 90        # 校准段（阈值 / Isotonic 拟合，独立于测试）
TEST_DAYS = 90       # 测试段（与 B7 测试段完全一致：窗口最后 90d）
STRIDE_DAYS = 45
GRID_PER_DAY = 8
DAYS_TOTAL = TRAIN_DAYS + CAL_DAYS + TEST_DAYS  # 820 = 同 B7 窗口跨度

SEEDS = [0, 1, 2]

# 多任务权重（M1/M2/M4 同 B7；方向头 w_dir 本实验新增）
W_M1, W_M2, W_M4, W_DIR = 1.0, 3.0, 2.0, 1.0


class DirectionNet(nn.Module):
    """B7 RamsNet + 方向头（B,H 二元 logits，P(Δ>0) 用）。

    forward 返回 (m1, m2, m4, dir_logits)：
      m1: (B,3H) p10/p50/p90（abs_delta 归一化尺度）
      m2: (B,2) 分层分类
      m4: (B,4) 预警分级
      dir: (B,H) 每视界 Δ>0 的 logits
    """

    def __init__(self, feat_dim: int, horizon: int, hidden: int = 64,
                 n_layers: int = 1, n_classes: int = 2, n_levels: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden, n_layers, dropout)
        self.m1 = M1Head(hidden, horizon, quantile=True)
        self.m2 = M2Head(hidden, n_classes)
        self.m4 = M4Head(hidden, n_levels)
        self.dir = M1Head(hidden, horizon, quantile=False)  # (B,H) logits
        self.horizon = horizon

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        return self.m1(h), self.m2(h), self.m4(h), self.dir(h)


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


def build_window(wide, i0, i1, feat_cols, strat_col="delta_T"):
    """窗口 [i0,i1)：返回特征/目标/标签 + 训练/校准/测试样本切分索引。

    与 B7 相同的窗口化方式，但窗口内按 640/90/90 天分训练/校准/测试。
    测试段 = B7 测试段（窗口最后 90d），可与 B7 数字直接对照。
    Returns:
      Xw (n_w,T,F), cur_raw (n_w,), y_abs (n_w,H),
      strat_w (n_w,), warn_w (n_w,), up (n_w,H) bool,
      n_tr, n_va  （样本数切分：train[:n_tr], cal[n_tr:n_tr+n_va], test[n_tr+n_va:]）
    """
    df = wide.iloc[i0:i1]
    n = len(df)
    n_tr_rows = int(n * TRAIN_DAYS / DAYS_TOTAL)   # 特征标准化用训练行

    # 特征标准化（只用训练段）
    Xtr = df[feat_cols].values[:n_tr_rows].astype(np.float32)
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0) + 1e-8
    X = ((df[feat_cols].values.astype(np.float32) - mu) / sd).astype(np.float32)

    y_raw = df["conc_0.5"].values.astype(np.float64)
    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)
    cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)

    # M2 分层标签（训练段中位阈值）
    delta = df[strat_col].values
    thr = float(np.median(delta[:n_tr_rows]))
    strat_w = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)

    # M4 预警标签（训练段峰值分位数阈值）
    warn_val = y_abs.max(axis=1)
    n_win_tr = int(len(y_abs) * TRAIN_DAYS / DAYS_TOTAL)
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    # 方向标签：up = Δ>0（Δ=0 记作"不上"；占比约 1.3%，影响可忽略）
    up = (y_abs - cur_raw[:, None]) > 0

    n_tr = int(n_w * TRAIN_DAYS / DAYS_TOTAL)
    n_va = int(n_w * CAL_DAYS / DAYS_TOTAL)
    return Xw, cur_raw, y_abs, strat_w, warn_w, up.astype(np.float32), n_tr, n_va


def quantile_loss(m1_out, y, device):
    """M1 分位数损失（同 MultiTaskLoss，p10/p50/p90）。"""
    qs = torch.tensor(QUANTILES, device=device)
    yq = y.unsqueeze(1)
    m1_q = m1_out.reshape(-1, 3, H)
    e = yq - m1_q
    losses = [torch.mean(torch.maximum(q * e[:, i], (q - 1) * e[:, i]))
              for i, q in enumerate(qs)]
    return torch.stack(losses).mean()


def train_model(Xw, y_norm, strat_w, warn_w, up, n_tr, epochs, device, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    model = DirectionNet(feat_dim=Xw.shape[2], horizon=H).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ce2 = nn.CrossEntropyLoss()
    counts = np.bincount(warn_w[:n_tr], minlength=4)
    inv = 1.0 / (counts.astype(np.float64) + 1.0)
    w4w = torch.tensor(inv / inv.sum() * 4, dtype=torch.float32, device=device)
    ce4 = nn.CrossEntropyLoss(weight=w4w)
    bce = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(Xw[:n_tr])
    ytr = torch.tensor(y_norm[:n_tr])
    str = torch.tensor(strat_w[:n_tr])
    warn_tr = torch.tensor(warn_w[:n_tr])
    up_tr = torch.tensor(up[:n_tr])
    ds = torch.utils.data.TensorDataset(Xtr, ytr, str, warn_tr, up_tr)
    dl = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False)

    for ep in range(epochs):
        model.train()
        for xb, yb, sb, wb, ub in dl:
            xb = xb.to(device); yb = yb.to(device); sb = sb.to(device)
            wb = wb.to(device); ub = ub.to(device)
            opt.zero_grad()
            m1, m2, m4, dlog = model(xb)
            l1 = quantile_loss(m1, yb, device)
            l2 = ce2(m2, sb)
            l4 = ce4(m4, wb)
            l5 = bce(dlog, ub)
            total = W_M1 * l1 + W_M2 * l2 + W_M4 * l4 + W_DIR * l5
            total.backward()
            opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                Xv = torch.tensor(Xw[n_tr:]).to(device)
                m1v, m2v, m4v, dv = model(Xv)
                rmse = torch.sqrt(torch.mean((m1v[:, H:2 * H] - torch.tensor(y_norm[n_tr:]).to(device)) ** 2)).item()
                dir_acc = ((dv.sigmoid() > 0.5).float()
                           == torch.tensor(up[n_tr:]).to(device)).float().mean().item()
            print(f"  ep{ep} loss={total.item():.4f} val_rmse={rmse:.4f} dir_acc={dir_acc:.4f}", flush=True)
    return model


def predict_all(model, X, device):
    """返回 (q_conc(N,3,H), P_dir(N,H))：q 为 conc 单位分位数（需 cur 还原），P_dir 为方向头 sigmoid。"""
    model.eval()
    with torch.no_grad():
        m1, _, _, dlog = model(torch.tensor(X).to(device))
        q = torch.stack([m1[:, :H], m1[:, H:2 * H], m1[:, 2 * H:]], dim=1).cpu().numpy()
        pdir = torch.sigmoid(dlog).cpu().numpy()
    return q.astype(np.float64), pdir.astype(np.float64)


# ---------- 分位数 → P(Δ>0) 反推 ----------

def p_gauss(q10, q50, q90):
    """P = Φ(q50/σ), σ=(q90-q10)/2.563（高斯）。"""
    sigma = (q90 - q10) / 2.563
    with np.errstate(all="ignore"):
        z = np.where(sigma > 1e-9, q50 / np.maximum(sigma, 1e-12),
                     np.where(q50 > 0, 5.0, np.where(q50 < 0, -5.0, 0.0)))
        p = norm.cdf(z)
    return np.clip(p, 1e-6, 1 - 1e-6)


def _linear_cdf_at_zero(q10, q50, q90):
    """线性分段 CDF（5 结点 x0/q10/q50/q90/x4，概率 0/0.1/0.5/0.9/1）→ F(0)。

    与 PCHIP 同一插值框架（线性 vs 单调三次），只在段内做线性插值，
    0 在结点外时取端点概率（0 或 1）。输入可为任意形状，展平计算后还原。
    """
    shape = np.broadcast(q10, q50, q90).shape
    q10 = np.asarray(q10, dtype=np.float64).reshape(-1)
    q50 = np.asarray(q50, dtype=np.float64).reshape(-1)
    q90 = np.asarray(q90, dtype=np.float64).reshape(-1)
    x0 = q10 - (q50 - q10) / 4.0
    x4 = q90 + (q90 - q50) / 4.0
    x = np.stack([x0, q10, q50, q90, x4], axis=-1)   # (N,5)
    y = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    # 段索引 = (#元素 ≤ 0) - 1（searchsorted 的矢量化形式，x 每行升序）
    idx = np.clip(np.sum(x <= 0.0, axis=-1) - 1, 0, 4)
    # 0 在第一个结点左侧 → F=0；在最后一个结点右侧 → F=1
    left_of = x[..., 0] >= 0.0
    right_of = x[..., 4] <= 0.0
    xL = x[np.arange(len(x)), idx]
    xR = x[np.arange(len(x)), np.clip(idx + 1, 0, 4)]
    yL = y[idx]; yR = y[np.clip(idx + 1, 0, 4)]
    hd = xR - xL
    tt = (0.0 - xL) / np.where(hd == 0, 1.0, hd)
    F = yL + tt * (yR - yL)
    F = np.where(left_of, 0.0, np.where(right_of, 1.0, F))
    return np.clip(F, 0.0, 1.0).reshape(shape)


def _pchip_cdf_at_zero(q10, q50, q90):
    """PCHIP 单调三次 CDF 插值（Fritsch–Carlson 斜率）→ F(0)。矢量实现，任意形状。"""
    shape = np.broadcast(q10, q50, q90).shape
    q10 = np.asarray(q10, dtype=np.float64).reshape(-1)
    q50 = np.asarray(q50, dtype=np.float64).reshape(-1)
    q90 = np.asarray(q90, dtype=np.float64).reshape(-1)
    out = np.full_like(q10, 0.5)
    deg = (q90 - q10) < 1e-9
    # 退化（区间塌缩）：按 q50 符号给确定性，否则 0.5
    out = np.where(deg & (q50 > 1e-9), 0.0,
                   np.where(deg & (q50 < -1e-9), 1.0, out))
    nd = ~deg
    if nd.any():
        q10 = q10[nd]; q50 = q50[nd]; q90 = q90[nd]
        x0 = q10 - (q50 - q10) / 4.0
        x4 = q90 + (q90 - q50) / 4.0
        x = np.stack([x0, q10, q50, q90, x4], axis=-1)   # (N,5)
        y = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        h = np.diff(x, axis=-1)                          # (N,4)
        s = np.diff(y)[None, :] / h                      # (N,4) secant
        d = np.zeros_like(x)
        # 端点：单侧割线
        d[..., 0] = s[..., 0]
        d[..., 4] = s[..., 3]
        # 内部：Fritsch-Carlson 单调斜率
        for i in (1, 2, 3):
            w1 = 2 * h[..., i - 1] + h[..., i]
            w2 = h[..., i - 1] + 2 * h[..., i]
            d1 = s[..., i - 1]; d2 = s[..., i]
            sign_chg = np.sign(d1) != np.sign(d2)
            denom = w1 / d1 + w2 / d2
            d[..., i] = np.where(sign_chg | (np.abs(denom) < 1e-12),
                                 0.0, (w1 + w2) / np.where(np.abs(denom) < 1e-12, 1.0, denom))
        # 找 0 所在段（searchsorted 矢量化：x 每行升序）
        idx = np.clip(np.sum(x <= 0.0, axis=-1) - 1, 0, 3)
        ar = np.arange(len(x))
        xL = x[ar, idx]
        xR = x[ar, idx + 1]
        yL = y[idx]; yR = y[idx + 1]
        dL = d[ar, idx]
        dR = d[ar, idx + 1]
        hd = xR - xL
        tt = (0.0 - xL) / np.where(hd == 0, 1.0, hd)
        h00 = 2 * tt ** 3 - 3 * tt ** 2 + 1
        h10 = tt ** 3 - 2 * tt ** 2 + tt
        h01 = -2 * tt ** 3 + 3 * tt ** 2
        h11 = tt ** 3 - tt ** 2
        F = h00 * yL + h10 * hd * dL + h01 * yR + h11 * hd * dR
        F = np.clip(F, 0.0, 1.0)
        # 边界：0 ≤ min 结点 → F=0（Δ 必>0）；0 ≥ max 结点 → F=1
        F = np.where(x[..., 0] >= 0.0, 0.0, np.where(x[..., 4] <= 0.0, 1.0, F))
        out = out.copy()
        out[nd] = F
    return np.clip(out, 0.0, 1.0).reshape(shape)


# ---------- 评估 ----------

def acc_at(P, y, tau):
    return float(np.mean((P > tau) == y))


def ece_binned(P, y, n_bins=10):
    """期望校准误差：预测概率分箱 vs 实际频率。返回 (ece, bin_table)。"""
    P = np.asarray(P, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    b = np.clip(np.floor(P * n_bins).astype(np.int64), 0, n_bins - 1)
    table = []
    ece = 0.0
    N = len(P)
    for k in range(n_bins):
        m = b == k
        n = int(m.sum())
        if n == 0:
            continue
        pbar = float(P[m].mean())
        obar = float(y[m].mean())
        ece += (n / N) * abs(pbar - obar)
        table.append({"bin": k, "n": n, "p_mean": pbar, "obs_rate": obar})
    return ece, table


def eval_method(P, y):
    """P(N,H) 概率预测 vs y(N,H) 二元 → 每视界指标。"""
    H_ = P.shape[1]
    m = {k: np.full(H_, np.nan) for k in
         ["acc05", "auc", "brier", "ece"]}
    for h in range(H_):
        p = P[:, h]; yy = y[:, h]
        m["acc05"][h] = acc_at(p, yy, 0.5)
        m["brier"][h] = float(np.mean((p - yy) ** 2))
        ece, _ = ece_binned(p, yy)
        m["ece"][h] = ece
        if yy.min() != yy.max():  # 两类都有才可算 AUC
            try:
                m["auc"][h] = float(roc_auc_score(yy, p))
            except ValueError:
                m["auc"][h] = float("nan")
    return m


def calibrate_val_to_test(P_cal, y_cal, P_te, y_te):
    """阈值校准（校准段扫 τ 最优）+ Isotonic 概率校准 → 测试段指标。

    返回 dict：每视界 acc_tau / acc_iso / brier_iso / ece_iso。
    """
    H_ = P_cal.shape[1]
    out = {k: np.full(H_, np.nan) for k in ["acc_tau", "acc_iso", "brier_iso", "ece_iso", "tau"]}
    taus = np.linspace(0.02, 0.98, 97)
    for h in range(H_):
        pc = P_cal[:, h]; yc = y_cal[:, h]
        pt = P_te[:, h]; yt = y_te[:, h]
        # 阈值校准：校准段逐 τ 扫方向命中
        accs = [acc_at(pc, yc, tau) for tau in taus]
        best = int(np.argmax(accs))
        tau = float(taus[best])
        out["tau"][h] = tau
        out["acc_tau"][h] = acc_at(pt, yt, tau)
        # Isotonic 概率校准
        if pc.min() != pc.max() and yc.min() != yc.max():
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(pc, yc)
            pcal = iso.predict(pt)
        else:
            pcal = np.full_like(pt, np.clip(yc.mean(), 1e-6, 1 - 1e-6))
        out["acc_iso"][h] = acc_at(pcal, yt, 0.5)
        out["brier_iso"][h] = float(np.mean((pcal - yt) ** 2))
        ece, _ = ece_binned(pcal, yt)
        out["ece_iso"][h] = ece
    return out


def main():
    ap = argparse.ArgumentParser(description="B_direction_prob P(Δ>0) 判别力实证")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-json", default="exp/model_enhancement/b_direction_prob/results.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    t0 = time.time()
    print(f"== B_direction_prob：P(Δ>0) 判别力实证 ==", flush=True)
    print(f"   seed: {seeds} | 协议: 训练 {TRAIN_DAYS}d / 校准 {CAL_DAYS}d / 测试 {TEST_DAYS}d / "
          f"步长 {STRIDE_DAYS}d（窗口 820d 同 B7，测试段一致）", flush=True)
    print(f"   权重 W_M1/M2/M4/DIR = {W_M1}/{W_M2}/{W_M4}/{W_DIR} | 方向头 BCE 联合训练", flush=True)

    wide = load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列（3h 网格）", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    if "conc_0.5" not in feat_cols:
        feat_cols = feat_cols + ["conc_0.5"]
    print(f"   特征列 {len(feat_cols)} 个（temp_* + 气象 + conc_t）", flush=True)

    days = DAYS_TOTAL
    windows = []
    for i0 in range(0, n - days * GRID_PER_DAY + 1, STRIDE_DAYS * GRID_PER_DAY):
        windows.append((i0, i0 + days * GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    Nw = len(windows)
    print(f"[2] 滚动窗口 {Nw} 个", flush=True)

    methods = ["p50_sign", "gauss", "linear", "pchip", "binary"]
    # 聚合器：每方法 每视界（seed×window 求均值/方差）
    agg = {m: {k: [] for k in ["acc05", "acc_tau", "acc_iso", "auc", "brier", "brier_iso",
                                "ece", "ece_iso", "tau"]}
           for m in methods}
    agg["p50_sign"] = {"acc05": []}  # 非概率方法只需 acc
    baseline_acc = {h: [] for h in range(H)}
    up_rate_te_all = []
    model_sanity = {"coverage": [], "crps": []}
    window_rows = []
    calib_bins = {m: [] for m in ["gauss", "linear", "pchip", "binary"]}   # 校准曲线 bin 表（seed×窗口）

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}  "
              f"({(en - st).days} 天)", flush=True)

        Xw, cur_raw, y_abs, strat_w, warn_w, up, n_tr, n_va = build_window(wide, i0, i1, feat_cols)
        n_win = len(Xw)
        te_sl = slice(n_tr + n_va, n_win)
        ca_sl = slice(n_tr, n_tr + n_va)

        Xte = Xw[te_sl]; cur_te = cur_raw[te_sl]; y_te = y_abs[te_sl]; up_te = up[te_sl]
        Xca = Xw[ca_sl]; cur_ca = cur_raw[ca_sl]; y_ca = y_abs[ca_sl]; up_ca = up[ca_sl]
        Nte = len(Xte)

        up_rate_te = float(up_te.mean())
        up_rate_te_all.append(up_rate_te)
        baseline_acc_this = {h: max(float(up_te[:, h].mean()), 1 - float(up_te[:, h].mean()))
                             for h in range(H)}
        for h in range(H):
            baseline_acc[h].append(baseline_acc_this[h])

        # 归一化目标（abs_delta，训练段 scale 防泄漏）
        raw = y_abs - cur_raw[:, None]
        scale = float(np.std(raw[:n_tr])) + 1e-8
        y_norm = (raw / scale).astype(np.float32)

        per_seed = {m: {k: [] for k in agg[m].keys()} for m in methods}
        per_seed["p50_sign"] = {"acc05": []}
        model_sanity_this = {"coverage": [], "crps": []}

        for sd in seeds:
            model = train_model(Xw, y_norm, strat_w, warn_w, up, n_tr, args.epochs, args.device, sd)
            q_ca, pdir_ca = predict_all(model, Xca, args.device)
            q_te, pdir_te = predict_all(model, Xte, args.device)

            # conc 单位 Δ 分位数：M1 头输出归一化 Δ（y_norm=raw/scale），乘 scale 还原
            qd_te = q_te * scale   # (N,3,H)
            qd_ca = q_ca * scale
            q10_te, q50_te, q90_te = qd_te[:, 0], qd_te[:, 1], qd_te[:, 2]
            q10_ca, q50_ca, q90_ca = qd_ca[:, 0], qd_ca[:, 1], qd_ca[:, 2]

            # 各方法 P(Δ>0)
            P = {}
            P["p50_sign"] = (q50_te > 0).astype(np.float64)   # 二值"方向"
            P["gauss"] = p_gauss(q10_te, q50_te, q90_te)
            P["linear"] = 1.0 - _linear_cdf_at_zero(q10_te, q50_te, q90_te)
            P["pchip"] = 1.0 - _pchip_cdf_at_zero(q10_te, q50_te, q90_te)
            P["binary"] = pdir_te
            P_cal = {}
            # 校准段 P（与测试段同一公式）
            P_cal["gauss"] = p_gauss(q10_ca, q50_ca, q90_ca)
            P_cal["linear"] = 1.0 - _linear_cdf_at_zero(q10_ca, q50_ca, q90_ca)
            P_cal["pchip"] = 1.0 - _pchip_cdf_at_zero(q10_ca, q50_ca, q90_ca)
            P_cal["binary"] = pdir_ca

            # p50 sign：acc（B7 口径 sign 命中）
            pred_sign = (q50_te > 0)
            true_sign = up_te.astype(bool)
            per_seed["p50_sign"]["acc05"].append(
                np.mean(pred_sign == true_sign, axis=0))   # (H,)

            # 概率方法
            for m in ["gauss", "linear", "pchip", "binary"]:
                ev = eval_method(P[m], up_te)
                cal = calibrate_val_to_test(P_cal[m], up_ca, P[m], up_te)
                for k in ev:
                    per_seed[m][k].append(ev[k])
                for k in cal:
                    per_seed[m][k].append(cal[k])
                # 校准曲线 bin 表（测试段，合并全部视界）
                for h in range(H):
                    _, tbl = ece_binned(P[m][:, h], up_te[:, h])
                    calib_bins[m].append({"window": wi + 1, "seed": sd, "horizon": h + 1, "bins": tbl})

            # 模型 sanity（M1 质量，conc 单位：还原后覆盖率/CRPS 与 B7 对照）
            q_conc_te = cur_te[:, None, None] + qd_te   # conc 单位 p10/p50/p90（Δ 还原 + cur）
            cov = float(np.mean((y_te >= q_conc_te[:, 0]) & (y_te <= q_conc_te[:, 2])))
            crps_h = [float(np.mean(crps_quantiles(q_conc_te[:, 0, h], q_conc_te[:, 1, h],
                                                   q_conc_te[:, 2, h], y_te[:, h])))
                      for h in range(H)]
            model_sanity_this["coverage"].append(cov)
            model_sanity_this["crps"].append(float(np.mean(crps_h)))

        # 窗口内聚合（seed 均值）→ 汇总
        window_rows.append({"window": wi + 1, "start": str(st), "end": str(en),
                            "up_rate_te": up_rate_te,
                            "p50_sign_acc": float(np.mean(per_seed["p50_sign"]["acc05"]))})
        for m in methods:
            for k in per_seed[m]:
                arr = np.array(per_seed[m][k])   # (seed, H)
                agg[m][k].append(arr.mean(axis=0))
        model_sanity["coverage"].append(float(np.mean(model_sanity_this["coverage"])))
        model_sanity["crps"].append(float(np.mean(model_sanity_this["crps"])))

        print(f"    [test] 测试段 n={Nte}  up_rate={up_rate_te:.3f}  常量基线 acc="
              f"{np.mean(list(baseline_acc_this.values())):.3f}", flush=True)
        print(f"    p50_sign acc={np.mean(per_seed['p50_sign']['acc05']):.3f}  "
              f"binary acc05={np.nanmean(per_seed['binary']['acc05']):.3f}  "
              f"binary AUC={np.nanmean(per_seed['binary']['auc']):.3f}  "
              f"binary Brier={np.nanmean(per_seed['binary']['brier']):.3f}", flush=True)
        print(f"    [sanity] M1 覆盖率={model_sanity['coverage'][-1]:.3f}  "
              f"CRPS={model_sanity['crps'][-1]:.4f}（B7 参考 0.794 / 0.895）", flush=True)

    # ---- 汇总 ----
    def agg_arr(m, key):
        """返回 (Nw, H) 数组：窗口 × 视界 的 seed 均值。"""
        return np.array(agg[m][key]) if len(agg[m][key]) else np.full((1, H), np.nan)

    print("\n===== 聚合（17 窗口 × 3 seed，逐视界均值再整体平均）=====", flush=True)
    print(f"  up_rate 漂移: 均值 {np.mean(up_rate_te_all):.3f}  std {np.std(up_rate_te_all):.3f}  "
          f"[min {np.min(up_rate_te_all):.3f}, max {np.max(up_rate_te_all):.3f}]", flush=True)
    print(f"  常量基线 acc（测试段 max(up,1-up) 平均）: {np.mean([np.mean(v) for v in baseline_acc.values()]):.3f}",
          flush=True)
    print(f"  [模型 sanity] M1 覆盖 {np.mean(model_sanity['coverage']):.3f}（B7=0.794） "
          f"CRPS {np.mean(model_sanity['crps']):.4f}（B7=0.895）", flush=True)

    def overall(m, key):
        return float(np.nanmean(agg_arr(m, key)))

    header = ["方法", "acc@0.5", "acc@τ_cal", "acc@iso", "AUC", "Brier", "Brier_iso", "ECE", "ECE_iso"]
    keymap = {"acc@0.5": "acc05", "acc@τ_cal": "acc_tau", "acc@iso": "acc_iso",
              "AUC": "auc", "Brier": "brier", "Brier_iso": "brier_iso",
              "ECE": "ece", "ECE_iso": "ece_iso"}
    print(f"  {'方法':<10}" + "".join(f"{h:>11}" for h in header[1:]), flush=True)
    for m in methods:
        row = [f"{m:<10}"]
        for k in header[1:]:
            key = keymap[k]
            if key in agg[m]:
                row.append(f"{overall(m, key):.3f}")
            else:
                row.append(f"{'-':>11}")
        print("".join(row), flush=True)

    # 逐视界表（binary / p50_sign / gauss，AUC 与 acc05）
    print("\n===== 逐视界（binary / p50_sign / gauss，AUC 与 acc05）=====", flush=True)
    for h in range(H):
        def per_h(m, key):
            return float(np.nanmean(agg_arr(m, key)[:, h]))
        print(f"  h{h + 1}:  p50_sign acc={per_h('p50_sign', 'acc05'):.3f}  "
              f"binary acc05={per_h('binary', 'acc05'):.3f} AUC={per_h('binary', 'auc'):.3f}  "
              f"gauss AUC={per_h('gauss', 'auc'):.3f}  const_baseline="
              f"{np.mean(baseline_acc[h]):.3f}", flush=True)

    # 校准曲线（合并全部 seed×窗口×视界的 bin 表 → 每方法 10 bin 均值 p_mean / obs_rate）
    print("\n===== 校准曲线（P(Δ>0) 预测概率 vs 实际频率，合并 17窗×3seed×8视界）=====", flush=True)
    calib_summary = {}
    for m in ["gauss", "pchip", "binary"]:
        nb = 10
        pm = np.zeros(nb); om = np.zeros(nb); nm = np.zeros(nb)
        for rec in calib_bins[m]:
            for b in rec["bins"]:
                k = b["bin"]
                nm[k] += b["n"]
                pm[k] += b["p_mean"] * b["n"]
                om[k] += b["obs_rate"] * b["n"]
        pm = np.where(nm > 0, pm / np.maximum(nm, 1), np.nan)
        om = np.where(nm > 0, om / np.maximum(nm, 1), np.nan)
        calib_summary[m] = {"bins": [{"bin": int(k), "n": int(nm[k]),
                                      "p_mean": float(pm[k]), "obs_rate": float(om[k])}
                                     for k in range(nb)]}
        print(f"  {m:<8} p_mean → obs_rate:", flush=True)
        for k in range(nb):
            if nm[k] > 0:
                print(f"    bin{k} (n={int(nm[k])}): pred {pm[k]:.3f} → obs {om[k]:.3f}", flush=True)

    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    res = {
        "protocol": {"train_days": TRAIN_DAYS, "cal_days": CAL_DAYS, "test_days": TEST_DAYS,
                     "stride_days": STRIDE_DAYS, "T": T, "H": H, "epochs": args.epochs,
                     "n_windows": Nw, "seeds": seeds,
                     "weights": {"w_m1": W_M1, "w_m2": W_M2, "w_m4": W_M4, "w_dir": W_DIR},
                     "target": "abs_delta (conc_{t+h}-conc_t), normalized"},
        "up_rate_te": {"mean": float(np.mean(up_rate_te_all)),
                       "std": float(np.std(up_rate_te_all)),
                       "min": float(np.min(up_rate_te_all)),
                       "max": float(np.max(up_rate_te_all))},
        "baseline_const_acc": float(np.mean([np.mean(v) for v in baseline_acc.values()])),
        "model_sanity": {"coverage_mean": float(np.mean(model_sanity["coverage"])),
                         "crps_mean": float(np.mean(model_sanity["crps"]))},
        "per_window": window_rows,
        "calibration_curve": calib_summary,
        "methods": {},
    }
    for m in methods:
        entry = {}
        for k in agg[m]:
            arr = np.array(agg[m][k])          # (window, H)
            entry[k] = {
                "mean": [float(np.nanmean(arr[:, h])) for h in range(H)],
                "overall": float(np.nanmean(arr)),
                "window_std": [float(np.nanstd(arr[:, h])) for h in range(H)],
            }
        res["methods"][m] = entry
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
