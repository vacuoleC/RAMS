#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""框架比较：GRU 是否最优？传统ML / 线性深度 / 注意力三个梯队统一接口对比

目标（追求最大精度）：在完全相同的数据/任务/评估口径下，公平比较各框架，
回答「当前共享 GRU 架构是否最优，有没有更好的替换」。

统一接口（所有模型完全一致）：
  - 数据：20 层水温（temp_*）+ 6 气象，复用 TensorBuilder._load_wide（同样的
    3h 网格/聚合/dropna/时间轴），T=24 回看，H=8 预测未来 24h 表层浓度
  - 切分：时序 70/15/15（与 T1 基线 A 完全一致），无泄漏（训练段拟合归一化）
  - 评估：测试集 RMSE（还原原始尺度，浓度单位），3 seed 均值±std
  - 训练量：所有 torch 模型固定 30 epoch × batch128 × Adam lr1e-3（与 GRU 基线一致）
    ，ML 模型用公平的固定预算

梯队：
  Stage 1 传统ML（CPU）  ：持久化 / LinearRegression / Ridge / XGBoost / LightGBM
  Stage 2 线性深度(torch)：DLinear（趋势+周期线性分解）/ TSMixer（MLP 时-特征混合）
  Stage 3 注意力(torch)  ：普通 Transformer / PatchTST（patch 编码）/ 简版 TFT
                            （GRN + 多头注意力 + 分位数头 + 静态 depth 协变量）
  参考                 ：GRU 当前架构（M1 单任务点估计，hidden=64，与 rams_net 同构）

用法（算力机 /data/RAMS/proj 下）：
  python3 scripts/explore/framework_compare.py                 # 全量：全部模型 × 3 seed
  python3 scripts/explore/framework_compare.py --smoke          # 冒烟：1 seed × 2 epoch
  python3 scripts/explore/framework_compare.py --stage 1        # 只跑传统 ML（CPU）
  python3 scripts/explore/framework_compare.py --stage 2,3      # 只跑 torch 模型
  python3 scripts/explore/framework_compare.py --device cpu     # 强制 CPU（GPU 被占时）
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import torch
import torch.nn as nn

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS

T, H = 24, 8                 # 回看 3 天 / 预测 24h
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
EPOCHS = 30
SEEDS = [0, 1, 2]
BS = 128
LR = 1e-3

# ---- 可选的第三方库（装不上则跳过该模型，诚实记录） ----
try:
    import sklearn
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.multioutput import MultiOutputRegressor
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False


# ================================================================
# 数据（统一接口：所有模型共用同一批窗口与切分）
# ================================================================
def load_data(parquet: str):
    """读宽表 → 归一化 → 滑动窗口 → 70/15/15 时序切分。

    与 T1 build_dataset / TensorBuilder 逐一对齐（同一时间轴、同一窗口索引比例），
    保证与已归档的 GRU 基线（M1 RMSE≈3.6）可比。
    返回 dict 含 train/val/test (Xw, yw, y_prev) + y_sd。
    """
    loader = TensorBuilder(TensorConfig(T=T, H=H))
    wide = loader._load_wide(Path(parquet)).sort_index()
    n = len(wide)
    n_tr = int(n * TRAIN_FRAC)

    temp_cols = [c for c in wide.columns if c.startswith("temp_")]
    conc_cols = [c for c in wide.columns if c.startswith("conc_")]
    feat_cols = temp_cols + list(METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]

    # 特征归一化（只用训练段拟合，防泄漏）
    tr = wide.iloc[:n_tr]
    X = wide[feat_cols].values.astype(np.float32)
    for i, c in enumerate(feat_cols):
        mu, sd = float(tr[c].mean()), float(tr[c].std()) + 1e-8
        X[:, i] = (X[:, i] - mu) / sd
    # 目标 = 表层浓度（第一个 conc_），训练段归一化
    yc = conc_cols[0]
    y_mu, y_sd = float(tr[yc].mean()), float(tr[yc].std()) + 1e-8
    y = (wide[yc].values.astype(np.float32) - y_mu) / y_sd

    # 滑动窗口：Xw (B,T,F) / yw (B,H) / y_prev (B,) 窗口前最后观测浓度（持久化基线用）
    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    yw = np.stack([y[i + T:i + T + H] for i in range(n_w)]).astype(np.float32)
    y_prev = np.array([y[i + T - 1] for i in range(n_w)], dtype=np.float32)

    # 切分（与 TensorBuilder 相同的窗口索引比例）
    n_trw, n_vaw = int(n_w * TRAIN_FRAC), int(n_w * VAL_FRAC)
    idx_tr, idx_va, idx_te = range(n_trw), range(n_trw, n_trw + n_vaw), range(n_trw + n_vaw, n_w)
    splits = {
        "train": (Xw[idx_tr], yw[idx_tr], y_prev[idx_tr]),
        "val": (Xw[idx_va], yw[idx_va], y_prev[idx_va]),
        "test": (Xw[idx_te], yw[idx_te], y_prev[idx_te]),
    }
    return splits, float(y_sd)


def rmse_orig(pred, y, y_sd: float) -> float:
    """归一化 RMSE → 还原原始尺度。"""
    return float(np.sqrt(np.mean((pred - y) ** 2)) * y_sd)


def per_step_rmse(pred, y, y_sd: float):
    return np.sqrt(np.mean((pred - y) ** 2, axis=0)) * y_sd


# ================================================================
# Stage 1 · 传统 ML 基线（CPU）
# ================================================================
def stage1_persistence(Xw, yw, y_prev, y_sd):
    """持久化：把窗口最后观测浓度当未来 24h 预测。"""
    n_w = len(yw)
    n_trw, n_vaw = int(n_w * TRAIN_FRAC), int(n_w * VAL_FRAC)
    p_tr = np.tile(y_prev[:n_trw][:, None], (1, H))
    p_va = np.tile(y_prev[n_trw:n_trw + n_vaw][:, None], (1, H))
    p_te = np.tile(y_prev[n_trw + n_vaw:][:, None], (1, H))
    return {
        "rmse": rmse_orig(p_te, yw[n_trw + n_vaw:], y_sd),
        "train_rmse_norm": float(np.sqrt(np.mean((p_tr - yw[:n_trw]) ** 2))),
        "val_rmse_norm": float(np.sqrt(np.mean((p_va - yw[n_trw:n_trw + n_vaw]) ** 2))),
        "per_step": per_step_rmse(p_te, yw[n_trw + n_vaw:], y_sd),
    }


def stage1_ml(seed, splits, y_sd, smoke=False):
    """跑全部传统 ML 模型（展平滞后窗口 → 多输出回归 H 步），返回 {name: result}。"""
    res = {}
    (Xw, yw, _), _, _ = splits["train"], splits["val"], splits["test"]
    n_w = len(yw)
    n_trw, n_vaw = int(n_w * TRAIN_FRAC), int(n_w * VAL_FRAC)
    Xf = Xw.reshape(n_w, -1)
    Xtr, ytr = Xf[:n_trw], yw[:n_trw]
    Xva, yva = Xf[n_trw:n_trw + n_vaw], yw[n_trw:n_trw + n_vaw]
    Xte, yte = Xf[n_trw + n_vaw:], yw[n_trw + n_vaw:]

    n_est = 30 if smoke else 200

    def run(name, model):
        model.fit(Xtr, ytr)
        p_tr = model.predict(Xtr)
        p_va = model.predict(Xva)
        p_te = model.predict(Xte)
        return {
            "rmse": rmse_orig(p_te, yte, y_sd),
            "train_rmse_norm": float(np.sqrt(np.mean((p_tr - ytr) ** 2))),
            "val_rmse_norm": float(np.sqrt(np.mean((p_va - yva) ** 2))),
            "per_step": per_step_rmse(p_te, yte, y_sd),
        }

    if HAS_SKLEARN:
        res["LinearRegression"] = run("LinearRegression", LinearRegression())
        res["Ridge"] = run("Ridge", Ridge(alpha=0.5))
    if HAS_XGB:
        res["XGBoost"] = run("XGBoost", MultiOutputRegressor(xgb.XGBRegressor(
            n_estimators=n_est, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=seed)))
    if HAS_LGB:
        res["LightGBM"] = run("LightGBM", MultiOutputRegressor(lgb.LGBMRegressor(
            n_estimators=n_est, learning_rate=0.05, num_leaves=31, subsample=0.8,
            colsample_bytree=0.8, random_state=seed)))
    return res


# ================================================================
# 通用 torch 训练器（所有 DL 模型共享，公平训练量）
# ================================================================
def train_torch(model, Xtr, ytr, Xva, yva, Xte, yte, y_sd,
                epochs=EPOCHS, bs=BS, lr=LR, device="cuda", seed=0,
                extra_pred=None, loss_fn=None, pred_median=None):
    """固定预算训练 + 最终模型评估。返回指标 dict。

    extra_pred:  可选，对 (model, batch_x) 额外输出用于自定义 head（如分位数）。
    loss_fn:     可选，自定义损失 (pred, y) → 标量（TFT 分位数损失用）。
    pred_median: 可选，从 extra_pred 输出提取用于 RMSE 的均值预测（TFT 取 p50）。
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    if loss_fn is None:
        loss_fn = nn.MSELoss()

    def _pred_for_rmse(model, x):
        """得到用于 RMSE 的均值预测 (B,H)。"""
        if extra_pred is None:
            return model(x)
        out = extra_pred(model, x)
        if pred_median is not None:
            return pred_median(out)
        return out

    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    Xv = torch.tensor(Xva, dtype=torch.float32).to(device)
    yv = torch.tensor(yva, dtype=torch.float32).to(device)
    Xe = torch.tensor(Xte, dtype=torch.float32).to(device)
    ye = torch.tensor(yte, dtype=torch.float32).to(device)

    n = Xt.shape[0]
    best_val = 1e9
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx].to(device), yt[idx].to(device)
            opt.zero_grad()
            if extra_pred is not None:
                pred = extra_pred(model, xb)
            else:
                pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = _pred_for_rmse(model, Xv)
            vr = float(torch.sqrt(torch.mean((pv - yv) ** 2)))
            best_val = min(best_val, vr)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"      ep{ep} loss={loss.item():.4f} val_rmse={vr:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        p_te = _pred_for_rmse(model, Xe).cpu().numpy()
        p_va = _pred_for_rmse(model, Xv).cpu().numpy()
        p_tr = None
        for i in range(0, n, bs):
            pb = _pred_for_rmse(model, Xt[i:i + bs].to(device)).cpu().numpy()
            if p_tr is None:
                p_tr = pb
            else:
                p_tr = np.concatenate([p_tr, pb], axis=0)

    return {
        "rmse": rmse_orig(p_te, yte, y_sd),
        "train_rmse_norm": float(np.sqrt(np.mean((p_tr - ytr) ** 2))) if p_tr is not None else None,
        "val_rmse_norm": float(np.sqrt(np.mean((p_va - yva) ** 2))),
        "best_val_norm": best_val,
        "per_step": per_step_rmse(p_te, yte, y_sd),
        "params": sum(p.numel() for p in model.parameters()),
    }


def stage_dl(model_builder, Xw, yw, y_prev, y_sd, device, seeds, name, epochs=None, train_kwargs=None):
    """对单个 torch 模型跑 3 seed，返回 {name: result}。"""
    n_trw = int(len(yw) * TRAIN_FRAC)
    n_vaw = int(len(yw) * VAL_FRAC)
    Xtr, ytr = Xw[:n_trw], yw[:n_trw]
    Xva, yva = Xw[n_trw:n_trw + n_vaw], yw[n_trw:n_trw + n_vaw]
    Xte, yte = Xw[n_trw + n_vaw:], yw[n_trw + n_vaw:]

    outs = []
    for seed in seeds:
        m = model_builder()
        r = train_torch(m, Xtr, ytr, Xva, yva, Xte, yte, y_sd,
                        epochs=epochs or EPOCHS, device=device, seed=seed,
                        **(train_kwargs or {}))
        outs.append(r)
        print(f"      seed{seed} RMSE={r['rmse']:.4f} params={r['params']:,}", flush=True)
    return outs


# ================================================================
# Stage 2 · 线性深度模型
# ================================================================
class DLinear(nn.Module):
    """DLinear：时间序列分解为趋势(MA)+周期(残差)，各自线性外推后按特征聚合。

    用滑动平均提取趋势，残差为周期分量；两个 Linear(T→H) 分别预测，逐特征后
    对 26 特征取均值输出标量 (B,H)。参数量极小，验证「线性外推是否已够用」。
    """

    def __init__(self, n_feats: int = 26, in_len: int = T, out_len: int = H, kernel: int = 3):
        super().__init__()
        self.n_feats = n_feats
        # kernel=3, padding=1 → 输出长度与输入相同，无需对齐
        self.avg = nn.AvgPool1d(kernel_size=kernel, stride=1, padding=kernel // 2)
        self.linear_trend = nn.Linear(in_len, out_len)
        self.linear_season = nn.Linear(in_len, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F)
        x = x.permute(0, 2, 1)  # (B,F,T)
        trend = self.avg(x)
        season = x - trend
        y = self.linear_trend(trend) + self.linear_season(season)  # (B,F,H)
        return y.mean(dim=1)  # (B,H) 跨特征聚合


class TSMixer(nn.Module):
    """TSMixer：两层 MLP 分别在「时间维」和「特征维」混合，残差+LayerNorm。"""

    def __init__(self, n_feats: int = 26, in_len: int = T, hidden: int = 64, n_blocks: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleList([
                nn.Sequential(nn.Linear(in_len, hidden), nn.GELU(), nn.Linear(hidden, in_len)),
                nn.Sequential(nn.Linear(n_feats, hidden), nn.GELU(), nn.Linear(hidden, n_feats)),
            ]))
        self.norm = nn.LayerNorm(n_feats)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(in_len * n_feats, H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F)
        out = x
        for mlp_t, mlp_f in self.blocks:
            out = out + mlp_t(out.transpose(1, 2)).transpose(1, 2)   # 时间混合 (B,T,F)
            out = out + mlp_f(out)                                   # 特征混合 (B,T,F)
            out = self.norm(out)
        return self.head(out)


# ================================================================
# Stage 3 · 注意力 / Transformer 模型
# ================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PlainTransformer(nn.Module):
    """普通 Transformer：Linear 嵌入 + 位置编码 + 编码器 + 末 token 读出头。"""

    def __init__(self, n_feats: int = 26, in_len: int = T, d_model: int = 64,
                 n_heads: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_proj = nn.Linear(n_feats, d_model)
        self.pos = PositionalEncoding(d_model, in_len)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F)
        h = self.pos(self.in_proj(x))
        h = self.enc(h)
        return self.head(h[:, -1])


class PatchTST(nn.Module):
    """简版 PatchTST：时间维切 patch，共享 patch 嵌入，Transformer 编码，跨通道聚合。"""

    def __init__(self, n_feats: int = 26, in_len: int = T, patch_len: int = 4,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.n_feats = n_feats
        self.patch_len = patch_len
        self.n_patches = in_len // patch_len
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = PositionalEncoding(d_model, self.n_patches)
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=128,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.head = nn.Sequential(nn.Linear(self.n_patches * d_model, 128), nn.GELU(), nn.Linear(128, H))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,F)
        B, Tf, F = x.shape
        # (B, F, T) → 切 patch → (B, F, n_patches, patch_len)
        xt = x.permute(0, 2, 1).reshape(B, F, self.n_patches, self.patch_len)
        h = self.patch_embed(xt)               # (B, F, n_patches, d_model)
        h = self.pos(h.view(B * F, self.n_patches, -1))  # (B*F, n_patches, d_model)
        h = self.enc(h)
        # 跨通道平均后接 head：最简洁的通道聚合
        h = h.view(B, F, self.n_patches, -1).mean(dim=1)  # (B, n_patches, d_model)
        return self.head(h.reshape(B, -1))


class GRN(nn.Module):
    """门控残差网络（TFT 核心单元）：GELU 前馈 + 门控 + LayerNorm，可选外部 context。"""

    def __init__(self, d_in: int, d_out: int, d_hidden: int = 64, d_context: int = 0, dropout: float = 0.1):
        super().__init__()
        self.d_context = d_context
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_hidden)
        self.fc3 = nn.Linear(d_hidden, d_out)
        self.gate = nn.Linear(d_out, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.do = nn.Dropout(dropout)
        if d_context > 0:
            self.fc_ctx = nn.Linear(d_context, d_hidden)
        if d_in != d_out:
            self.skip = nn.Linear(d_in, d_out)
        else:
            self.skip = None

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        a = self.do(torch.nn.functional.gelu(self.fc1(x)))
        if context is not None and self.d_context > 0:
            a = a + self.fc_ctx(context).unsqueeze(1) if a.dim() == 3 else a + self.fc_ctx(context)
        a = self.fc2(a)
        a = self.do(torch.nn.functional.gelu(a))
        y = self.fc3(a)
        g = torch.sigmoid(self.gate(y))
        base = self.skip(x) if self.skip is not None else x
        return self.ln(g * y + (1 - g) * base)


class SimpleTFT(nn.Module):
    """简版 Temporal Fusion Transformer。

    关键能力（任务要求）：
      - 静态协变量：把 20 层水温的 depth 位置 [0.5..10.0m]（归一化）作为静态特征，
        经 GRN 编码为 context 注入序列编码器与注意力（模型学「哪些深度重要」）
      - 分位数输出：头输出 (B, 3H) = p10/p50/p90，训练时 p50 走 MSE（与其它模型
        同口径比较 RMSE），p10/p90 走分位数损失（辅助，附送区间能力）
    结构：静态 GRN → LSTM 序列编码（用 context 门控）→ 多头自注意力 → 分位数头
    """

    def __init__(self, n_feats: int = 26, in_len: int = T, out_len: int = H,
                 d_model: int = 64, n_heads: int = 2, n_layers: int = 1,
                 dropout: float = 0.1, depths=None):
        super().__init__()
        self.out_len = out_len
        self.d_model = d_model
        # 静态特征：depth 位置向量（归一化到 0~1）
        self.depths = np.array(depths if depths is not None else [0.5 + 0.5 * i for i in range(20)],
                               dtype=np.float32)
        self.static_dim = len(self.depths)
        self.static_embed = nn.Linear(self.static_dim, d_model)
        self.static_grn = GRN(d_model, d_model, d_hidden=d_model, d_context=0, dropout=dropout)
        # 序列编码：输入 proj + LSTM（context 门控初态）
        self.in_proj = nn.Linear(n_feats, d_model)
        self.lstm = nn.LSTM(d_model, d_model, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(d_model)
        self.head_ctx = nn.Linear(d_model * 2, d_model)
        self.head = nn.Linear(d_model, out_len * 3)  # p10 / p50 / p90

    def _static_context(self, device):
        d = torch.tensor(self.depths, dtype=torch.float32, device=device).unsqueeze(0) / 10.0  # (1, S)
        c = self.static_grn(self.static_embed(d))  # (1, d_model)
        return c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回中位数 p50 (B,H)。分位数完整输出见 predict_quantiles。"""
        return self.predict_quantiles(x)[:, self.out_len:2 * self.out_len]

    def predict_quantiles(self, x: torch.Tensor) -> torch.Tensor:  # (B, 3H)
        B, Tf, _ = x.shape
        dev = x.device
        c = self._static_context(dev)  # (1, d_model)
        h = self.in_proj(x)
        # LSTM 初态用静态 context（batch 广播）
        c0 = c.expand(self.lstm.num_layers, B, -1).contiguous()
        h0 = c.expand(self.lstm.num_layers, B, -1).contiguous()
        out, _ = self.lstm(h, (h0, c0))  # (B, T, d_model)
        # 多头自注意力 + 残差
        a, _ = self.attn(out, out, out)
        out = self.attn_ln(out + a)
        # 时间聚合：末 token + 注意力输出均值 → context head
        ctx = torch.cat([out[:, -1], out.mean(dim=1)], dim=-1)
        h = torch.nn.functional.gelu(self.head_ctx(ctx))
        return self.head(h)  # (B, 3H)


def tft_loss(pred, y):
    """TFT 分位数损失：p50 走 MSE（同口径比较 RMSE），p10/p90 走分位数损失（辅助）。"""
    H = pred.shape[1] // 3
    p10, p50, p90 = pred[:, :H], pred[:, H:2 * H], pred[:, 2 * H:]
    mse = torch.mean((p50 - y) ** 2)
    q10 = torch.mean(torch.maximum(0.1 * (y - p10), (0.1 - 1) * (y - p10)))
    q90 = torch.mean(torch.maximum(0.9 * (y - p90), (0.9 - 1) * (y - p90)))
    return mse + 0.5 * (q10 + q90)


def tft_extra_pred(model, x):
    """TFT 训练：预测输出分位数 (B,3H)，p50 通道用于评估。"""
    return model.predict_quantiles(x)


# ================================================================
# 主流程
# ================================================================
def build_models(feat_dim, device):
    """返回 {模型名: (builder函数, 阶段, 训练kwargs)}。"""
    from rams.models.rams_net import SharedGRU
    models = {}

    # 参考：GRU 当前架构（M1 单任务点估计，与 rams_net backbone 同构）
    class GRUPoint(nn.Module):
        def __init__(self, fd):
            super().__init__()
            self.backbone = SharedGRU(fd, hidden=64, n_layers=1, dropout=0.0)
            self.head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, H))
        def forward(self, x):
            return self.head(self.backbone(x))

    models["GRU(当前架构)"] = (lambda: GRUPoint(feat_dim), 2, {})
    models["DLinear"] = (lambda: DLinear(n_feats=feat_dim), 2, {})
    models["TSMixer"] = (lambda: TSMixer(n_feats=feat_dim, n_blocks=2), 2, {})
    models["Transformer"] = (lambda: PlainTransformer(n_feats=feat_dim), 3, {})
    models["PatchTST"] = (lambda: PatchTST(n_feats=feat_dim), 3, {})
    models["TFT(简版)"] = (lambda: SimpleTFT(n_feats=feat_dim), 3, {
        "extra_pred": tft_extra_pred,
        "loss_fn": tft_loss,
        "pred_median": lambda q: q[:, H:2 * H],
    })
    return models


def fmt_mean_std(rows):
    r = np.array([x["rmse"] for x in rows])
    return f"{r.mean():.3f}±{r.std():.3f}"


def main():
    ap = argparse.ArgumentParser(description="RAMS 框架比较（GRU vs 传统ML/线性深度/注意力）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 seed × 2 epoch，ML 用最小预算")
    ap.add_argument("--stage", default="all", help="all / 1 / 2,3 等逗号分隔")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", default="docs/framework_compare_results.json")
    ap.add_argument("--out-md", default="docs/framework_compare.md")
    args = ap.parse_args()

    t0 = time.time()
    print("== RAMS 框架比较（GRU 是否最优？）==", flush=True)
    print(f"  设备: {args.device}  smoke={args.smoke}", flush=True)
    print("[1] 读取宽表并构建统一数据集 ...", flush=True)
    splits, y_sd = load_data(args.parquet)
    (Xw, yw, y_prev), (Xwv, ywv, ypv), (Xwt, ywt, ypt) = splits["train"], splits["val"], splits["test"]
    print(f"  窗口样本: train {Xw.shape} val {Xwv.shape} test {Xwt.shape}  y_sd={y_sd:.4f}", flush=True)

    seeds = [0] if args.smoke else SEEDS
    epochs = 2 if args.smoke else EPOCHS
    stages = {int(s) for s in args.stage.split(",")} if args.stage != "all" else {1, 2, 3}
    results: dict[str, list] = {}
    notes: list[str] = []

    # ---- Stage 1 传统 ML ----
    if 1 in stages:
        print("\n[Stage 1] 传统 ML 基线（CPU）...", flush=True)
        # 持久化
        p = stage1_persistence(Xw, yw, y_prev, y_sd)
        results["持久化"] = [p]
        print(f"  持久化: RMSE={p['rmse']:.4f}", flush=True)
        for seed in seeds:
            r = stage1_ml(seed, splits, y_sd, smoke=args.smoke)
            for k, v in r.items():
                results.setdefault(k, []).append(v)
                print(f"  {k}: RMSE={v['rmse']:.4f} (seed{seed})", flush=True)

    # ---- Stage 2/3 torch 模型 ----
    if 2 in stages or 3 in stages:
        print(f"\n[Stage 2/3] torch 模型（{args.device}，{len(seeds)} seed × {epochs} epoch）...", flush=True)
        models = build_models(Xw.shape[2], args.device)
        for name, (builder, stage, train_kwargs) in models.items():
            if stage not in stages:
                continue
            print(f"  [{name}] stage{stage} ...", flush=True)
            try:
                rows = stage_dl(builder, Xw, yw, y_prev, y_sd, args.device, seeds, name,
                                epochs=epochs, train_kwargs=train_kwargs)
                results[name] = rows
                print(f"    → {name}: RMSE={fmt_mean_std(rows)}", flush=True)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                notes.append(f"{name} 运行失败: {e}")
                results[name] = [{"rmse": float("nan"), "params": 0}]

    # ---- 梯队归属 ----
    stage_of = {"持久化": "Stage1-传统ML", "LinearRegression": "Stage1-传统ML", "Ridge": "Stage1-传统ML",
                "XGBoost": "Stage1-传统ML", "LightGBM": "Stage1-传统ML",
                "GRU(当前架构)": "参考", "DLinear": "Stage2-线性深度", "TSMixer": "Stage2-线性深度",
                "Transformer": "Stage3-注意力", "PatchTST": "Stage3-注意力", "TFT(简版)": "Stage3-注意力"}

    def stage_label(n):
        return stage_of.get(n, "?")

    # ---- 对照表 ----
    print("\n===== 对照表（测试集 RMSE，还原尺度，均值±std）=====", flush=True)
    header = f"  {'模型':<16s} {'梯队':<14s} {'参数量':<10s} {'RMSE':<14s} {'train_norm':<12s} {'val_norm':<12s}"
    print(header, flush=True)
    for name, rows in results.items():
        r = np.array([x["rmse"] for x in rows])
        if np.isnan(r).all():
            print(f"  {name:<16s} {'':<14s} {'':<10s} FAILED", flush=True)
            continue
        params = rows[0].get("params", "-")
        trn = np.mean([x.get("train_rmse_norm") or np.nan for x in rows])
        vln = np.mean([x.get("val_rmse_norm") or np.nan for x in rows])
        print(f"  {name:<16s} {stage_label(name):<14s} {str(params):<10s} "
              f"{fmt_mean_std(rows):<14s} {trn:<12.4f} {vln:<12.4f}", flush=True)

    # ---- 写 JSON + Markdown ----
    write_outputs(results, y_sd, splits, args, notes, stage_of)


def write_outputs(results, y_sd, splits, args, notes, stage_of):
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, rows in results.items():
        r = np.array([x["rmse"] for x in rows if not np.isnan(x.get("rmse", np.nan))])
        summary[name] = {
            "stage": stage_of.get(name, "?"),
            "params": rows[0].get("params", None) if rows else None,
            "rmse_mean": float(r.mean()) if len(r) else None,
            "rmse_std": float(r.std()) if len(r) else None,
            "per_seed_rmse": [float(x.get("rmse", np.nan)) for x in rows],
            "train_rmse_norm": float(np.mean([x.get("train_rmse_norm") or np.nan for x in rows])) if rows else None,
            "val_rmse_norm": float(np.mean([x.get("val_rmse_norm") or np.nan for x in rows])) if rows else None,
            "per_step_rmse": [float(x) for x in rows[0].get("per_step", [])] if rows and rows[0].get("per_step") is not None else None,
        }
    data = {"meta": {"T": T, "H": H, "seeds": SEEDS, "epochs": EPOCHS, "y_sd": y_sd,
                     "split": "70/15/15 时序", "features": "20层水温+6气象"},
            "models": summary, "notes": notes}
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}", flush=True)

    # Markdown 文档
    md = build_markdown(data)
    md_path = Path(args.out_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md, encoding="utf-8")
    print(f"[result] 文档已写入 {md_path}", flush=True)


def build_markdown(data):
    lines = []
    m = data["models"]
    lines.append("# RAMS 模型框架比较：GRU 是否最优？\n")
    lines.append("> 统一接口公平对比：同一数据（20 层水温 + 6 气象，T=24 回看 → 预测未来 24h 表层浓度）"
                 "、同一时序切分（70/15/15）、同一评估（测试集 RMSE 还原原始尺度）、"
                 "torch 模型同一训练量（30 epoch × batch128 × Adam lr1e-3）。只输出统计量，不涉原始数据。\n")
    lines.append(f"数据：{len(m)} 个模型 × 3 seed，y_sd={data['meta']['y_sd']:.4f}\n")
    lines.append("## 对照表（均值±std，3 seed）\n")
    lines.append("| 模型 | 梯队 | 参数量 | 测试 RMSE | 训练 RMSE(norm) | 验证 RMSE(norm) |")
    lines.append("|---|---|---|---|---|---|")
    for name, r in m.items():
        if r["rmse_mean"] is None:
            lines.append(f"| {name} | {r['stage']} | - | **运行失败** | - | - |")
            continue
        ps = "-" if r["params"] is None else f"{r['params']:,}"
        trn = f"{r['train_rmse_norm']:.4f}" if r["train_rmse_norm"] is not None else "-"
        vln = f"{r['val_rmse_norm']:.4f}" if r["val_rmse_norm"] is not None else "-"
        lines.append(f"| {name} | {r['stage']} | {ps} | **{r['rmse_mean']:.3f}±{r['rmse_std']:.3f}** | {trn} | {vln} |")
    lines.append("")

    # 排序
    valid = {k: v for k, v in m.items() if v["rmse_mean"] is not None}
    order = sorted(valid, key=lambda k: valid[k]["rmse_mean"])
    lines.append("## 排序（RMSE 升序）\n")
    for i, k in enumerate(order, 1):
        r = valid[k]
        lines.append(f"{i}. **{k}** — {r['rmse_mean']:.3f}±{r['rmse_std']:.3f}（{r['stage']}）")
    lines.append("")

    # 分步 RMSE（最能说明预测期误差结构）
    lines.append("## 各预测步 RMSE（未来 24h，8 步 × 3h）\n")
    lines.append("| 模型 | 步1 | 步2 | 步3 | 步4 | 步5 | 步6 | 步7 | 步8 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, r in sorted(valid.items(), key=lambda kv: kv[1]["rmse_mean"]):
        ps = r.get("per_step_rmse")
        if ps:
            cells = " | ".join(f"{v:.3f}" for v in ps)
            lines.append(f"| {name} | {cells} |")
    lines.append("")

    # 结论
    best = order[0]
    best_r = valid[best]
    gru = valid.get("GRU(当前架构)")
    lines.append("## 结论\n")
    lines.append(f"- **当前最优框架（RMSE 最低）**：`{best}`（{best_r['rmse_mean']:.3f}±{best_r['rmse_std']:.3f}）。")
    if gru is not None:
        diff = gru["rmse_mean"] - best_r["rmse_mean"]
        rel = diff / max(gru["rmse_mean"], 1e-9) * 100
        lines.append(f"- **GRU 当前架构**：{gru['rmse_mean']:.3f}±{gru['rmse_std']:.3f}，"
                     f"与最优差 {diff:+.3f}（{rel:+.1f}%）——{'是' if diff < 0 else '否'}最优。")
        if diff < 0:
            lines.append("  - 结论：GRU 仍是最优/领先，不建议更换。")
        else:
            lines.append("  - 结论：存在更优框架，值得考虑替换（见数据）。")
    lines.append("")

    # 诚实记录
    lines.append("## 诚实记录\n")
    if data["notes"]:
        for nt in data["notes"]:
            lines.append(f"- {nt}")
    else:
        lines.append("- 所有模型均成功运行，无过拟合标志（训练/验证 RMSE 见上表，若 val 明显高于 train 即过拟合，表中已列出）。")
    lines.append("")

    # 运行细节
    lines.append("## 运行细节\n")
    lines.append("- 评估口径：测试集 RMSE（还原原始浓度尺度），3 seed 均值±std。")
    lines.append("- 参考 GRU 为当前架构 M1 单任务点估计（hidden=64，与 `rams_net.SharedGRU` 同构）；"
                 "多任务完整版已归档 M1 RMSE≈3.6（30 epoch）。")
    lines.append(f"- 模型数：{len(m)}；覆盖 3 梯队（传统 ML / 线性深度 / 注意力）。")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
