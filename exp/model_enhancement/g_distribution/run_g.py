# -*- coding: utf-8 -*-
"""G 探索：完整分布输出（IQN 隐分位数网络）vs 3 分位数基线

核心假设：M1 目前只输出 3 个分位数（p10/p50/p90），CRPS 用分段线性 CDF 近似。
换成**完整预测分布**（IQN：对任意 τ∈(0,1) 输出该分位数值，预测时采样 N 个分位数重建 CDF，
CRPS 用能量形式精确估计）可能让 CRPS 更优（更精确的分布估计）。

对照（增量 abs_delta 协议，同一滚动窗口，17 窗口）：
  1. base3 : 冻结 RamsNet p10/p50/p90 分位数头（B1/B2/B7 基线，CRPS 用 B7 分段线性闭合形式）
  2. q9    : 简版候选——9 个固定分位数（pinball 损失，分段线性 CRPS 9 结），隔离"分位数点数"效应
  3. iqn    : 候选——IQN 隐分位数网络（余弦 τ 嵌入 + 共享 GRU 骨干，训练采 K=32 个 τ，
              预测采样 N=64 个分位数重建 CDF，CRPS 能量形式精确估计 + PIT 校准）

模型：全部复用 rams/ 冻结的 SharedGRU 骨干 + M2/M4 头（多任务损失权重与 Trainer 一致
w1=1 / w2=3 / w4=2）；仅 M1 头与损失不同（新头在 exp/ 内实现，不触碰冻结代码）。

评估（全部还原 conc 单位，逐视界 CRPS + RMSE + 覆盖率）：
  - CRPS：base3/q9 用分段线性 CDF 闭合形式；iqn 用 N=64 分位数能量形式 CRPS（对经验 CDF 精确）
  - 覆盖率：真实 conc 落在 [p10,p90]（iqn 直接求 τ=0.1/0.9 的值）→ 3 变体同口径
  - RMSE：p50（iqn 直接求 τ=0.5）
  - PIT 校准（iqn 专属）：真实值在 64 个分位数中的秩 → 应均匀 U(0,1)，5 分箱直方图
  - 持久化对照：abs_delta 零变化 = conc_t（同 B7），CRPS 相对技能 >0 模型更优

保密：只输出聚合统计量，不打印原始数据行；数据涉密只输出统计量。

算力机：sensecore H100（torch 2.3.1+cu121）。
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
B7_DIR = HERE.parents[0] / "b7_dimensionless"  # 复用 run_b7 的窗口协议/目标/还原/3分位基线
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(B7_DIR))
sys.path.insert(0, str(HERE.parents[2]))  # 项目根（g_distribution 比 b7 深一层）

# 先 import run_b7：其模块级会把 stdout 重包为 utf-8（与 B7 独立运行一致）。
# run_g 若在此之前也包一次，run_b7 import 会再包一次 → 旧 wrapper 被 GC 关闭共享 buffer，
# 导致 "I/O operation on closed file"（H100 py3.10 实测）。
import run_b7 as B7  # noqa: E402 （复用窗口协议/目标构造/还原/3分位基线评估）

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# 确保 utf-8 输出（run_b7 已包；未包时 reconfigure 兜底，避免再次重包）
if not getattr(sys.stdout, "encoding", "").lower().startswith("utf"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError, ValueError):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from rams.models.rams_net import SharedGRU, M2Head, M4Head  # noqa: E402

T, H = B7.T, B7.H
EPOCHS = 30
SEEDS = [0, 1, 2]

# 候选超参
Q9_LEVELS = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95])
N_IQN = 64          # 预测时采样分位数个数
K_TAU = 32          # 训练时每样本采样 τ 个数
N_COS = 64          # IQN τ 余弦嵌入维
EMB_DIM = 64


# ---------------------------------------------------------------- 自定义头（exp 内，不碰冻结代码）
class IQNHead(nn.Module):
    """隐分位数头：输入共享隐状态 h + 一组分位水平 τ，输出该 τ 下的预测值 (B, K, H)。

    标准 IQN（Dabney et al.）：τ 用 cos(π·i·τ) 嵌入，与 h 拼接过 MLP。
    """

    def __init__(self, hidden: int, n_out: int, n_cos: int = N_COS, emb_dim: int = EMB_DIM):
        super().__init__()
        self.n_cos = n_cos
        self.emb = nn.Linear(n_cos, emb_dim)
        self.fc1 = nn.Linear(hidden + emb_dim, emb_dim)
        self.fc2 = nn.Linear(emb_dim, n_out)
        self.register_buffer("cos_index", torch.arange(1, n_cos + 1).float())

    def forward(self, h: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # h: (B, hidden)   tau: (B, K) ∈ (0,1)
        B, K = tau.shape
        psi = torch.cos(math.pi * tau[..., None] * self.cos_index)  # (B,K,n_cos)
        emb = torch.relu(self.emb(psi))                             # (B,K,emb_dim)
        hh = h[:, None, :].expand(B, K, h.size(-1))                 # (B,K,hidden)
        feat = torch.relu(self.fc1(torch.cat([hh, emb], dim=-1)))   # (B,K,emb_dim)
        return self.fc2(feat)                                       # (B,K,n_out)


class IQNNet(nn.Module):
    """共享 GRU 骨干 + IQN 头 + M2/M4 头（多任务，权重与 Trainer 一致）。"""

    def __init__(self, feat_dim: int, horizon: int, hidden: int = 64,
                 n_cos: int = N_COS, emb_dim: int = EMB_DIM, n_classes: int = 2,
                 n_levels: int = 4):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden)
        self.iqn = IQNHead(hidden, horizon, n_cos, emb_dim)
        self.m2 = M2Head(hidden, n_classes)
        self.m4 = M4Head(hidden, n_levels)
        self.horizon = horizon

    def forward(self, x, tau):
        h = self.backbone(x)
        q = self.iqn(h, tau)
        return q, self.m2(h), self.m4(h)


class QNet(nn.Module):
    """简版候选：共享 GRU 骨干 + n_q 固定分位数头 + M2/M4。"""

    def __init__(self, feat_dim: int, horizon: int, n_q: int, hidden: int = 64,
                 n_classes: int = 2, n_levels: int = 4):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, n_q * horizon))
        self.m2 = M2Head(hidden, n_classes)
        self.m4 = M4Head(hidden, n_levels)
        self.horizon = horizon
        self.n_q = n_q

    def forward(self, x):
        h = self.backbone(x)
        q = self.head(h).reshape(-1, self.n_q, self.horizon)
        return q, self.m2(h), self.m4(h)


# ---------------------------------------------------------------- CRPS 工具
def crps_cdf_pline(q, p_levels, y):
    """分段线性 CDF 的 CRPS 闭合形式（B7 crps_quantiles 的推广：任意分位数结）。

    尾部：最低/最高结之外用最外侧段斜率线性外推到 p=0 / p=1（与 B7 一致）。
    q/p/y 均为 array；q: (..., n_q) 升序，p_levels: (n_q,) 升序，y: (...)。
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p_levels, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if q.shape[-1] != p.shape[0]:
        raise ValueError("q 最后一维须与 p_levels 等长")
    q = np.sort(q, axis=-1)
    p = np.sort(p)
    sL = (q[..., 1] - q[..., 0]) / (p[1] - p[0])
    sR = (q[..., -1] - q[..., -2]) / (p[-1] - p[-2])
    qk = np.concatenate([
        (q[..., 0] - sL * p[0])[..., None], q,
        (q[..., -1] + sR * (1.0 - p[-1]))[..., None]], axis=-1)
    ak = np.concatenate([[0.0], p, [1.0]])
    deg = (qk[..., -1] - qk[..., 0]) < 1e-9
    total = np.zeros_like(y, dtype=np.float64)
    for k in range(len(ak) - 1):
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
    out = np.where(deg, np.abs(y - np.median(q, axis=-1)), total)
    return np.maximum(out, 0.0)


def crps_energy(q, y):
    """能量形式 CRPS：对经验分布（N 个等权重分位数点）精确。

    CRPS(F, y) = E|X−y| − ½·E|X−X'|。q: (..., N), y: (...)。
    """
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m1 = np.mean(np.abs(q - y[..., None]), axis=-1)
    d = np.abs(q[..., :, None] - q[..., None, :])
    m2 = np.mean(d, axis=(-1, -2))
    return m1 - 0.5 * m2


def pit_from_q(q, y):
    """PIT：真实值在 N 个分位数中的秩（平分并列），应 ~ U(0,1)。q: (..., N)。"""
    q = np.asarray(q, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    below = np.mean(q < y[..., None], axis=-1)
    eq = np.mean(q == y[..., None], axis=-1)
    return below + 0.5 * eq


# ---------------------------------------------------------------- 训练
def _m4_weights(warn_tr, n_levels):
    counts = np.bincount(warn_tr, minlength=n_levels)
    inv = 1.0 / (counts.astype(np.float64) + 1.0)
    return torch.tensor(inv / inv.sum() * n_levels, dtype=torch.float32)


def _pinball(y, q, tau):
    """分位数损失：q (B,K,H), y (B,H), tau (B,K,1)。"""
    e = y[:, None, :] - q
    return torch.mean(torch.maximum(tau * e, (tau - 1.0) * e))


def train_model(model, Xw, yw, strat_w, warn_w, n_tr, epochs, device,
                variant, seed, n_q=None, k_tau=K_TAU, n_cos=N_COS,
                emb_dim=EMB_DIM, batch_size=128, lr=1e-3,
                w_m2=3.0, w_m4=2.0, fast_dev_run=False):
    """训练 base3/q9/iqn 三类模型（多任务，权重与冻结 Trainer 一致 w1=1/w2=3/w4=2）。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(device)
    ce2 = nn.CrossEntropyLoss()
    n_levels = model.m4.mlp[-1].out_features
    ce4 = nn.CrossEntropyLoss(weight=_m4_weights(warn_w, n_levels).to(device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.tensor(Xw[:n_tr])
    yt = torch.tensor(yw[:n_tr])
    st = torch.tensor(strat_w[:n_tr])
    wt = torch.tensor(warn_w[:n_tr])
    ds = torch.utils.data.TensorDataset(Xt, yt, st, wt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    Xv = torch.tensor(Xw[n_tr:]).to(device)
    yv = torch.tensor(yw[n_tr:]).to(device)
    sv = torch.tensor(strat_w[n_tr:]).to(device)

    levels = torch.tensor(Q9_LEVELS, dtype=torch.float32, device=device)
    epochs = 2 if fast_dev_run else epochs
    for ep in range(epochs):
        model.train()
        for bi, (xb, yb, sb, wb) in enumerate(dl):
            if fast_dev_run and bi >= 2:
                break
            xb = xb.to(device); yb = yb.to(device); sb = sb.to(device); wb = wb.to(device)
            opt.zero_grad()
            if variant == "iqn":
                tau = torch.rand(xb.size(0), k_tau, device=device)
                q, m2o, m4o = model(xb, tau)
                l1 = _pinball(yb, q, tau[..., None])
            else:  # base3(3 结) / q9(9 结)
                nq = 3 if variant == "base3" else n_q
                lv = levels[:nq] if variant == "q9" else torch.tensor(
                    [0.1, 0.5, 0.9], device=device)
                q, m2o, m4o = model(xb)
                e = yb[:, None, :] - q
                l1 = torch.mean(torch.mean(
                    torch.maximum(lv[None, :, None] * e, (lv[None, :, None] - 1.0) * e),
                    dim=1))
            l2 = ce2(m2o, sb)
            l4 = ce4(m4o, wb)
            total = l1 + w_m2 * l2 + w_m4 * l4
            total.backward()
            opt.step()
        if ep % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                if variant == "iqn":
                    tau50 = torch.full((Xv.size(0), 1), 0.5, device=device)
                    qv, m2v, m4v = model(Xv, tau50)
                    pred = qv[:, 0]
                else:
                    qv, m2v, m4v = model(Xv)
                    pred = qv[:, 1] if variant == "base3" else qv[:, qv.shape[1] // 2]
                val_rmse = torch.sqrt(torch.mean((pred - yv) ** 2)).item()
                val_acc = (m2v.argmax(1) == sv).float().mean().item()
            print(f"    ep{ep} loss={total.item():.4f} val_rmse={val_rmse:.4f} val_acc={val_acc:.4f}", flush=True)
    return model


# ---------------------------------------------------------------- 预测
def predict_qn(model, X_te, levels, device):
    """固定分位数水平求值（base3/q9 → (B,n_q,H)；iqn → (B,n_q,H)）。"""
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_te).to(device)
        if isinstance(model, IQNNet):
            tau = torch.tensor(levels, dtype=torch.float32, device=device)[None, :].expand(len(X_te), -1)
            q, _, _ = model(Xt, tau)
        else:
            q, _, _ = model(Xt)
            # q 的结序固定：base3 [0.1,0.5,0.9]，q9 [0.05,0.10,0.20,0.35,0.50,0.65,0.80,0.90,0.95]
    return q.cpu().numpy()


def predict_grid(model, X_te, N, device):
    """IQN：N 个分位数等权网格 (i+0.5)/N 重建 CDF。返回 (B,N,H)。"""
    levels = (np.arange(N) + 0.5) / N
    return predict_qn(model, X_te, levels, device)


def main():
    ap = argparse.ArgumentParser(description="G 完整分布输出（IQN vs 3 分位数）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--max-windows", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-iqn", type=int, default=N_IQN)
    ap.add_argument("--variants", default="base3,q9,iqn")
    ap.add_argument("--out-json", default=str(HERE / "results.json"))
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    N_IQN_EVAL = 16 if args.smoke else args.n_iqn

    t0 = time.time()
    print(f"== G 完整分布输出（IQN vs 3 分位数）==", flush=True)
    print(f"   变体: {variants}  seeds={seeds}  iqn_eval_N={N_IQN_EVAL}", flush=True)
    print(f"   协议: 训练 {B7.TRAIN_DAYS}d / 测试 {B7.TEST_DAYS}d / 步长 {B7.STRIDE_DAYS}d；"
          f"abs_delta 目标；目标 {H}h", flush=True)

    wide = B7.load_wide(args.parquet)
    n = len(wide)
    print(f"[1] 宽表 {n} 时刻 × {wide.shape[1]} 列", flush=True)

    feat_cols = [c for c in wide.columns if c.startswith("temp_")] + list(B7.METEO_COLS)
    feat_cols = [c for c in feat_cols if c in wide.columns]
    if "conc_0.5" not in feat_cols:
        feat_cols = feat_cols + ["conc_0.5"]

    y_all = wide["conc_0.5"].values.astype(np.float64)
    n_tr_global = int(n * B7.TRAIN_DAYS / (B7.TRAIN_DAYS + B7.TEST_DAYS + B7.STRIDE_DAYS))
    eps_den = float(np.quantile(y_all[:n_tr_global], B7.EPS_DEN_Q))
    print(f"   [数据] conc_0.5 min={y_all.min():.2f} 零值={(y_all == 0).sum()} 个 "
          f"(对数不需要，abs_delta 口径)", flush=True)

    days = B7.TRAIN_DAYS + B7.TEST_DAYS
    windows = []
    for i0 in range(0, n - days * B7.GRID_PER_DAY + 1, B7.STRIDE_DAYS * B7.GRID_PER_DAY):
        windows.append((i0, i0 + days * B7.GRID_PER_DAY))
    if args.smoke:
        windows = windows[:1]
    elif args.max_windows:
        windows = windows[:args.max_windows]
    print(f"[2] 滚动窗口 {len(windows)} 个", flush=True)

    Nw = len(windows)
    n_seed = len(seeds)
    agg = {v: {
        "crps": np.zeros((Nw, n_seed)), "crps_h": np.zeros((Nw, n_seed, H)),
        "cov": np.zeros((Nw, n_seed)), "rmse": np.zeros((Nw, n_seed)),
        "crps_p": np.zeros((Nw, n_seed)), "pit_mean": np.zeros((Nw, n_seed, H)),
        "pit_bins": np.zeros((Nw, n_seed, 5)),
    } for v in variants}
    per_window_cur = []

    for wi, (i0, i1) in enumerate(windows):
        st = wide.index[i0]
        en = wide.index[i1 - 1]
        print(f"\n  [3.{wi + 1}] 窗口 {wi + 1}/{Nw}  {st:%Y-%m-%d} → {en:%Y-%m-%d}", flush=True)

        Xw, cur_raw, y_abs, strat_w, warn_w = B7.build_window(wide, i0, i1, feat_cols)
        n_win = len(Xw)
        n_tr = int(n_win * B7.TRAIN_DAYS / (B7.TRAIN_DAYS + B7.TEST_DAYS))
        te_sl = slice(n_tr, n_win)
        Xte = Xw[te_sl]
        cur_te = cur_raw[te_sl]
        y_te = y_abs[te_sl]
        Nte = len(Xte)

        per_window_cur.append({
            "window": wi + 1, "start": str(st), "end": str(en), "n_test": Nte,
            "cur_med": float(np.median(cur_te)), "y_med": float(np.median(y_te))})

        raw, kind = B7.make_targets("abs_delta", cur_raw, y_abs, eps_den)
        scale = float(np.std(raw[:n_tr])) + 1e-8
        y_norm = (raw / scale).astype(np.float32)

        # 持久化（abs_delta 零变化 → conc_t）的逐视界 CRPS = E|y - conc_t|，3 变体同口径
        crps_p_h = np.array([np.mean(np.abs(y_te[:, h] - cur_te)) for h in range(H)])
        crps_p = float(crps_p_h.mean())

        for v in variants:
            for si, seed in enumerate(seeds):
                torch.manual_seed(seed); np.random.seed(seed)
                print(f"    —— {v} seed={seed} ——", flush=True)
                if v == "base3":
                    B7.SEED = seed  # run_b7.train_model 内部用模块级 SEED 重播种
                    model = B7.train_model(Xw, y_norm, strat_w, warn_w, n_tr,
                                           args.epochs, args.device)
                    q_norm = B7.predict_quantiles(model, Xte, args.device)  # (B,3,H)
                elif v == "q9":
                    model = train_model(
                        QNet(feat_dim=Xw.shape[2], horizon=H, n_q=len(Q9_LEVELS)),
                        Xw, y_norm, strat_w, warn_w, n_tr, args.epochs, args.device,
                        variant="q9", seed=seed, fast_dev_run=args.smoke)
                    q_norm = predict_qn(model, Xte, Q9_LEVELS, args.device)  # (B,9,H)
                elif v == "iqn":
                    model = train_model(
                        IQNNet(feat_dim=Xw.shape[2], horizon=H),
                        Xw, y_norm, strat_w, warn_w, n_tr, args.epochs, args.device,
                        variant="iqn", seed=seed, fast_dev_run=args.smoke)
                    q3 = predict_qn(model, Xte, np.array([0.1, 0.5, 0.9]), args.device)  # (B,3,H)
                    qN = predict_grid(model, Xte, N_IQN_EVAL, args.device)             # (B,N,H)
                else:
                    raise ValueError(v)

                q_conc = B7.back_to_conc("abs_delta", cur_te, q_norm, scale, eps_den)

                # CRPS
                if v == "base3":
                    crps_h = np.array([np.mean(B7.crps_quantiles(
                        q_conc[:, 0, h], q_conc[:, 1, h], q_conc[:, 2, h], y_te[:, h]))
                        for h in range(H)])
                elif v == "q9":
                    crps_h = np.array([np.mean(crps_cdf_pline(
                        q_conc[:, :, h], Q9_LEVELS, y_te[:, h])) for h in range(H)])
                elif v == "iqn":
                    qN_conc = B7.back_to_conc("abs_delta", cur_te, qN, scale, eps_den)
                    crps_h = np.array([np.mean(crps_energy(qN_conc[:, :, h], y_te[:, h]))
                                       for h in range(H)])

                # 覆盖率 [p10,p90] + p50 RMSE（iqn 用 τ=0.1/0.5/0.9 直接求值；
                # q9 用 9 通道中对应 p10/p50/p90 的通道：levels==0.1/0.5/0.9）
                if v == "iqn":
                    q3_conc = B7.back_to_conc("abs_delta", cur_te, q3, scale, eps_den)
                    cov = float(np.mean((y_te >= q3_conc[:, 0]) & (y_te <= q3_conc[:, 2])))
                    rmse = float(np.sqrt(np.mean((q3_conc[:, 1] - y_te) ** 2)))
                elif v == "q9":
                    i10 = int(np.where(np.isclose(Q9_LEVELS, 0.10))[0][0])
                    i50 = int(np.where(np.isclose(Q9_LEVELS, 0.50))[0][0])
                    i90 = int(np.where(np.isclose(Q9_LEVELS, 0.90))[0][0])
                    cov = float(np.mean((y_te >= q_conc[:, i10]) & (y_te <= q_conc[:, i90])))
                    rmse = float(np.sqrt(np.mean((q_conc[:, i50] - y_te) ** 2)))
                else:
                    cov = float(np.mean((y_te >= q_conc[:, 0]) & (y_te <= q_conc[:, 2])))
                    rmse = float(np.sqrt(np.mean((q_conc[:, 1] - y_te) ** 2)))

                # PIT 校准（iqn 专属）
                pit_m = np.zeros(H)
                pit_b = np.zeros(5)
                if v == "iqn":
                    pit = np.stack([pit_from_q(qN_conc[:, :, h], y_te[:, h])
                                    for h in range(H)], axis=1)  # (B,H)
                    pit_m = pit.mean(axis=0)
                    pit_b = np.histogram(pit, bins=np.linspace(0, 1, 6), density=False)[0]
                    pit_b = pit_b / max(pit_b.sum(), 1.0)

                crps_avg = float(np.mean(crps_h))
                agg[v]["crps"][wi, si] = crps_avg
                agg[v]["crps_h"][wi, si] = crps_h
                agg[v]["cov"][wi, si] = cov
                agg[v]["rmse"][wi, si] = rmse
                agg[v]["crps_p"][wi, si] = crps_p
                agg[v]["pit_mean"][wi, si] = pit_m
                agg[v]["pit_bins"][wi, si] = pit_b
                print(f"        覆盖={cov:.3f}  CRPS={crps_avg:.4f} (持久化 {crps_p:.4f})  "
                      f"p50RMSE={rmse:.3f}" + (f"  PIT均值={pit_m.mean():.3f}" if v == "iqn" else ""),
                      flush=True)

    # ---- 聚合：seed 均值 → 窗口均值 ----
    def s2(v, key):
        return agg[v][key].mean(axis=1)

    print("\n===== 逐窗口 conc 分布 =====", flush=True)
    print(pd.DataFrame(per_window_cur).to_string(index=False), flush=True)

    print("\n===== 变体对照（全部还原 conc 单位，17 窗口 × seed 均值）=====", flush=True)
    print(f"  {'变体':<8}{'覆盖':<8}{'CRPS':<9}{'持久化CRPS':<12}{'CRPS相对技能':<13}"
          f"{'p50RMSE':<9}{'PIT均值':<9}{'PIT|Δ0.5|':<9}", flush=True)
    for v in variants:
        a = agg[v]
        cp = s2(v, "crps_p").mean()
        rel = (cp - s2(v, "crps").mean()) / cp if cp else 0
        pit_mean = s2(v, "pit_mean").mean()
        pit_bins = s2(v, "pit_bins").mean(axis=0)
        print(f"  {v:<8}{s2(v,'cov').mean():<8.3f}{s2(v,'crps').mean():<9.4f}{cp:<12.4f}"
              f"{rel * 100:<13.1f}{s2(v,'rmse').mean():<9.3f}{pit_mean.mean():<9.3f}"
              f"{np.abs(pit_bins - 0.2).mean():<9.3f}", flush=True)

    print("\n===== 每变体 CRPS vs 持久化（%相对技能）=====", flush=True)
    for v in variants:
        a = agg[v]
        cp = s2(v, "crps_p").mean()
        rel = (cp - s2(v, "crps").mean()) / cp if cp else 0
        print(f"  {v:<8}: 模型CRPS {s2(v,'crps').mean():.4f} vs 持久化 {cp:.4f} → {rel * 100:+.1f}%", flush=True)

    print("\n===== 每视界 CRPS（conc 单位）=====", flush=True)
    print(f"  {'视界':<6}{'base3':<10}{'q9':<10}{'iqn':<10}", flush=True)
    for h in range(H):
        row = "  " + f"{h + 1:<6}"
        for v in variants:
            row += f"{s2(v, 'crps_h').mean(axis=0)[h]:<10.4f}"
        print(row, flush=True)

    print("\n===== IQN PIT 校准直方图（5 分箱，期望均匀 ≈0.2）=====", flush=True)
    if "iqn" in variants:
        pit_bins = s2("iqn", "pit_bins").mean(axis=0)
        print("  bins:", " ".join(f"{b:.3f}" for b in pit_bins), flush=True)

    # 输出 JSON（统计量，无原始数据行）
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {
        "protocol": {"train_days": B7.TRAIN_DAYS, "test_days": B7.TEST_DAYS,
                     "stride_days": B7.STRIDE_DAYS, "T": T, "H": H, "epochs": args.epochs,
                     "n_windows": Nw, "seeds": seeds, "iqn_eval_N": N_IQN_EVAL,
                     "iqn_train_K": K_TAU, "n_cos": N_COS, "target": "abs_delta"},
        "variants": {}, "windows_conc": per_window_cur,
    }
    for v in variants:
        a = agg[v]
        res["variants"][v] = {
            "coverage_mean": float(s2(v, "cov").mean()),
            "coverage_windows": s2(v, "cov").tolist(),
            "crps_mean": float(s2(v, "crps").mean()),
            "crps_h": s2(v, "crps_h").mean(axis=0).tolist(),
            "crps_persist": float(s2(v, "crps_p").mean()),
            "crps_rel_skill": float((s2(v, "crps_p").mean() - s2(v, "crps").mean())
                                    / s2(v, "crps_p").mean()),
            "rmse_conc_mean": float(s2(v, "rmse").mean()),
            "pit_mean_h": s2(v, "pit_mean").mean(axis=0).tolist() if v == "iqn" else None,
            "pit_bins": s2(v, "pit_bins").mean(axis=0).tolist() if v == "iqn" else None,
        }
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}（未含任何原始数据行）", flush=True)
    print(f"运行耗时: {(time.time() - t0) / 60:.1f} 分钟", flush=True)


if __name__ == "__main__":
    main()
