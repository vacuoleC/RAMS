# -*- coding: utf-8 -*-
"""B1 对照：绝对浓度直接预测（M1 口径）——验证增量是否优于"直接预测绝对值"。

与 b1_increment.py 相同滚动窗口协议，唯一差别：目标 = conc_{t+h}（绝对值），
分位数头直接输出绝对浓度 p10/p50/p90，评估 RMSE/CRPS 直接算。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # 复用 b1_increment 的工具函数

from b1_increment import (DEVICE, QUANTILES, SharedGRU, crps_quantile, load_wide,
                          make_increment_tensors, rmse, surface_conc_cols)


class AbsoluteQuantileNet(nn.Module):
    """与 IncrementQuantileNet 同架构，输出绝对浓度分位数。"""

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
        return self.head(self.backbone(x))


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, qs=QUANTILES) -> torch.Tensor:
    H = target.shape[1]
    e = target.unsqueeze(1) - pred.reshape(-1, 3, H)
    losses = [torch.mean(torch.maximum(q * e[:, i], (q - 1) * e[:, i]))
              for i, q in enumerate(qs)]
    return torch.stack(losses).mean()


def train_abs(model, X_tr, y_tr, X_va, y_va, epochs, lr, batch_size=256, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(X_tr)
    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=DEVICE)
    Xv = torch.tensor(X_va, device=DEVICE)
    yv = torch.tensor(y_va, dtype=torch.float32, device=DEVICE)
    H_ = y_tr.shape[1]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            loss = quantile_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xv).reshape(-1, 3, H_)
            torch.sqrt(((pv[:, 1, :] - yv) ** 2).mean())


def rolling_evaluate_abs(wide, T, Hmax, horizons, train_years, test_months,
                         stride_months, epochs, lr, hidden, seed, verbose):
    surf = surface_conc_cols(wide)[0]
    y_all = wide[surf].values.astype(np.float64)
    times = pd.DatetimeIndex(wide.index)
    idx = np.arange(len(wide))
    t_end = times[-1]

    X_all, delta_all, t_idx, feat_cols = make_increment_tensors(
        wide, T, Hmax)  # 同一特征（含表层浓度 conc_t），仅目标不同
    y_tgt = np.stack([y_all[t_idx + T + h - 1] for h in range(Hmax)], axis=1)  # 绝对值目标

    cur = times[0]
    fold = 0
    acc = {h: {"rmse_m": [], "rmse_p": [], "crps_m": [], "crps_p": []} for h in horizons}
    while True:
        t_train_end = cur + pd.DateOffset(years=train_years)
        t_test_end = t_train_end + pd.DateOffset(months=test_months)
        if t_test_end > t_end:
            break
        tr_rows = idx[(times >= cur) & (times < t_train_end)]
        te_rows = idx[(times >= t_train_end) & (times < t_test_end)]
        if len(tr_rows) < T + Hmax + 20 or len(te_rows) < 20:
            break
        tr_i = np.where((t_idx >= tr_rows[0]) & (t_idx + T - 1 <= tr_rows[-1] - 1))[0]
        te_i = np.where((t_idx >= tr_rows[-1]) & (t_idx + T - 1 <= te_rows[-1] - 1))[0]
        if len(tr_i) < 300 or len(te_i) < 20:
            break

        sub = wide.iloc[tr_rows]
        stats = {c: (float(sub[c].mean()), float(sub[c].std()) + 1e-8) for c in wide.columns}
        wide_norm = wide.copy()
        for c, (mu, sd) in stats.items():
            wide_norm[c] = (wide_norm[c] - mu) / sd
        X_norm, _, _, _ = make_increment_tensors(wide_norm, T, Hmax)

        tr_i_sorted = tr_i
        va_i = tr_i_sorted[-min(500, len(tr_i_sorted)):]
        if verbose:
            print(f"  fold{fold} @{cur.date()} train={len(tr_i)} test={len(te_i)}", flush=True)
        model = AbsoluteQuantileNet(feat_dim=X_norm.shape[2], horizon=Hmax,
                                    hidden=hidden).to(DEVICE)
        train_abs(model, X_norm[tr_i_sorted], y_tgt[tr_i_sorted],
                  X_norm[va_i], y_tgt[va_i], epochs=epochs, lr=lr, seed=seed)

        model.eval()
        with torch.no_grad():
            pred_q = model(torch.tensor(X_norm[te_i], device=DEVICE))
            pred_q = pred_q.reshape(-1, 3, Hmax).cpu().numpy()
        for h in horizons:
            i_rows = t_idx[te_i]
            base = y_all[i_rows + T - 1]
            obs = y_all[i_rows + T + h - 1]
            pred_conc = pred_q[:, 1, h - 1]
            pred_p10 = pred_q[:, 0, h - 1]
            pred_p90 = pred_q[:, 2, h - 1]
            pred_persist = base
            qs_model = np.stack([pred_p10, pred_conc, pred_p90], axis=1)
            acc[h]["rmse_m"].append(rmse(obs, pred_conc))
            acc[h]["rmse_p"].append(rmse(obs, pred_persist))
            acc[h]["crps_m"].append(crps_quantile(obs, qs_model, np.array([0.1, 0.5, 0.9])))
            acc[h]["crps_p"].append(crps_quantile(obs, np.full((len(obs), 3), base[:, None]), np.array([0.1, 0.5, 0.9])))

        cur = cur + pd.DateOffset(months=stride_months)
        fold += 1

    summary = {}
    for h in horizons:
        a = acc[h]
        summary[h] = {
            "model_rmse": float(np.mean(a["rmse_m"])),
            "persist_rmse": float(np.mean(a["rmse_p"])),
            "model_crps": float(np.mean(a["crps_m"])),
            "persist_crps": float(np.mean(a["crps_p"])),
            "folds": fold,
        }
    return summary, fold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="D:/coding/26AIendwork/data/processed/standard.parquet")
    ap.add_argument("--T", type=int, default=24)
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--train-years", type=int, default=2)
    ap.add_argument("--test-months", type=int, default=3)
    ap.add_argument("--stride-months", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="abs")
    args = ap.parse_args()

    t0 = time.time()
    wide = load_wide(args.parquet)
    horizons = list(range(1, args.H + 1))
    summary, folds = rolling_evaluate_abs(
        wide, args.T, args.H, horizons, args.train_years, args.test_months,
        args.stride_months, args.epochs, args.lr, args.hidden, args.seed, True)
    print(f"\n[{args.tag}] 绝对浓度直接预测（folds={folds}, {time.time()-t0:.0f}s）")
    print(f"{'h':>3} {'模型RMSE':>9} {'持久化':>9} | {'模型CRPS':>9} {'持久化':>9}")
    for h in horizons:
        s = summary[h]
        print(f"{h:>3} {s['model_rmse']:>9.4f} {s['persist_rmse']:>9.4f}"
              f" | {s['model_crps']:>9.4f} {s['persist_crps']:>9.4f}")
    print("\n超越持久化判定：")
    for h in horizons:
        s = summary[h]
        d = s["model_rmse"] - s["persist_rmse"]
        mark = "模型√" if d < 0 else "持久化√"
        print(f"  h={h}: 模型 {s['model_rmse']:.4f} vs 持久化 {s['persist_rmse']:.4f}（Δ={d:+.4f}）→ {mark}")


if __name__ == "__main__":
    main()
