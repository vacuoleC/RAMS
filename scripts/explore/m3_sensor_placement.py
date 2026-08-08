# -*- coding: utf-8 -*-
"""M3 垂向监测点位优化：GAT 打分器 + 贪心选层（Salem & Abokifa 思路）

目标：在 20 个水深（0.5~10.0m）中选出最优传感器子集（5~8 层），
使该子集能最好地重建全剖面水温 + 预测表层藻类浓度。

方法（核心设计，来自文献 Salem & Abokifa）：
  1. GAT 打分器：以「深度图」为输入 —— 20 个深度节点 + 1 个重建节点(R)。
     **在随机深度子集下训练**：每个样本随机采样可见层 k~U[1,20]，
     观测温度值只从可见层进入节点特征，隐藏层特征置零 + 相关边剔除，
     R 节点聚合可见信息，输出对全剖面的重建。
     可见层重建=观测值本身（传感器读数已知），模型只负责补齐隐藏层。
  2. 贪心前向选层：从空集开始，每步加入使验证集剖面重建误差下降最多的层，
     直到预算层数。

保密：只打印形状/统计量/结论，不打印原始数据数值行。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rams.data.tensor_builder import TensorBuilder, TensorConfig  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DEPTHS = [0.5 + 0.5 * i for i in range(20)]  # 0.5 ~ 10.0 m
N_DEPTH = len(DEPTHS)        # 20
R_NODE = N_DEPTH             # 重建节点 id = 20
N_NODES = N_DEPTH + 1        # 21
METEO_COLS = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
F_EMB = 16                   # 深度位置编码维度
N_METEO = len(METEO_COLS)    # 6


# ---------- 图结构（固定拓扑：距离 1/2 的深度边 + 深度→R） ----------
def _make_edges() -> tuple[np.ndarray, np.ndarray]:
    src, dst = [], []
    for d in (1, 2):
        for i in range(N_DEPTH - d):
            src.extend([i, i + d])
            dst.extend([i + d, i])
    for i in range(N_DEPTH):
        src.append(i)
        dst.append(R_NODE)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)


BASE_SRC, BASE_DST = _make_edges()


class GATScorer(nn.Module):
    """深度子集打分器：可见层观测值 → 图注意力聚合 → 重建全剖面 + 预测表层浓度。

    关键设计：
      - 节点特征 = [深度位置编码, 观测温度(仅可见层非零), 气象(全局),
        最近可见层温度 + 距离(插值先验, 使模型至少达到线性插值质量)],
        隐藏层其余特征置零 + 相关边剔除，只从可见子集传播信息。
      - 重建输出与观测值混合（blend）：可见层输出=观测值（传感器读数已知），
        隐藏层输出=模型补齐。损失只惩罚隐藏层的补齐误差。
    """

    def __init__(self, heads: int = 4, hid: int = 96, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(N_NODES, F_EMB)
        feat_in = F_EMB + 1 + N_METEO + 2   # emb + obs_val + meteo + (nearest_val, nearest_dist)
        self.gat = nn.ModuleList([
            GATConv(feat_in, hid // heads, heads=heads, concat=True, dropout=dropout),
            GATConv(hid, hid, heads=1, concat=True, dropout=dropout),
        ])
        self.recon_head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, N_DEPTH))
        self.conc_head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, 1))

    def _batch_edge_index(self, mask: torch.Tensor) -> torch.Tensor:
        """按各样本可见子集剔除边，拼成跨 batch 的 edge_index（每个图独立分量）。"""
        B = mask.shape[0]
        src = torch.as_tensor(BASE_SRC, device=mask.device)
        dst = torch.as_tensor(BASE_DST, device=mask.device)
        keep = mask[:, src].bool() & mask[:, dst].bool()  # (B, E)
        e_idx = keep.nonzero()                            # (K, 2): [b, e]
        b_idx, e = e_idx[:, 0], e_idx[:, 1]
        off = b_idx * N_NODES
        return torch.stack([src[e] + off, dst[e] + off])

    def _nearest_obs(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """每个深度节点最近的可见层观测值与其距离（深度索引差）。

        Returns: nearest_val (B,20), nearest_dist (B,20)。无可见层时安全返回 0 / N_DEPTH。
        """
        B = values.shape[0]
        vis = mask[:, :N_DEPTH]                              # (B,20)
        vals = values[:, :N_DEPTH]                           # (B,20)
        d = torch.arange(N_DEPTH, device=values.device).float()
        dist_mat = (d[:, None] - d[None, :]).abs()           # (20,20)
        dd = torch.where(vis.unsqueeze(1), dist_mat.unsqueeze(0),
                         torch.full((B, N_DEPTH, N_DEPTH), 1e6, device=values.device))
        anyv = vis.any(dim=1, keepdim=True)                  # (B,1)
        nd, ni = dd.min(dim=2)                               # (B,20)
        nd = torch.where(anyv, nd, torch.full_like(nd, float(N_DEPTH)))
        nv = torch.gather(vals, 1, ni)
        nv = torch.where(anyv, nv, torch.zeros_like(nv))
        return nv, nd / float(N_DEPTH)

    def forward(self, node_ids, values, mask, meteo):
        """node_ids:(B,N); values:(B,N) 观测温度(隐藏层为0); mask:(B,N) bool;
        meteo:(B,6)。返回 (recon (B,20), conc (B,1))。"""
        B, N = mask.shape
        nv, nd = self._nearest_obs(values, mask)             # (B,20) each
        extra = torch.stack([nv, nd], dim=-1)                # (B,20,2)
        extra = torch.cat([extra, torch.zeros(B, 1, 2, device=values.device)], dim=1)  # R 节点补零 (B,N,2)
        emb = self.embed(node_ids)                           # (B,N,F_EMB)
        obs = values.unsqueeze(-1) * mask.unsqueeze(-1).float()  # (B,N,1)
        met = meteo.unsqueeze(1).expand(B, N, -1)            # (B,N,6)
        x = torch.cat([emb, obs, met, extra], dim=-1)        # (B,N,feat_in)
        x = x.reshape(B * N, -1)
        maskf = mask.reshape(-1, 1).float()
        x = x * maskf                                        # 隐藏节点整段置零
        edge_index = self._batch_edge_index(mask)
        for layer in self.gat:
            x = layer(x, edge_index)
            x = x * maskf
        r = x.view(B, N, -1)[:, R_NODE]                      # (B, hid)
        recon = self.recon_head(r)                           # (B, 20) 全剖面预测
        # blend：可见层输出 = 观测值本身（读数已知），隐藏层 = 模型补齐
        obs = values[:, :N_DEPTH]
        vis = mask[:, :N_DEPTH]
        recon = torch.where(vis, obs, recon)
        return recon, self.conc_head(r)


# ---------- 数据 ----------
def load_data(parquet: str, val_frac: float = 0.15) -> dict:
    """读宽表（按训练段归一化），返回训练/验证/测试张量与归一化参数。"""
    cfg = TensorConfig()
    builder = TensorBuilder(cfg)
    wide = builder._load_wide(parquet).sort_index()
    n = len(wide)
    n_tr, n_va = int(n * 0.7), int(n * val_frac)
    builder._fit_stats(wide)                 # 只用训练段拟合
    wide = builder._normalize(wide)

    temp_cols = [c for c in wide.columns if c.startswith("temp_")]
    conc_cols = [c for c in wide.columns if c.startswith("conc_")]
    temp = wide[temp_cols].values.astype(np.float32)        # (n, 20) 归一化
    conc = wide[conc_cols[0]].values.astype(np.float32)     # (n,) 表层浓度 归一化
    meteo = wide[METEO_COLS].values.astype(np.float32)      # (n, 6) 归一化

    # 每层 std / 表层浓度 std（还原真实单位，仅用于报告）
    tr = wide.iloc[:n_tr]
    temp_std = tr[temp_cols].std().values.astype(np.float32)
    conc_std = float(tr[conc_cols[0]].std()) + 1e-8

    return {
        "temp": temp, "conc": conc, "meteo": meteo, "n": n,
        "idx_tr": np.arange(n_tr), "idx_va": np.arange(n_tr, n_tr + n_va),
        "idx_te": np.arange(n_tr + n_va, n),
        "temp_std": torch.from_numpy(temp_std).float().to(DEV),
        "conc_std": conc_std,
    }


def _random_subset_mask(batch_size: int, device) -> torch.Tensor:
    """每个样本随机采样 k ~ U[1,20] 个可见深度层，R 节点始终可见。"""
    mask = np.zeros((batch_size, N_NODES), dtype=np.bool_)
    for b in range(batch_size):
        k = np.random.randint(1, N_DEPTH + 1)
        idx = np.random.choice(N_DEPTH, k, replace=False)
        mask[b, idx] = True
    mask[:, R_NODE] = True
    return torch.from_numpy(mask).to(device)


def train_scorer(data: dict, epochs: int, batch_size: int, lr: float, seed: int) -> GATScorer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GATScorer().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    node_ids = torch.arange(N_NODES, device=DEV).long().unsqueeze(0)  # (1, N)

    temp = torch.from_numpy(data["temp"]).to(DEV)
    conc = torch.from_numpy(data["conc"]).to(DEV)
    meteo = torch.from_numpy(data["meteo"]).to(DEV)
    tr_idx, va_idx = data["idx_tr"], data["idx_va"]

    t0 = time.time()
    for ep in range(epochs):
        model.train()
        order = np.random.permutation(tr_idx)
        epoch_loss, n_batch = 0.0, 0
        for s in range(0, len(order), batch_size):
            idx = order[s:s + batch_size]
            B = len(idx)
            ids = node_ids.repeat(B, 1)
            mask = _random_subset_mask(B, DEV)
            vals = torch.zeros(B, N_NODES, device=DEV)
            vals[:, :N_DEPTH] = temp[idx]
            recon, cpred = model(ids, vals, mask, meteo[idx])
            xb = temp[idx]; cb = conc[idx]
            l_prof = F.mse_loss(recon, xb)
            l_conc = F.mse_loss(cpred.squeeze(1), cb)
            loss = l_prof + 0.5 * l_conc
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item(); n_batch += 1

        # 验证：随机子集（不同大小）下的剖面重建质量
        model.eval()
        with torch.no_grad():
            ids = node_ids.repeat(len(va_idx), 1)
            vmask = _random_subset_mask(len(va_idx), DEV)
            vvals = torch.zeros(len(va_idx), N_NODES, device=DEV)
            vvals[:, :N_DEPTH] = temp[va_idx]
            vr, vc = model(ids, vvals, vmask, meteo[va_idx])
            prof_rmse = torch.sqrt(((vr - temp[va_idx]) ** 2).mean(dim=0) * data["temp_std"] ** 2).mean()
            conc_rmse = torch.sqrt(((vc.squeeze(1) - conc[va_idx]) ** 2).mean()) * data["conc_std"]
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  ep{ep:02d} loss={epoch_loss/n_batch:.4f} "
                  f"val_prof_rmse={prof_rmse.item():.4f}°C val_conc_rmse={conc_rmse.item():.4f}", flush=True)
    print(f"  训练完成 {epochs} epochs, {time.time()-t0:.1f}s", flush=True)
    return model


def _eval_subset(model, data, subset, idx):
    """给定深度子集在 idx 上的剖面重建 RMSE(°C) 与表层浓度 RMSE。"""
    model.eval()
    with torch.no_grad():
        B = len(idx)
        node_ids = torch.arange(N_NODES, device=DEV).long().unsqueeze(0).repeat(B, 1)
        mask = torch.zeros((B, N_NODES), dtype=torch.bool, device=DEV)
        for s in subset:
            mask[:, s] = True
        mask[:, R_NODE] = True
        vals = torch.zeros(B, N_NODES, device=DEV)
        vals[:, :N_DEPTH] = torch.from_numpy(data["temp"][idx]).to(DEV)
        meteo = torch.from_numpy(data["meteo"][idx]).to(DEV)
        temp = torch.from_numpy(data["temp"][idx]).to(DEV)
        conc = torch.from_numpy(data["conc"][idx]).to(DEV)
        recon, cpred = model(node_ids, vals, mask, meteo)
        prof_rmse = torch.sqrt(((recon - temp) ** 2).mean(dim=0) * data["temp_std"] ** 2).mean().item()
        conc_rmse = torch.sqrt(((cpred.squeeze(1) - conc) ** 2).mean()).item() * data["conc_std"]
        return prof_rmse, conc_rmse


def _hidden_mask(subset) -> np.ndarray:
    """返回 (N_DEPTH,) bool：不在子集中的层为 True（需补齐的隐藏层）。"""
    return np.array([i not in subset for i in range(N_DEPTH)])


def _te_report_subset(model, data, subset, label: str):
    """测试集报告一个给定子集（用于均匀基线等非贪心子集）。"""
    idx_te = data["idx_te"]
    p, c = _eval_subset(model, data, subset, idx_te)
    ih = _interp_baseline(data, subset, idx_te)
    print(f"  {label:<10s} k={len(subset):<2d} 子集={[f'{DEPTHS[i]:.1f}' for i in subset]}", flush=True)
    print(f"             GAT prof_rmse={p:.4f}°C  conc_rmse={c:.4f}  "
          f"线性插值隐藏层RMSE={ih:.4f}°C", flush=True)
    return p, c


def _interp_baseline(data, subset, idx) -> float:
    """线性插值基线：用可见层深度-温度线性插值补齐隐藏层，返回隐藏层 RMSE(°C)。

    空子集（无传感器）退化为「各层用该层时间均值」的平凡基线。
    """
    temp = data["temp"][idx]          # (B, 20) 归一化
    hid = _hidden_mask(subset)        # (20,) bool
    t_std = data["temp_std"].cpu().numpy()
    errs = []
    if not subset:
        # 平凡基线：用各层训练均值预测 → 误差 = 各层 std
        return float(np.sqrt(np.mean(t_std ** 2)))
    obs_depths = np.array([DEPTHS[i] for i in subset])
    for i in range(len(idx)):
        obs_vals = temp[i, subset]
        interp = np.interp(DEPTHS, obs_depths, obs_vals)  # 全 20 层线性插值
        errs.append((interp[hid] - temp[i, hid]) * t_std[hid])
    errs = np.concatenate(errs)
    return float(np.sqrt(np.mean(errs ** 2)))


def greedy_interp(data: dict, budget: int) -> list[int]:
    """贪心线性插值基线：用插值重建误差选层（对照，验证 GAT 打分器是否优于平凡法）。"""
    idx_va = data["idx_va"]
    selected: list[int] = []
    remaining = list(range(N_DEPTH))
    prev = _interp_baseline(data, [], idx_va)
    print(f"  [插值基线 无传感器] prof_rmse={prev:.4f}°C", flush=True)
    for step in range(1, budget + 1):
        best_err, best_layer = float("inf"), None
        for l in remaining:
            cand = selected + [l]
            err = _interp_baseline(data, cand, idx_va)
            if err < best_err:
                best_err, best_layer = err, l
        selected.append(best_layer)
        remaining.remove(best_layer)
        print(f"  插值选第{step:2d}层 depth={DEPTHS[best_layer]:.1f}m → "
              f"prof_rmse={best_err:.4f}°C  Δ={prev - best_err:+.4f}", flush=True)
        prev = best_err
    return selected


def greedy_select(model, data: dict, budget: int) -> dict:
    """贪心前向选层：从空集开始，每步加入使验证集剖面重建误差下降最多的层。"""
    idx_va = data["idx_va"]
    selected: list[int] = []
    remaining = list(range(N_DEPTH))
    history = []  # (step, layer_idx, prof_rmse, conc_rmse, delta)

    # 全 20 层参考（重建下限：所有层可见 → 误差≈0）
    base_prof, base_conc = _eval_subset(model, data, list(range(N_DEPTH)), idx_va)
    # 空集基线（无传感器：仅气象 → 剖面预测），贪心起点
    empty_prof, empty_conc = _eval_subset(model, data, [], idx_va)
    print(f"  [全20层] prof_rmse={base_prof:.4f}°C  conc_rmse={base_conc:.4f}", flush=True)
    print(f"  [无传感器] prof_rmse={empty_prof:.4f}°C  conc_rmse={empty_conc:.4f}（仅气象）", flush=True)
    prev = empty_prof

    for step in range(1, budget + 1):
        best_err, best_conc, best_layer = float("inf"), float("inf"), None
        for l in remaining:
            cand = selected + [l]
            prof_rmse, conc_rmse = _eval_subset(model, data, cand, idx_va)
            if prof_rmse < best_err:
                best_err, best_conc, best_layer = prof_rmse, conc_rmse, l
        delta = prev - best_err
        selected.append(best_layer)
        remaining.remove(best_layer)
        history.append((step, best_layer, best_err, best_conc, delta))
        prev = best_err
        print(f"  选第{step:2d}层 depth={DEPTHS[best_layer]:.1f}m → "
              f"prof_rmse={best_err:.4f}°C  conc_rmse={best_conc:.4f}  Δ={delta:+.4f}", flush=True)
    return {"selected": selected, "history": history, "base_prof": base_prof, "base_conc": base_conc}


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 垂向监测点位优化（GAT+贪心）")
    parser.add_argument("--parquet", default="data/processed/standard.parquet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--budget", type=int, default=8, help="贪心选层预算")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="冒烟：2 epochs + budget=5")
    args = parser.parse_args()

    if args.smoke:
        args.epochs, args.budget = 2, 5

    data = load_data(args.parquet)
    print(f"宽表样本数: {data['n']}  train/val/test = "
          f"{len(data['idx_tr'])}/{len(data['idx_va'])}/{len(data['idx_te'])}", flush=True)
    print(f"深度层: {DEPTHS[0]}~{DEPTHS[-1]}m 共 {N_DEPTH} 层, 图节点数={N_NODES}, "
          f"基础边数={len(BASE_SRC)}, 气象特征={N_METEO}", flush=True)

    model = train_scorer(data, args.epochs, args.batch_size, args.lr, args.seed)

    print("\n=== 贪心选层（验证集） ===", flush=True)
    res = greedy_select(model, data, args.budget)
    sel, hist = res["selected"], res["history"]

    print("\n=== 对照：线性插值贪心选层（验证集） ===", flush=True)
    sel_interp = greedy_interp(data, args.budget)
    interp_5 = _interp_baseline(data, sel_interp[:5], data["idx_te"])
    print(f"  插值贪心最优5层(测试集): {[f'{DEPTHS[i]:.1f}m' for i in sel_interp[:5]]} "
          f"prof_rmse={interp_5:.4f}°C", flush=True)

    # ---- 测试集最终报告 ----
    idx_te = data["idx_te"]
    print("\n=== 测试集评估 ===", flush=True)
    p_all, c_all = _eval_subset(model, data, list(range(N_DEPTH)), idx_te)
    p_empty, c_empty = _eval_subset(model, data, [], idx_te)
    print(f"  [全20层]   prof_rmse={p_all:.4f}°C  conc_rmse={c_all:.4f}", flush=True)
    print(f"  [无传感器] prof_rmse={p_empty:.4f}°C  conc_rmse={c_empty:.4f}（仅气象）", flush=True)

    def te_report(k: int, label: str):
        subset = sel[:k]
        p, c = _eval_subset(model, data, subset, idx_te)
        ih = _interp_baseline(data, subset, idx_te)
        print(f"  {label:<10s} k={k:<2d} 子集={[f'{DEPTHS[i]:.1f}' for i in subset]}", flush=True)
        print(f"             GAT prof_rmse={p:.4f}°C  conc_rmse={c:.4f}  "
              f"线性插值隐藏层RMSE={ih:.4f}°C", flush=True)
        return p, c

    p1, _ = te_report(1, "最优单层")
    p5, c5 = te_report(5, "贪心最优5层")
    k8 = min(8, args.budget)
    p8, c8 = te_report(k8, f"贪心最优{k8}层")
    uniform5 = [i for i in range(0, N_DEPTH, 4)]  # 0.5/2.5/4.5/6.5/8.5m
    pu, cu = te_report(5, "均匀5层基线") if sel[:5] == uniform5 else (
        _te_report_subset(model, data, uniform5, "均匀5层基线"))
    _ = p1, c5, p8, c8

    print("\n=== 结论 ===", flush=True)
    print(f"  最优 5 层子集: {[f'{DEPTHS[i]:.1f}m' for i in sel[:5]]}", flush=True)
    print(f"  选层顺序(贪心): {[f'{DEPTHS[i]:.1f}m' for i in sel]}", flush=True)
    print("  剖面重建误差随层数（验证集）:", flush=True)
    for step, layer, perr, cerr, delta in hist:
        print(f"    k={step:2d} → 加入 {DEPTHS[layer]:.1f}m | prof_rmse={perr:.4f}°C "
              f"(Δ={delta:+.4f})", flush=True)
    print(f"  5层 vs 全20层: +{p5 - p_all:.4f}°C 剖面重建损失", flush=True)
    print(f"  5层 vs 无传感器基线: 剖面误差下降 {p_empty - p5:.4f}°C "
          f"（{100*(p_empty - p5)/max(p_empty, 1e-9):.1f}%）", flush=True)
    print(f"  5层表层浓度精度损失 vs 全20层: {c5 - c_all:+.4f}", flush=True)
    print(f"  5层 vs 均匀5层基线: 剖面误差降低 {pu - p5:.4f}°C", flush=True)
    print(f"  5层 vs 插值贪心5层: 剖面误差降低 {interp_5 - p5:.4f}°C", flush=True)


if __name__ == "__main__":
    main()
