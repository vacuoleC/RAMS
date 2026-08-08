# -*- coding: utf-8 -*-
"""RAMS 训练编排：多任务 loss 加权（M1 回归 + M2 分层 + M4 预警）+ 分位数损失

依据探索测试结论：
  - loss 归一化/加权策略：M1 回归 + M2 分类，M2 权重更高（w2 分类优先最优）
  - 分位数损失（10/50/90）训练 M1，中位数作预测，区间作不确定性
  - M4 预警分级作为第三任务，用训练段阈值生成标签（防泄漏）
  - 支持 fast_dev_run（冒烟）和 3-seed 复现

两阶段训练（架构红线）：
  Stage 1: M1+M2+M4 联合训练共享 backbone（此处实现）
  Stage 2: 冻结 backbone，微调多头（预留接口）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.rams_net import RamsNet, QUANTILES


class MultiTaskLoss(nn.Module):
    """多任务损失：M1 分位数损失 + M2 交叉熵 + M4 交叉熵，可配权。"""

    def __init__(self, horizon: int, w_m1: float = 1.0, w_m2: float = 3.0,
                 w_m4: float = 2.0, use_m4: bool = True):
        """w_m2/w_m4 高于 w_m1：探索结论——辅助任务高权重对 M1 有正则/信息增益。"""
        super().__init__()
        self.horizon = horizon
        self.w_m1 = w_m1
        self.w_m2 = w_m2
        self.w_m4 = w_m4
        self.use_m4 = use_m4
        self.ce = nn.CrossEntropyLoss()

    def forward(self, m1_out, m2_out, y, strat_label, m4_out=None, warn_label=None):
        """m1_out: (B, 3H) 分位数; y: (B, H); strat_label: (B,); m4/warn 可选。"""
        # M1 分位数损失
        H = self.horizon
        qs = torch.tensor(QUANTILES, device=y.device)
        yq = y.unsqueeze(1)  # (B, 1, H)
        m1_q = m1_out.reshape(-1, 3, H)  # (B, 3, H)
        e = yq - m1_q
        losses = [torch.mean(torch.maximum(q * e[:, i], (q - 1) * e[:, i]))
                  for i, q in enumerate(qs)]
        l1 = torch.stack(losses).mean()

        # M2 交叉熵
        l2 = self.ce(m2_out, strat_label)

        # M4 预警交叉熵（可选）
        l4 = None
        if self.use_m4 and m4_out is not None and warn_label is not None:
            l4 = self.ce(m4_out, warn_label)

        total = self.w_m1 * l1 + self.w_m2 * l2
        if l4 is not None:
            total = total + self.w_m4 * l4
        return total, l1, l2, l4


class Trainer:
    """轻量训练器（不依赖 Lightning，探索阶段够用）。"""

    def __init__(self, model: RamsNet, lr: float = 1e-3, w_m1: float = 1.0,
                 w_m2: float = 3.0, w_m4: float = 2.0, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.use_m4 = model.use_m4
        self.criterion = MultiTaskLoss(model.horizon, w_m1, w_m2, w_m4, self.use_m4)

    def fit(self, X_tr, y_tr, strat_tr, X_va, y_va, strat_va,
            warn_tr=None, warn_va=None, epochs: int = 30,
            batch_size: int = 128, fast_dev_run: bool = False):
        """训练。fast_dev_run=True 时只跑 1-2 个 batch 冒烟。"""
        # 构建数据集（含可选的 M4 标签）
        tensors = [torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(strat_tr)]
        if self.use_m4 and warn_tr is not None:
            tensors.append(torch.tensor(warn_tr))
        ds = TensorDataset(*tensors)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        Xv = torch.tensor(X_va).to(self.device)
        yv = torch.tensor(y_va).to(self.device)
        sv = torch.tensor(strat_va).to(self.device)
        wv = torch.tensor(warn_va).to(self.device) if (self.use_m4 and warn_va is not None) else None

        epochs = 2 if fast_dev_run else epochs
        history = []
        for ep in range(epochs):
            self.model.train()
            for bi, batch in enumerate(dl):
                if fast_dev_run and bi >= 2:
                    break
                xb = batch[0].to(self.device)
                yb = batch[1].to(self.device)
                sb = batch[2].to(self.device)
                wb = batch[3].to(self.device) if len(batch) > 3 else None
                self.opt.zero_grad()
                m1, m2, m4 = self.model(xb)
                loss, l1, l2, l4 = self.criterion(m1, m2, yb, sb, m4, wb)
                loss.backward()
                self.opt.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                m1v, m2v, m4v = self.model(Xv)
                pred = self.model.predict_mean(m1v)
                val_rmse = torch.sqrt(torch.mean((pred - yv) ** 2)).item()
                val_acc = (m2v.argmax(1) == sv).float().mean().item()
                extra = ""
                if m4v is not None and wv is not None:
                    val_wacc = (m4v.argmax(1) == wv).float().mean().item()
                    extra = f" val_wacc={val_wacc:.4f}"
            history.append((loss.item(), val_rmse, val_acc))
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"  ep{ep} loss={loss.item():.4f} val_rmse={val_rmse:.4f} val_acc={val_acc:.4f}{extra}", flush=True)
        return history

    def evaluate(self, X_te, y_te, strat_te, warn_te=None, y_sd: float = 1.0):
        """测试集评估：RMSE（还原尺度）+ M2 acc + M4 acc + 区间覆盖率。"""
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_te).to(self.device)
            m1, m2, m4 = self.model(Xt)
            pred = self.model.predict_mean(m1).cpu().numpy()
            acc = (m2.argmax(1).cpu() == torch.tensor(strat_te)).float().mean().item()
            rmse_norm = np.sqrt(np.mean((pred - np.array(y_te)) ** 2))
            result = {
                "rmse": float(rmse_norm * y_sd),
                "rmse_norm": float(rmse_norm),
                "acc": float(acc),
            }
            if m4 is not None and warn_te is not None:
                result["warn_acc"] = float(
                    (m4.argmax(1).cpu() == torch.tensor(warn_te)).float().mean().item())
            if self.model.quantile:
                p10, p90 = self.model.predict_interval(m1)
                p10 = p10.cpu().numpy(); p90 = p90.cpu().numpy()
                cover = np.mean((np.array(y_te) >= p10) & (np.array(y_te) <= p90))
                result["coverage"] = float(cover)
                result["interval_width"] = float(np.mean(p90 - p10) * y_sd)
        return result


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="RAMS 训练（冒烟）")
    parser.add_argument("--parquet", default="data/processed/standard.parquet")
    parser.add_argument("--fast-dev-run", action="store_true", help="冒烟模式")
    args = parser.parse_args()

    sys.path.insert(0, ".")
    from rams.data.tensor_builder import TensorBuilder, TensorConfig

    cfg = TensorConfig(T=24, H=8, warn_as_task=True)
    ds = TensorBuilder(cfg).build(args.parquet)
    (X_tr, y_tr, s_tr, w_tr), (X_va, y_va, s_va, w_va), (X_te, y_te, s_te, w_te) = (
        ds["train"], ds["val"], ds["test"])

    model = RamsNet(feat_dim=ds["feat_dim"], horizon=cfg.H, use_m4=True)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    trainer = Trainer(model)
    trainer.fit(X_tr, y_tr, s_tr, X_va, y_va, s_va,
                warn_tr=w_tr, warn_va=w_va, epochs=2, fast_dev_run=args.fast_dev_run)
    if not args.fast_dev_run:
        res = trainer.evaluate(X_te, y_te, s_te, w_te, ds["y_sd"])
        print(f"测试: RMSE={res['rmse']:.4f} acc={res['acc']:.4f} "
              f"warn_acc={res.get('warn_acc', 0):.4f} coverage={res.get('coverage', 0):.3f}")
    print("冒烟通过")
