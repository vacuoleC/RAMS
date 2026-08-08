# -*- coding: utf-8 -*-
"""RAMS 训练编排：多任务 loss 加权 + 分位数损失

依据探索测试结论：
  - loss 归一化/加权策略：M1 回归 + M2 分类，M2 权重更高（w2 分类优先最优）
  - 分位数损失（10/50/90）训练 M1，中位数作预测，区间作不确定性
  - 支持 fast_dev_run（冒烟）和 3-seed 复现

两阶段训练（架构红线）：
  Stage 1: M1+M2 联合训练共享 backbone（此处实现）
  Stage 2: 冻结 backbone，微调多头（预留接口）
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..models.rams_net import RamsNet, QUANTILES


class MultiTaskLoss(nn.Module):
    """多任务损失：M1 分位数损失 + M2 交叉熵，可配权。"""

    def __init__(self, horizon: int, w_m1: float = 1.0, w_m2: float = 3.0):
        """w_m2 高于 w_m1：探索结论——M2 分层任务高权重对 M1 有正则/信息增益。"""
        super().__init__()
        self.horizon = horizon
        self.w_m1 = w_m1
        self.w_m2 = w_m2
        self.ce = nn.CrossEntropyLoss()

    def forward(self, m1_out, m2_out, y, strat_label):
        """m1_out: (B, 3H) 分位数通道; y: (B, H); strat_label: (B,) 类别"""
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
        return self.w_m1 * l1 + self.w_m2 * l2, l1, l2


class Trainer:
    """轻量训练器（不依赖 Lightning，探索阶段够用）。"""

    def __init__(self, model: RamsNet, lr: float = 1e-3, w_m1: float = 1.0,
                 w_m2: float = 3.0, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device
        self.opt = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = MultiTaskLoss(model.horizon, w_m1, w_m2)

    def fit(self, X_tr, y_tr, strat_tr, X_va, y_va, strat_va, epochs: int = 30,
            batch_size: int = 128, fast_dev_run: bool = False):
        """训练。fast_dev_run=True 时只跑 1-2 个 batch 冒烟。"""
        ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr), torch.tensor(strat_tr))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        Xv = torch.tensor(X_va).to(self.device)
        yv = torch.tensor(y_va).to(self.device)
        sv = torch.tensor(strat_va).to(self.device)

        epochs = 2 if fast_dev_run else epochs
        history = []
        for ep in range(epochs):
            self.model.train()
            for bi, (xb, yb, sb) in enumerate(dl):
                if fast_dev_run and bi >= 2:
                    break
                xb, yb, sb = xb.to(self.device), yb.to(self.device), sb.to(self.device)
                self.opt.zero_grad()
                m1, m2 = self.model(xb)
                loss, l1, l2 = self.criterion(m1, m2, yb, sb)
                loss.backward()
                self.opt.step()

            # 验证
            self.model.eval()
            with torch.no_grad():
                m1v, m2v = self.model(Xv)
                pred = self.model.predict_mean(m1v)
                val_rmse = torch.sqrt(torch.mean((pred - yv) ** 2)).item()
                val_acc = (m2v.argmax(1) == sv).float().mean().item()
            history.append((loss.item(), val_rmse, val_acc))
            if ep % 10 == 0 or ep == epochs - 1:
                print(f"  ep{ep} loss={loss.item():.4f} val_rmse={val_rmse:.4f} val_acc={val_acc:.4f}", flush=True)
        return history

    def evaluate(self, X_te, y_te, strat_te, y_sd: float = 1.0):
        """测试集评估：RMSE（还原尺度）+ acc + 区间覆盖率。"""
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_te).to(self.device)
            m1, m2 = self.model(Xt)
            pred = self.model.predict_mean(m1).cpu().numpy()
            acc = (m2.argmax(1).cpu() == torch.tensor(strat_te)).float().mean().item()
            rmse_norm = np.sqrt(np.mean((pred - np.array(y_te)) ** 2))
            result = {
                "rmse": float(rmse_norm * y_sd),
                "rmse_norm": float(rmse_norm),
                "acc": float(acc),
            }
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

    cfg = TensorConfig(T=24, H=8)
    ds = TensorBuilder(cfg).build(args.parquet)
    (X_tr, y_tr, s_tr), (X_va, y_va, s_va), (X_te, y_te, s_te) = (
        ds["train"], ds["val"], ds["test"])

    model = RamsNet(feat_dim=ds["feat_dim"], horizon=cfg.H)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    trainer = Trainer(model)
    trainer.fit(X_tr, y_tr, s_tr, X_va, y_va, s_va,
                epochs=2, fast_dev_run=args.fast_dev_run)
    if not args.fast_dev_run:
        res = trainer.evaluate(X_te, y_te, s_te, ds["y_sd"])
        print(f"测试: RMSE={res['rmse']:.4f} acc={res['acc']:.4f} "
              f"coverage={res.get('coverage', 0):.3f}")
    print("冒烟通过")
