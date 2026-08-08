# -*- coding: utf-8 -*-
"""RAMS 模型：共享 GRU Backbone + M1 回归头 + M2 分层头

架构依据探索测试结论（3-seed 验证）：
  - 20 层输入（全剖面水温 + 气象）比单层降误差 32%
  - 共享 backbone + 多任务（M2 分层头）比单任务再降 8-15%
  - M2 分层任务权重略高（w2 分类优先）最优
  - 分位数输出（10/50/90）校准良好，RMSE 不损失

结构：
  Input (B, T, D×C) → GRU → shared hidden → [M1 头] → (B, H) 或 (B, 3H)
                                        → [M2 头] → (B, 2)  分层分类
"""
from __future__ import annotations

import torch
import torch.nn as nn

QUANTILES = (0.1, 0.5, 0.9)


class SharedGRU(nn.Module):
    """共享 GRU backbone：输入时序张量，输出末时刻隐状态。"""

    def __init__(self, feat_dim: int, hidden: int = 64, n_layers: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.gru = nn.GRU(feat_dim, hidden, n_layers, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feat_dim)
        out, _ = self.gru(x)
        return out[:, -1]  # (B, hidden) 末时刻隐状态


class M1Head(nn.Module):
    """M1 藻类浓度预测头（分位数输出可选）。

    quantile=True 时输出 (B, 3×H)，三通道分别为 p10/p50/p90。
    """

    def __init__(self, hidden: int, n_out: int, quantile: bool = True):
        super().__init__()
        self.n_out = n_out
        self.quantile = quantile
        out_dim = n_out * 3 if quantile else n_out
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class M2Head(nn.Module):
    """M2 热分层识别头（二分类：是否分层）。"""

    def __init__(self, hidden: int, n_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class M4Head(nn.Module):
    """M4 藻华预警分级头（多分类：安全/注意/警告/危险）。

    用共享表征直接预测预警等级（文献共识：直接预测等级优于回归转阈值）。
    """

    def __init__(self, hidden: int, n_levels: int = 4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_levels),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class RamsNet(nn.Module):
    """共享 backbone + M1/M2/M4 三头（多任务架构）。

    forward 返回 (m1_out, m2_out, m4_out)：
      - m1_out: (B, 3H) 分位数或 (B, H)
      - m2_out: (B, n_classes) 分层分类
      - m4_out: (B, n_levels) 预警分级（可选）
    """

    def __init__(self, feat_dim: int, horizon: int, hidden: int = 64,
                 n_layers: int = 1, quantile: bool = True, n_classes: int = 2,
                 n_levels: int = 4, use_m4: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden, n_layers, dropout)
        self.m1 = M1Head(hidden, horizon, quantile)
        self.m2 = M2Head(hidden, n_classes)
        self.m4 = M4Head(hidden, n_levels) if use_m4 else None
        self.horizon = horizon
        self.quantile = quantile
        self.use_m4 = use_m4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        h = self.backbone(x)
        m4 = self.m4(h) if self.m4 is not None else None
        return self.m1(h), self.m2(h), m4

    def predict_mean(self, m1_out: torch.Tensor) -> torch.Tensor:
        """从分位数输出取中位数预测。"""
        H = self.horizon
        return m1_out[:, H:2 * H] if self.quantile else m1_out

    def predict_interval(self, m1_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """从分位数输出取 p10/p90 区间。"""
        H = self.horizon
        if not self.quantile:
            raise ValueError("quantile=False 时无区间输出")
        return m1_out[:, :H], m1_out[:, 2 * H:]


def count_parameters(model: nn.Module) -> int:
    """打印参数量（架构约束验证）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 冒烟：小张量前向
    torch.manual_seed(0)
    net = RamsNet(feat_dim=26, horizon=8, quantile=True)
    x = torch.randn(4, 24, 26)
    m1, m2 = net(x)
    print(f"输入: {tuple(x.shape)}")
    print(f"M1 输出: {tuple(m1.shape)} (3×H 分位数)")
    print(f"M2 输出: {tuple(m2.shape)} (二分类)")
    print(f"参数量: {count_parameters(net):,} (≈1.9M 的 {count_parameters(net)/1.9e6*100:.0f}%)")
    assert m1.shape == (4, 24), f"M1 形状错误: {m1.shape}"
    assert m2.shape == (4, 2), f"M2 形状错误: {m2.shape}"
    print("冒烟通过")
