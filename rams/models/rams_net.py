"""RAMS 0.2.0 正式模型 —— 共享 GRU Backbone + M1 增量分位数头 + M2 分层头 + M4 藻华预警头

模块目标（frozen design `modules/mdl-model-integrate/module_design.yaml`）：
  - M1 增量Δ + q9 分位数输出（GRU backbone，hidden 可配）
  - M2/M4 多任务头（M4 用藻华状态标签）
  - 两阶段训练（ts_freeze，见 rams/training/trainer.py）

设计来源（探索实证，`exp/model_enhancement/`）：
  - 框架比较：GRU backbone 最优，不换。
  - 增量目标（B1）：M1 预测 Δ=conc_{t+h}−conc_t，评估时还原 conc_t+Δ。
  - q9 分位数（G）：M1 输出 9 个固定分位数 [0.05,0.10,0.20,0.35,0.50,0.65,0.80,0.90,0.95]，
    比 3 分位 CRPS +2%（点数效应，成本 ≈ 一行）。
  - 多任务（A）：M1+M2+M4，w=1/3/2；M2 保区间校准。
  - 两阶段（K）：Stage1 单任务训 M1 → Stage2 冻结 backbone 微调多头（精度最优）。

数据保密红线：本模块只输出形状/统计量，不打印任何原始数据行。

接口（I/O 形状）：
  - `SharedGRU(x)`：x (B, T, F) → h (B, hidden)
  - `RamsNet.forward(x)` → (m1, m2, m4)：
      m1 (B, n_q*H) 分位数输出（默认 n_q=9）或 (B, H)（quantile=False）
      m2 (B, n_classes)，m4 (B, n_levels) 或 None（use_m4=False）
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# q9 固定分位数（G 探索，`g_distribution/results.md`）：9 结分段线性 CDF 求 CRPS
QUANTILE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95)
# 3 分位（0.1.0 兼容 / 旧测试契约，`rams_net` 导出符号）
QUANTILES: tuple[float, ...] = (0.1, 0.5, 0.9)
# M4 预警等级数（安全/注意/警告/危险）
WARN_LEVELS: int = 4


class SharedGRU(nn.Module):
    """共享 GRU backbone：输入时序张量，输出末时刻隐状态。

    Args:
        feat_dim: 展平特征数 F（日级 X_flat (B, T, F)）。
        hidden: 隐状态维（64）。
        n_layers: GRU 层数。
        dropout: dropout 率。

    I/O:
        forward(x): x (B, T, F) → (B, hidden)
    """

    def __init__(self, feat_dim: int, hidden: int = 64, n_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.gru = nn.GRU(feat_dim, hidden, n_layers, batch_first=True, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, feat_dim)
        out, _ = self.gru(x)
        return out[:, -1]  # (B, hidden) 末时刻隐状态


class M1Head(nn.Module):
    """M1 浓度预测头（增量分位数输出）。

    默认 n_quantiles=9（q9，G 探索采纳）；n_quantiles=3 时与 0.1.0 契约一致。

    I/O:
        forward(h): h (B, hidden) → (B, n_quantiles*H)
    """

    def __init__(
        self, hidden: int, n_out: int, n_quantiles: int | None = None, quantile: bool = True
    ):
        super().__init__()
        self.n_out = n_out
        self.n_quantiles = (n_quantiles or len(QUANTILES)) if quantile else 1
        out_dim = n_out * self.n_quantiles
        # 分位数水平（3=0.1.0 兼容；9=q9；其他 = 线性内插 n 个均匀水平）
        if self.n_quantiles == 3:
            self.quantile_levels = list(QUANTILES)
        elif self.n_quantiles == len(QUANTILE_LEVELS):
            self.quantile_levels = list(QUANTILE_LEVELS)
        else:
            self.quantile_levels = [
                round((i + 1) / (self.n_quantiles + 1), 4) for i in range(self.n_quantiles)
            ]
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class M2Head(nn.Module):
    """M2 热分层识别头（二分类：是否分层）。

    I/O:
        forward(h): h (B, hidden) → (B, n_classes)
    """

    def __init__(self, hidden: int, n_classes: int = 2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class M4Head(nn.Module):
    """M4 藻华预警分级头（多分类：安全/注意/警告/危险）。

    用共享表征直接预测预警等级；M4 标签由训练段阈值生成（防泄漏，见 data 管线）。

    I/O:
        forward(h): h (B, hidden) → (B, n_levels)
    """

    def __init__(self, hidden: int, n_levels: int = WARN_LEVELS):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_levels),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)


class RamsNet(nn.Module):
    """共享 backbone + M1/M2/M4 三头（多任务架构）。

    Args:
        feat_dim: 输入特征数 F（X_flat 的最后一维）。
        horizon: M1 预测视界 H。
        hidden: GRU 隐状态维。
        n_quantiles: M1 分位数个数（默认 9=q9；3=0.1.0 兼容）。
        quantile: False 时 M1 输出 (B, H) 点预测（无区间）。
        n_classes: M2 类别数（默认 2 分层）。
        n_levels: M4 预警等级数（默认 4）。
        use_m4: False 时不建 M4 头（m4 返回 None）。
        n_layers / dropout: GRU 参数。

    I/O:
        forward(x): x (B, T, F) → (m1, m2, m4)：
            m1 (B, n_quantiles*H) 分位数输出（通道顺序 = 分位数结 × 视界）
            m2 (B, n_classes)
            m4 (B, n_levels) 或 None
    """

    def __init__(
        self,
        feat_dim: int,
        horizon: int,
        hidden: int = 64,
        n_layers: int = 1,
        quantile: bool = True,
        n_quantiles: int | None = None,
        n_classes: int = 2,
        n_levels: int = WARN_LEVELS,
        use_m4: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = SharedGRU(feat_dim, hidden, n_layers, dropout)
        self.m1 = M1Head(hidden, horizon, n_quantiles, quantile)
        self.m2 = M2Head(hidden, n_classes)
        self.m4 = M4Head(hidden, n_levels) if use_m4 else None
        self.horizon = horizon
        self.quantile = quantile
        self.n_quantiles = self.m1.n_quantiles
        self.use_m4 = use_m4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        h = self.backbone(x)
        m4 = self.m4(h) if self.m4 is not None else None
        return self.m1(h), self.m2(h), m4

    def predict_mean(self, m1_out: torch.Tensor) -> torch.Tensor:
        """从分位数输出取中位数预测。

        Args:
            m1_out: (B, n_quantiles*H)

        Returns:
            (B, H) p50（分位数为奇数时取中间结）。
        """
        H = self.horizon
        if not self.quantile:
            return m1_out
        if self.n_quantiles == 3:
            return m1_out[:, H : 2 * H]
        mid = self.n_quantiles // 2
        return m1_out[:, mid * H : (mid + 1) * H]

    def predict_interval(self, m1_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """从分位数输出取 p10/p90 区间。

        Args:
            m1_out: (B, n_quantiles*H)

        Returns:
            (p10, p90)，各 (B, H)。默认 q9 用结 0.10 / 0.90；3 分位同 0.1.0。
        """
        H = self.horizon
        if not self.quantile:
            raise ValueError("quantile=False 时无区间输出")
        i10 = int(np.argmin(np.abs(np.array(self.m1.quantile_levels) - 0.10)))
        i90 = int(np.argmin(np.abs(np.array(self.m1.quantile_levels) - 0.90)))
        return m1_out[:, i10 * H : (i10 + 1) * H], m1_out[:, i90 * H : (i90 + 1) * H]

    def quantile_matrix(self, m1_out: torch.Tensor) -> torch.Tensor:
        """把 M1 分位数输出重整为 (B, n_quantiles, H)（分位数 × 视界）。

        供训练/评估统一取各结预测；结序与 `QUANTILE_LEVELS`（或 3 分位）一致。
        """
        H = self.horizon
        n_q = self.n_quantiles
        return m1_out.reshape(-1, n_q, H)


def count_parameters(model: nn.Module) -> int:
    """可训练参数量（架构约束验证）。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 冒烟：小张量前向（形状验证，不涉数据）
    torch.manual_seed(0)
    net = RamsNet(feat_dim=29, horizon=7, quantile=True, n_quantiles=9)
    x = torch.randn(4, 30, 29)
    m1, m2, m4 = net(x)
    print(f"输入: {tuple(x.shape)}")
    print(f"M1 输出: {tuple(m1.shape)} (q9 分位数, 结×视界)")
    print(f"M2 输出: {tuple(m2.shape)} (分层分类)")
    print(f"M4 输出: {tuple(m4.shape)} (预警分级)")
    print(f"参数量: {count_parameters(net):,}")
    assert m1.shape == (4, 9 * 7), f"M1 形状错误: {m1.shape}"
    assert m2.shape == (4, 2), f"M2 形状错误: {m2.shape}"
    assert m4 is not None, "M4 头不存在"
    assert m4.shape == (4, 4), f"M4 形状错误: {m4.shape}"
    qm = net.quantile_matrix(m1)
    assert qm.shape == (4, 9, 7), f"quantile_matrix 形状错误: {qm.shape}"
    p50 = net.predict_mean(m1)
    assert p50.shape == (4, 7), f"p50 形状错误: {p50.shape}"
    p10, p90 = net.predict_interval(m1)
    assert p10.shape == (4, 7), f"p10 形状错误: {p10.shape}"
    assert p90.shape == (4, 7), f"p90 形状错误: {p90.shape}"
    # 旧 3 分位契约（test_smoke.py 兼容）
    net3 = RamsNet(feat_dim=26, horizon=8, quantile=True, n_quantiles=3)
    m1_3, _, _ = net3(torch.randn(4, 24, 26))
    assert m1_3.shape == (4, 24), f"3分位 M1 形状错误: {m1_3.shape}"
    print("冒烟通过")
