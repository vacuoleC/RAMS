"""RAMS training 包：训练编排（mdl-model-integrate）。

RAMS 0.2.0 正式训练（冻结设计）：
  - `Trainer`：轻量训练器——分位数损失 + 多任务加权（w=1/3/2）+ 两阶段（ts_freeze）。
  - `MultiTaskLoss` / `QuantileLoss`：多任务损失。
  - `crps_cdf_pline` / `crps_quantiles`：CRPS 评估（T4 协议）。
  - `make_m4_labels`：M4 预警等级标签（日级协议，peak_quantile / bloom）。
"""

from rams.training.trainer import (
    W_M1,
    W_M2,
    W_M4,
    MultiTaskLoss,
    QuantileLoss,
    Trainer,
    crps_cdf_pline,
    crps_quantiles,
    make_m4_labels,
)

__all__ = [
    "Trainer",
    "MultiTaskLoss",
    "QuantileLoss",
    "crps_cdf_pline",
    "crps_quantiles",
    "make_m4_labels",
    # 多任务权重（w=1/3/2）
    "W_M1",
    "W_M2",
    "W_M4",
]
