"""RAMS models 包：共享 GRU backbone + M1/M2/M4 多任务头（mdl-model-integrate）。

RAMS 0.2.0 正式模型（冻结设计）：
  - `SharedGRU`：GRU backbone（hidden 可配）。
  - `RamsNet`：多任务网，M1 增量Δ分位数头（q9 默认）+ M2 分层头 + M4 藻华预警头。
  - `QUANTILE_LEVELS` / `QUANTILES`：q9 / 3 分位数水平。
"""

from rams.models.rams_net import (
    QUANTILE_LEVELS,
    QUANTILES,
    WARN_LEVELS,
    M1Head,
    M2Head,
    M4Head,
    RamsNet,
    SharedGRU,
    count_parameters,
)

__all__ = [
    "RamsNet",
    "SharedGRU",
    "M1Head",
    "M2Head",
    "M4Head",
    "count_parameters",
    # 常量
    "QUANTILE_LEVELS",
    "QUANTILES",
    "WARN_LEVELS",
]
