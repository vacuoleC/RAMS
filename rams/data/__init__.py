"""RAMS data 包：数据读取、张量构建、藻华状态标签。

RAMS 0.2.0 数据管线（mdl-data-scale）：
  - DailyConfig / DailyTensorBuilder：standard.parquet → 日级张量 (B, T=30, D, C)
  - BloomLabeler：藻华状态/事件标签（N 定义，顶层带 + 多层联动 + 连续 ≥2 天）
  - TensorConfig / TensorBuilder：3h 原网格（0.1.0 兼容，供既有探索脚本/测试）
  - make_rolling_anchors / build_daily_dataset：滚动窗口与便捷入口

数据保密：本包只读 `data/processed/standard.parquet`，绝不写入或修改数据文件。
"""
from rams.data.tensor_builder import (
    BLOOM_GAP_DAYS,
    BLOOM_MIN_DAYS,
    CONC_COLS,
    DAILY_GRID,
    DEPTHS,
    GRID,
    LINK_BAND_DEPTHS,
    M3_RECOMMENDED_DEPTHS,
    METEO_COLS,
    STRAT_COLS,
    TARGET_DEPTH,
    TEMP_COLS,
    TOP_BAND_DEPTHS,
    BloomLabeler,
    DailyConfig,
    DailyDataset,
    DailyTensorBuilder,
    TensorBuilder,
    TensorConfig,
    build_daily_dataset,
    make_rolling_anchors,
)

__all__ = [
    "BloomLabeler",
    "DailyConfig",
    "DailyDataset",
    "DailyTensorBuilder",
    "TensorConfig",
    "TensorBuilder",
    "build_daily_dataset",
    "make_rolling_anchors",
    # 常量
    "BLOOM_GAP_DAYS",
    "BLOOM_MIN_DAYS",
    "CONC_COLS",
    "DAILY_GRID",
    "DEPTHS",
    "GRID",
    "LINK_BAND_DEPTHS",
    "M3_RECOMMENDED_DEPTHS",
    "METEO_COLS",
    "STRAT_COLS",
    "TEMP_COLS",
    "TOP_BAND_DEPTHS",
    "TARGET_DEPTH",
]
