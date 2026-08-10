"""RAMS 0.2.0 正式代码包（重建）。

模块图（frozen design）：
  - rams.data：日级张量 + 藻华标签（mdl-data-scale）
  - rams.models：共享 GRU + M1/M2/M4 多任务头（mdl-model-integrate）
  - rams.training：训练编排（分位数损失 + 多任务 + 两阶段 ts_freeze）
"""

__version__ = "0.1.0"
