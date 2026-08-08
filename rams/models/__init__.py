"""模型子包：Backbone + 五个任务头。

- `backbone.py`：RAMS-Net（ChannelEmbedding → TemporalBlock → VerticalBlock → FusionBlock），≈1.9M 参数
- `heads/m1.py`：M1 藻类预测（回归）
- `heads/m2.py`：M2 热分层识别（分类）
- `heads/m3.py`：M3 点位优化（GAT + 贪心）
- `heads/m4.py`：M4 藻华预警（分位数 + 有序回归）
- `heads/m5.py`：M5 机理时滞（PCMCI+ 调用封装，无需训练）
- `lit_module.py`：LightningModule 封装 + 多任务损失
"""

__all__ = []
