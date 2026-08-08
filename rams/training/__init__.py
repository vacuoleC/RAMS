"""训练编排子包：两阶段训练（联合预训练 → 冻结微调）+ MLflow 记录。

- `pretrain.py`：阶段一，M1+M2+M4 联合预训练 Backbone
- `finetune.py`：阶段二，冻结 Backbone 微调五个头
- `callbacks.py`：早停、检查点、fast_dev_run 开关
- `mlflow_utils.py`：本地 MLflow 追踪（**禁外发**）
"""

__all__ = []
