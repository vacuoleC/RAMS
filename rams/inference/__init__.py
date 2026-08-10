"""推理子包：ONNX 导出 + INT8 量化 + FastAPI 服务。

- `onnx_export.py`：Backbone + 五个头各自导出 ONNX
- `quantize.py`：INT8 动态量化（目标 < 3MB）
- `serve.py`：FastAPI 推理服务（ONNX Runtime 后端）
- `predict.py`：单批次预测 CLI
"""

__all__ = []
