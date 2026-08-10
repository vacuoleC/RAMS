# -*- coding: utf-8 -*-
"""T2 补充：归一化口径下模型 vs 平凡基线对比（评估口径敏感性的解释）"""
import sys
import numpy as np

sys.path.insert(0, ".")
from rams.data.tensor_builder import TensorBuilder, TensorConfig

builder = TensorBuilder(TensorConfig(T=24, H=8))
wide = builder._load_wide("data/processed/standard.parquet")
c = wide["conc_0.5"].values.astype(np.float64)
n = len(c)
n_tr, n_va = int(n * 0.7), int(n * 0.15)

T, H = 24, 8
Xw, yw = [], []
for i in range(n - T - H):
    Xw.append(c[i:i + T])
    yw.append(c[i + T:i + T + H])
Xw = np.stack(Xw)
yw = np.stack(yw)
N = len(Xw)
test = slice(n_tr + n_va, N)
y_te = yw[test]
x_last = Xw[test, -1]
persist = np.repeat(x_last[:, None], H, axis=1)

y_sd = 13.886
rmse_persist = np.sqrt(np.mean((y_te - persist) ** 2))
rmse_norm_persist = rmse_persist / y_sd
model_rmse = 4.525  # 本地 CPU GRU 单次
model_rmse_norm = model_rmse / y_sd

print("持久化 RMSE(原始): %.3f   归一化: %.4f" % (rmse_persist, rmse_norm_persist))
print("本地GRU RMSE(原始): %.3f   归一化: %.4f" % (model_rmse, model_rmse_norm))
print(">>> 归一化口径下 模型/持久化 = %.2fx（模型仍差）" % (model_rmse_norm / rmse_norm_persist))
print(">>> 解释: 模型在 2021-2024 高波动段训练、测试 2025 低波动段，追不上快速季节变化")
print(">>> 这暴露 RMSE+固定时序切分 协议的最严场景；EFI 挑战用 CRPS 且滚动窗口，更公平")
