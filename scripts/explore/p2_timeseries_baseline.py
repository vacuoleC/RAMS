# -*- coding: utf-8 -*-
"""P2 时序基线：LSTM/GRU 预测未来 24h（8 步 × 3h）

对照 P1 宽表，构建滑动窗口训练：
  - 输入: 过去 T 个时刻的 (20 层水温 + 气象)
  - 输出: 未来 8 步 (24h) 的 0.5m 层总浓度（M1 简化：单层预测）
关键验证：时序任务下基线是否"不饱和"（预测未来比拟合当期难，架构差异显现）
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}", flush=True)

# ========== 数据 ==========
df = pd.read_parquet("/data/RAMS/tensor_ready.parquet")
print(f"数据: {df.shape}", flush=True)

# 输入特征：20 层水温 + 6 气象 = 26 维
feat_cols = [c for c in df.columns if c.startswith("temp_")] + \
            ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
target_col = "conc_0.5"  # M1 简化：预测 0.5m 层总浓度未来

X = df[feat_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

# 归一化
x_mu, x_sd = X.mean(0), X.std(0) + 1e-8
y_mu, y_sd = y.mean(), y.std() + 1e-8
X = (X - x_mu) / x_sd
y = (y - y_mu) / y_sd

# ========== 滑动窗口切分 ==========
T = 24        # 回看窗口：24 时刻 = 3 天
H = 8         # 预测未来 24h (8×3h)
n = len(X)
print(f"回看 T={T} (3天), 预测 H={H} (24h)", flush=True)

# 构建窗口样本 (X_seq, y_seq)
X_windows, y_windows = [], []
for i in range(n - T - H):
    X_windows.append(X[i:i+T])
    y_windows.append(y[i+T:i+T+H])
X_w = np.stack(X_windows).astype(np.float32)
y_w = np.stack(y_windows).astype(np.float32)
print(f"窗口样本: X={X_w.shape} y={y_w.shape}", flush=True)

# 时序切分（按时间顺序，无泄漏：训练段不跨验证段）
n_w = len(X_w)
n_tr, n_va = int(n_w*0.7), int(n_w*0.15)
X_tr, X_va, X_te = X_w[:n_tr], X_w[n_tr:n_tr+n_va], X_w[n_tr+n_va:]
y_tr, y_va, y_te = y_w[:n_tr], y_w[n_tr:n_tr+n_va], y_w[n_tr+n_va:]
print(f"切分: train {X_tr.shape} val {X_va.shape} test {X_te.shape}", flush=True)

# ========== 模型 ==========
class SeqModel(nn.Module):
    def __init__(self, in_dim, hidden, n_out, cell="lstm", n_layers=1):
        super().__init__()
        self.cell_type = cell.lower()
        rnn_cls = nn.LSTM if self.cell_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(in_dim, hidden, n_layers, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])  # 取最后时刻隐状态

def make_loader(X, y, bs):
    t = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(t, batch_size=bs, shuffle=False)

def train(model, Xtr, ytr, Xva, yva, epochs=30, bs=128, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    dl = make_loader(Xtr, ytr, bs)
    Xvt = torch.tensor(Xva); yvt = torch.tensor(yva)
    best = 1e9
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward(); opt.step()
        if ep % 10 == 0 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                vp = model(Xvt.to(DEVICE)).cpu()
                vr = torch.sqrt(torch.mean((vp - yvt)**2)).item()
            if vr < best: best = vr
            print(f"    ep{ep} loss={loss.item():.4f} val_rmse={vr:.4f}", flush=True)
    return best

def evaluate(model, Xte, yte):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte).to(DEVICE)).cpu()
    # 各步 RMSE（还原原始尺度）
    rmse_per_step = torch.sqrt(torch.mean((pred - torch.tensor(yte))**2, dim=0)).numpy() * y_sd
    # 整体（未来 24h 平均）
    overall = np.sqrt(np.mean((pred.numpy() - yte)**2)) * y_sd
    # 持久化基线（用最后观测值预测未来 24h 不变）
    persist = np.sqrt(np.mean((yte - Xte[:, -1, 0][:, None])**2)) * y_sd  # temp_0.5 是首特征
    return rmse_per_step, overall, persist

print("\n===== P2 时序基线 =====", flush=True)
feat_dim = X_w.shape[2]

# LSTM
print("\n[P2-LSTM] hidden=64, 1层", flush=True)
m = SeqModel(feat_dim, 64, H, "lstm").to(DEVICE)
train(m, X_tr, y_tr, X_va, y_va)
rps, overall, persist = evaluate(m, X_te, y_te)
print(f"  LSTM 测试整体RMSE={overall:.4f} 持久化基线={persist:.4f}", flush=True)
print(f"  分步RMSE: {np.round(rps, 3)}", flush=True)

# GRU
print("\n[P2-GRU] hidden=64, 1层", flush=True)
m2 = SeqModel(feat_dim, 64, H, "gru").to(DEVICE)
train(m2, X_tr, y_tr, X_va, y_va)
rps2, overall2, persist2 = evaluate(m2, X_te, y_te)
print(f"  GRU 测试整体RMSE={overall2:.4f} 持久化基线={persist2:.4f}", flush=True)
print(f"  分步RMSE: {np.round(rps2, 3)}", flush=True)

# 简单 MLP（当期基线对照，来自第一轮：拟合当期 RMSE≈0.19）
print("\n===== 结论 =====", flush=True)
print(f"LSTM(预测未来24h): {overall:.4f}", flush=True)
print(f"GRU(预测未来24h):  {overall2:.4f}", flush=True)
print(f"持久化基线(预测未来24h): {persist:.4f}", flush=True)
print(f"第一轮 MLP(拟合当期): 0.1887", flush=True)
print("→ 若时序 RMSE 明显 > 当期 RMSE，说明预测未来确实更难，架构差异将显现", flush=True)
