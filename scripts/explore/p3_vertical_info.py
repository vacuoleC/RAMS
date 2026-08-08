# -*- coding: utf-8 -*-
"""P3 垂直信息价值验证

假说 H2：20 层垂直信息 > 单层 0.5m
对比：
  A. 单层输入（只用 0.5m 水温+气象）→ 预测 0.5m 浓度
  B. 20 层输入（全部水温剖面+气象）→ 预测 0.5m 浓度
  C. 20 层输入 + 分层状态特征 → 预测 0.5m 浓度（旁路用法）
  D. 20 层输入 + 分层状态标签辅助（多任务，M2 分支）→ 预测 0.5m 浓度

都用 GRU（P2 里表现更好），同 seed，报告相对增益。
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

df = pd.read_parquet("/data/RAMS/tensor_ready.parquet")
print(f"数据: {df.shape}", flush=True)

temp_cols = [c for c in df.columns if c.startswith("temp_")]
meteo_cols = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
target_col = "conc_0.5"

T, H = 24, 8
feat_meta = {
    "A_single": temp_cols[:1] + meteo_cols,          # 单层：0.5m 水温 + 气象
    "B_full": temp_cols + meteo_cols,                 # 20 层 + 气象
    "C_full_strat_feat": temp_cols + meteo_cols + ["delta_T"],  # 20层+气象+分层状态特征
}

y = df[target_col].values.astype(np.float32)
y_mu, y_sd = y.mean(), y.std() + 1e-8
y_n = (y - y_mu) / y_sd

n = len(df)
def build(X):
    Xw, yw = [], []
    for i in range(n - T - H):
        Xw.append(X[i:i+T]); yw.append(y_n[i+T:i+T+H])
    return np.stack(Xw).astype(np.float32), np.stack(yw).astype(np.float32)

def split(Xw, yw):
    nw = len(Xw); nt, nv = int(nw*0.7), int(nw*0.15)
    return (Xw[:nt], yw[:nt]), (Xw[nt:nt+nv], yw[nt:nt+nv]), (Xw[nt+nv:], yw[nt+nv:])

class GRUModel(nn.Module):
    def __init__(self, in_dim, hidden=64, n_out=H):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1])

def train_eval(Xtr, ytr, Xva, yva, Xte, yte, epochs=30):
    model = GRUModel(Xtr.shape[2]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)), batch_size=128, shuffle=False)
    Xvt, yvt = torch.tensor(Xva), torch.tensor(yva)
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); out = model(xb); loss = loss_fn(out, yb)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte).to(DEVICE)).cpu().numpy()
    rmse = np.sqrt(np.mean((pred - yte)**2)) * y_sd
    return rmse, pred

results = {}
for name, cols in feat_meta.items():
    X = df[cols].values.astype(np.float32)
    x_mu, x_sd = X.mean(0), X.std(0)+1e-8
    X = (X - x_mu) / x_sd
    Xw, yw = build(X)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = split(Xw, yw)
    print(f"\n[P3-{name}] 特征维={X.shape[1]}", flush=True)
    rmse, _ = train_eval(Xtr, ytr, Xva, yva, Xte, yte)
    results[name] = rmse
    print(f"  测试 RMSE = {rmse:.4f}", flush=True)

# D: 多任务（GRU + M1回归头 + M2分层头）
class MultiTaskGRU(nn.Module):
    def __init__(self, in_dim, hidden=64, n_out=H):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.head_m1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.head_m2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1]
        return self.head_m1(h), self.head_m2(h)

strat_label = df["strat_binary"].values.astype(np.int64)
strat_n = (strat_label - strat_label.mean())  # 中心化
# 窗口目标：M2 用最后时刻的分层状态
X_full = df[temp_cols + meteo_cols].values.astype(np.float32)
x_mu, x_sd = X_full.mean(0), X_full.std(0)+1e-8
X_full = (X_full - x_mu) / x_sd
Xw, yw = build(X_full)
strat_w = np.array([strat_label[i+T-1] for i in range(n-T-H)])  # 窗口末时刻分层
nw = len(Xw); nt, nv = int(nw*0.7), int(nw*0.15)
(Xtr, ytr), (Xva, yva), (Xte, yte) = (Xw[:nt], yw[:nt]), (Xw[nt:nt+nv], yw[nt:nt+nv]), (Xw[nt+nv:], yw[nt+nv:])
strat_tr, strat_va, strat_te = strat_w[:nt], strat_w[nt:nt+nv], strat_w[nt+nv:]

model = MultiTaskGRU(Xtr.shape[2]).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_m1 = nn.MSELoss(); loss_m2 = nn.CrossEntropyLoss()
dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(strat_tr)), batch_size=128, shuffle=False)
for ep in range(30):
    model.train()
    for xb, yb, sb in dl:
        xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
        opt.zero_grad()
        rp, cp = model(xb)
        loss = loss_m1(rp, yb) + loss_m2(cp, sb)  # 未归一化（第一轮发现未归一化更好）
        loss.backward(); opt.step()
model.eval()
with torch.no_grad():
    rp, _ = model(torch.tensor(Xte).to(DEVICE)).cpu() if False else (lambda: model(torch.tensor(Xte).to(DEVICE)))(), None
    rp = model(torch.tensor(Xte).to(DEVICE))[0].cpu().numpy()
rmse_d = np.sqrt(np.mean((rp - yte)**2)) * y_sd
results["D_multitask"] = rmse_d
print(f"\n[P3-D_multitask] 测试 RMSE = {rmse_d:.4f}", flush=True)

print("\n===== P3 汇总 =====", flush=True)
base = results["B_full"]
for k, v in results.items():
    rel = (v - base) / base * 100
    print(f"  {k}: RMSE={v:.4f}  相对B={rel:+.1f}%", flush=True)
