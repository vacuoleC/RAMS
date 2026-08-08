# -*- coding: utf-8 -*-
"""P4+P5：多任务 loss 加权策略 + 分位数输出验证

基于 P3 的最佳架构（GRU 共享 backbone + 双头）细化：

P4  loss 加权策略网格（回应第一轮 E2 的意外发现）：
  - w0: 未归一化（回归主导，M1 loss 大）
  - w1: 简单归一化（两个 loss 同量纲）
  - w2: 分类优先（给 M2 更高权重）—— 因为分层是"核心驱动"

P5  分位数输出（10/50/90）在时序预测下的表现：
  - 分位数损失训练，中位数预测 RMSE vs 普通回归
  - 区间覆盖率（真实值是否落在 [p10, p90]）
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
temp_cols = [c for c in df.columns if c.startswith("temp_")]
meteo_cols = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
target_col = "conc_0.5"
T, H = 24, 8

# ========== 数据准备（同 P3） ==========
X = df[temp_cols + meteo_cols].values.astype(np.float32)
x_mu, x_sd = X.mean(0), X.std(0)+1e-8
X = (X - x_mu) / x_sd
y = df[target_col].values.astype(np.float32)
y_mu, y_sd = y.mean(), y.std() + 1e-8
y_n = (y - y_mu) / y_sd
strat_label = df["strat_binary"].values.astype(np.int64)

n = len(df)
Xw, yw, strat_w = [], [], []
for i in range(n - T - H):
    Xw.append(X[i:i+T]); yw.append(y_n[i+T:i+T+H])
    strat_w.append(strat_label[i+T-1])
Xw = np.stack(Xw).astype(np.float32); yw = np.stack(yw).astype(np.float32)
strat_w = np.array(strat_w)
nw = len(Xw); nt, nv = int(nw*0.7), int(nw*0.15)
(Xtr, ytr), (Xva, yva), (Xte, yte) = (Xw[:nt], yw[:nt]), (Xw[nt:nt+nv], yw[nt:nt+nv]), (Xw[nt+nv:], yw[nt+nv:])
str_tr, str_va, str_te = strat_w[:nt], strat_w[nt:nt+nv], strat_w[nt+nv:]
print(f"数据: {Xw.shape}, train {len(Xtr)}", flush=True)

# ========== 模型 ==========
class MultiTaskGRU(nn.Module):
    def __init__(self, in_dim, hidden=64, n_out=H, quantile=False):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.head_m1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                     nn.Linear(hidden, n_out*3 if quantile else n_out))
        self.head_m2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
        self.quantile = quantile
    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1]
        return self.head_m1(h), self.head_m2(h)

def train_mt(Xtr, ytr, str_tr, Xva, yva, str_va, w_m1=1.0, w_m2=1.0,
             quantile=False, epochs=30, bs=128):
    model = MultiTaskGRU(Xtr.shape[2], quantile=quantile).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_m1 = nn.MSELoss()
    if quantile:
        qs = torch.tensor([0.1, 0.5, 0.9], device=DEVICE)
    loss_m2 = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr), torch.tensor(str_tr)),
                    batch_size=bs, shuffle=False)
    for ep in range(epochs):
        model.train()
        for xb, yb, sb in dl:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            opt.zero_grad()
            rp, cp = model(xb)
            if quantile:
                # 分位数损失（三通道）
                yq = yb.unsqueeze(1)
                n_out = H
                losses = []
                for i, q in enumerate(qs):
                    qhat = rp[:, i*n_out:(i+1)*n_out]
                    e = yq - qhat
                    losses.append(torch.mean(torch.maximum(q*e, (q-1)*e)))
                l1 = torch.stack(losses).mean()
            else:
                l1 = loss_m1(rp, yb)
            l2 = loss_m2(cp, sb)
            loss = w_m1 * l1 + w_m2 * l2
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        rp, cp = model(torch.tensor(Xva).to(DEVICE))
    if quantile:
        pred = rp[:, H:2*H]  # 中位数通道
    else:
        pred = rp
    val_rmse = torch.sqrt(torch.mean((pred.cpu() - torch.tensor(yva))**2)).item()
    val_acc = (cp.argmax(1).cpu() == torch.tensor(str_va)).float().mean().item()
    return val_rmse, val_acc, model

def eval_test(model, Xte, yte, str_te, quantile=False):
    model.eval()
    with torch.no_grad():
        rp, cp = model(torch.tensor(Xte).to(DEVICE))
    if quantile:
        pred = rp[:, H:2*H].cpu().numpy()
        p10 = rp[:, :H].cpu().numpy(); p90 = rp[:, 2*H:].cpu().numpy()
    else:
        pred = rp.cpu().numpy()
    rmse = np.sqrt(np.mean((pred - yte)**2)) * y_sd
    acc = (cp.argmax(1).cpu() == torch.tensor(str_te)).float().mean().item()
    if quantile:
        # 覆盖率：真实值在 [p10, p90] 的比例（还原尺度前比较）
        cover = np.mean((yte >= p10) & (yte <= p90))
        # 区间宽度（原始尺度）
        width = np.mean((p90 - p10)) * y_sd
        return rmse, acc, cover, width
    return rmse, acc

print("\n===== P4: loss 加权策略 =====", flush=True)
results = {}
# w0 未归一化（回归主导）
vr, va, m = train_mt(Xtr, ytr, str_tr, Xva, yva, str_va, w_m1=1.0, w_m2=1.0)
te_rmse, te_acc = eval_test(m, Xte, yte, str_te)
results["w0_unnorm"] = (te_rmse, te_acc)
print(f"[w0 未归一化]     test_rmse={te_rmse:.4f} test_acc={te_acc:.4f}", flush=True)

# w1 归一化（两个 loss 同量纲——CE≈0.7 vs MSE≈0.1，故给回归降权）
vr, va, m = train_mt(Xtr, ytr, str_tr, Xva, yva, str_va, w_m1=0.15, w_m2=1.0)
te_rmse, te_acc = eval_test(m, Xte, yte, str_te)
results["w1_norm"] = (te_rmse, te_acc)
print(f"[w1 归一化]       test_rmse={te_rmse:.4f} test_acc={te_acc:.4f}", flush=True)

# w2 分类优先（M2 权重更高）
vr, va, m = train_mt(Xtr, ytr, str_tr, Xva, yva, str_va, w_m1=1.0, w_m2=3.0)
te_rmse, te_acc = eval_test(m, Xte, yte, str_te)
results["w2_cls_pri"] = (te_rmse, te_acc)
print(f"[w2 分类优先]     test_rmse={te_rmse:.4f} test_acc={te_acc:.4f}", flush=True)

print("\n===== P5: 分位数输出 =====", flush=True)
vr, va, mq = train_mt(Xtr, ytr, str_tr, Xva, yva, str_va, w_m1=1.0, w_m2=1.0, quantile=True)
te_rmse, te_acc, cover, width = eval_test(mq, Xte, yte, str_te, quantile=True)
results["q_quantile"] = (te_rmse, te_acc, cover, width)
print(f"[分位数 中位数预测] test_rmse={te_rmse:.4f} test_acc={te_acc:.4f}", flush=True)
print(f"[分位数 区间覆盖率] p10-p90 覆盖率={cover:.3f} 平均区间宽={width:.3f}", flush=True)

print("\n===== P4+P5 汇总 =====", flush=True)
print(f"w0 未归一化:   RMSE={results['w0_unnorm'][0]:.4f} acc={results['w0_unnorm'][1]:.4f}", flush=True)
print(f"w1 归一化:     RMSE={results['w1_norm'][0]:.4f} acc={results['w1_norm'][1]:.4f}", flush=True)
print(f"w2 分类优先:   RMSE={results['w2_cls_pri'][0]:.4f} acc={results['w2_cls_pri'][1]:.4f}", flush=True)
print(f"分位数(中位):  RMSE={results['q_quantile'][0]:.4f} acc={results['q_quantile'][1]:.4f}", flush=True)
print(f"  覆盖率={results['q_quantile'][2]:.3f} 区间宽={results['q_quantile'][3]:.3f}", flush=True)
