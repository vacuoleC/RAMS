# -*- coding: utf-8 -*-
"""探索性测试 · 验证架构提升点（在 sensecore H100 上跑）

验证四个说法：
  E1 分位数输出（10/50/90）是否可行、能否输出合理置信区间 —— 不确定性主线
  E2 多任务 loss 尺度归一化是否必要（回归 RMSE vs 分类 CE 联合训练）
  E3 M2 分层状态作为旁路喂给 M1 是否提升预测
  E4 滞后窗口特征（降水/风 1-3 天）是否提升预测

设计：用最小模型 + 少量数据快速验证"相对增益"，不做完整训练。
数据：/data/RAMS/algae_long.parquet + meteo.parquet
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import time

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}  torch: {torch.__version__}")

# ========== 数据准备 ==========
print("读取数据 ...", flush=True)
algae = pd.read_parquet("/data/RAMS/algae_long.parquet")
meteo = pd.read_parquet("/data/RAMS/meteo.parquet")

# 取一个深度层（0.5m），按时间排序
s = algae[algae["depth"] == 0.5].sort_values("timestamp").reset_index(drop=True)
print(f"0.5m 层: {s.shape} 时间 {s['timestamp'].min()} ~ {s['timestamp'].max()}")

# 用 timestamp 对齐气象（最近时刻，merge_asof —— 藻类带秒 vs 气象整 10 分钟）
meteo_sorted = meteo.sort_values("timestamp")
s = s.sort_values("timestamp").reset_index(drop=True)
s = pd.merge_asof(
    s, meteo_sorted[["timestamp", "wind_speed", "air_temp", "rainfall", "humidity"]],
    on="timestamp", direction="nearest", tolerance=pd.Timedelta("10min"))
print(f"合并气象后: {s.shape} 非空 wind_speed={s['wind_speed'].notna().sum()}", flush=True)

# 特征列（不用原始值，只用统计验证——这里用标准化后的）
feat_cols = ["water_temp", "total_conc", "cyano_conc", "green_conc",
             "wind_speed", "air_temp", "rainfall", "humidity"]
for c in feat_cols:
    if c in s.columns:
        s[c] = pd.to_numeric(s[c], errors="coerce")

# 归一化（用全量统计量，探索测试允许）
norm = {}
for c in feat_cols:
    mu, sd = s[c].mean(), s[c].std() + 1e-8
    norm[c] = (mu, sd)
    s[c] = (s[c] - mu) / sd

# 目标：total_conc（下一时刻预测）——构建 (X, y)
s = s.dropna(subset=feat_cols + ["total_conc"]).reset_index(drop=True)
data_X = s[feat_cols].values.astype(np.float32)
data_y = s["total_conc"].values.astype(np.float32)

# 时间序列切分（70/15/15 按时间顺序）
n = len(data_X)
n_tr, n_va = int(n*0.7), int(n*0.15)
X_tr, y_tr = data_X[:n_tr], data_y[:n_tr]
X_va, y_va = data_X[n_tr:n_tr+n_va], data_y[n_tr:n_tr+n_va]
X_te, y_te = data_X[n_tr+n_va:], data_y[n_tr+n_va:]

# 目标归一化
y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-8
y_tr_n, y_va_n, y_te_n = (y_tr-y_mu)/y_sd, (y_va-y_mu)/y_sd, (y_te-y_mu)/y_sd
print(f"训练 {len(X_tr)} 验证 {len(X_va)} 测试 {len(X_te)}")

# ========== 工具 ==========
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim))
    def forward(self, x):
        return self.net(x)

def run_epoch(model, X, y, opt, loss_fn, quantile=False):
    model.train()
    Xt, yt = torch.tensor(X, device=DEVICE), torch.tensor(y, device=DEVICE)
    opt.zero_grad()
    out = model(Xt)
    if quantile:
        # 分位数损失: 三个分位 [0.1, 0.5, 0.9]
        qs = torch.tensor([0.1, 0.5, 0.9], device=DEVICE)
        yq = yt.unsqueeze(1)
        loss = torch.mean(torch.stack([
            torch.mean(torch.maximum(q*(yq-out[:,i:i+1]), (q-1)*(yq-out[:,i:i+1])))
            for i, q in enumerate(qs)]))
    else:
        loss = loss_fn(out.squeeze(), yt)
    loss.backward()
    opt.step()
    return loss.item()

def eval_rmse(model, X, y, quantile=False, median_idx=1):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, device=DEVICE)
        out = model(Xt)
        if quantile:
            pred = out[:, median_idx]  # 中位数预测
        else:
            pred = out.squeeze()
    return torch.sqrt(torch.mean((pred.cpu() - torch.tensor(y))**2)).item() * y_sd  # 还原尺度

def train_model(in_dim, out_dim, X_tr, y_tr, X_va, y_va, X_te, y_te,
                quantile=False, epochs=60, hidden=32):
    model = MLP(in_dim, out_dim, hidden).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        loss = run_epoch(model, X_tr, y_tr, opt, loss_fn, quantile)
        if ep % 20 == 0:
            rmse = eval_rmse(model, X_va, y_va, quantile)
            print(f"    ep{ep} loss={loss:.4f} val_rmse={rmse:.4f}", flush=True)
    te_rmse = eval_rmse(model, X_te, y_te, quantile)
    return model, te_rmse

# ========== 实验矩阵 ==========
print("\n========== 探索性测试 ==========")
results = {}

# --- E0 基线: 普通回归 ---
print("\n[E0] 基线 普通回归 (MSE)", flush=True)
_, r0 = train_model(X_tr.shape[1], 1, X_tr, y_tr_n, X_va, y_va_n, X_te, y_te_n, quantile=False, epochs=50)
results["E0_baseline"] = r0
print(f"  → 测试 RMSE = {r0:.4f}", flush=True)

# --- E1 分位数输出 (不确定性) ---
print("\n[E1] 分位数输出 (10/50/90)", flush=True)
_, r1 = train_model(X_tr.shape[1], 3, X_tr, y_tr_n, X_va, y_va_n, X_te, y_te_n, quantile=True, epochs=50)
results["E1_quantile"] = r1
print(f"  → 测试 RMSE(中位数) = {r1:.4f}", flush=True)

# --- E2 多任务 loss 归一化验证 ---
print("\n[E2] 多任务 loss 归一化 vs 未归一化", flush=True)
# 构造 M2 分类伪任务（基于水温>中位数做标签，代表分层状态）
strat_label = (s["water_temp"] > s["water_temp"].median()).astype(np.float32).values
# 用训练/测试切分的相同索引
y_strat_tr, y_strat_va, y_strat_te = strat_label[:n_tr], strat_label[n_tr:n_tr+n_va], strat_label[n_tr+n_va:]

class MultiHead(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU())
        self.head_reg = nn.Linear(hidden, 1)
        self.head_cls = nn.Linear(hidden, 2)
    def forward(self, x):
        h = self.shared(x)
        return self.head_reg(h).squeeze(), self.head_cls(h)

def train_multitask(X_tr, y_tr, yc_tr, X_va, y_va, yc_va, X_te, y_te, yc_te, loss_scale=True):
    model = MultiHead(X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE)
    yct = torch.tensor(yc_tr, dtype=torch.long, device=DEVICE)
    Xv = torch.tensor(X_va, device=DEVICE); yv = torch.tensor(y_va, device=DEVICE)
    ycv = torch.tensor(yc_va, dtype=torch.long, device=DEVICE)
    reg_loss = nn.MSELoss(); cls_loss = nn.CrossEntropyLoss()
    reg_scale = 1.0 if not loss_scale else (y_tr.std() + 1e-8)  # 归一化后的 std≈1，用近似尺度
    for ep in range(50):
        model.train(); opt.zero_grad()
        rp, cp = model(Xt)
        lr_ = reg_loss(rp, yt) * reg_scale
        lc = cls_loss(cp, yct)
        loss = lr_ + lc
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        rp, cp = model(Xv)
        val_rmse = torch.sqrt(torch.mean((rp - yv)**2)).item()
        val_acc = (cp.argmax(1).cpu() == ycv.cpu()).float().mean().item()
    return val_rmse, val_acc

# 未归一化: 回归 loss 大（y~N(0,1) 且 MSE ~0.1-1），分类 CE ~0.7，未归一化时回归主导
vr0, va0 = train_multitask(X_tr, y_tr_n, y_strat_tr, X_va, y_va_n, y_strat_va, X_te, y_te_n, y_strat_te, loss_scale=False)
print(f"  未归一化: val_rmse={vr0:.4f} val_acc={va0:.4f}", flush=True)
# 归一化: 让两个 loss 同量纲
vr1, va1 = train_multitask(X_tr, y_tr_n, y_strat_tr, X_va, y_va_n, y_strat_va, X_te, y_te_n, y_strat_te, loss_scale=True)
print(f"  已归一化: val_rmse={vr1:.4f} val_acc={va1:.4f}", flush=True)
results["E2_scale"] = (vr0, va0, vr1, va1)

# --- E3 M2 旁路 → M1 (分层状态作为特征) ---
print("\n[E3] 分层状态作为特征喂给预测 (旁路)", flush=True)
# 无旁路基线（已在 E0），这里做"加 strat 特征"
X_tr_s = np.column_stack([X_tr, strat_label[:n_tr].reshape(-1,1)])
X_va_s = np.column_stack([X_va, strat_label[n_tr:n_tr+n_va].reshape(-1,1)])
X_te_s = np.column_stack([X_te, strat_label[n_tr+n_va:].reshape(-1,1)])
_, r3 = train_model(X_tr_s.shape[1], 1, X_tr_s, y_tr_n, X_va_s, y_va_n, X_te_s, y_te_n, quantile=False, epochs=50)
results["E3_bypass"] = r3
print(f"  → 测试 RMSE = {r3:.4f} (基线 E0 = {r0:.4f})", flush=True)

# --- E4 滞后窗口特征 ---
print("\n[E4] 滞后窗口特征 (1-3天降水/风累计)", flush=True)
# 构建 1/2/3 天前的气象特征（这里用 pandas shift，按气象时间步）
lag_df = s[["timestamp", "rainfall", "wind_speed"]].copy()
for lag_days in [1, 2, 3]:
    lag_df[f"rain_lag{lag_days}"] = lag_df["rainfall"].shift(int(lag_days * 24 / 3))  # ~3h/步近似
    lag_df[f"wind_lag{lag_days}"] = lag_df["wind_speed"].shift(int(lag_days * 24 / 3))
lag_df = lag_df.fillna(0)
lag_feats = lag_df[["rain_lag1", "rain_lag2", "rain_lag3", "wind_lag1", "wind_lag2", "wind_lag3"]].values.astype(np.float32)
X_tr_l = np.column_stack([X_tr, lag_feats[:n_tr]])
X_va_l = np.column_stack([X_va, lag_feats[n_tr:n_tr+n_va]])
X_te_l = np.column_stack([X_te, lag_feats[n_tr+n_va:]])
_, r4 = train_model(X_tr_l.shape[1], 1, X_tr_l, y_tr_n, X_va_l, y_va_n, X_te_l, y_te_n, quantile=False, epochs=50)
results["E4_lag"] = r4
print(f"  → 测试 RMSE = {r4:.4f} (基线 E0 = {r0:.4f})", flush=True)

# ========== 汇总 ==========
print("\n========== 汇总 ==========")
print(f"E0 基线:         {results['E0_baseline']:.4f}")
print(f"E1 分位数(中位数): {results['E1_quantile']:.4f}")
print(f"E3 旁路:         {results['E3_bypass']:.4f}")
print(f"E4 滞后特征:     {results['E4_lag']:.4f}")
print(f"E2 多任务: 未归一化(val_rmse={results['E2_scale'][0]:.4f}, acc={results['E2_scale'][1]:.4f}) "
      f"vs 归一化(val_rmse={results['E2_scale'][2]:.4f}, acc={results['E2_scale'][3]:.4f})")
