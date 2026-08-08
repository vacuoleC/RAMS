# -*- coding: utf-8 -*-
"""M4 汇总：3 seed 稳定性验证关键结论

验证 P3/P4 的核心结论在 3 seed 下是否稳定：
  1. 垂直信息：单层 vs 20层
  2. 多任务 vs 单任务
  3. loss 加权：w0 未归一化 vs w1 归一化
输出均值±std，给出最终架构决策建议
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"设备: {DEVICE}", flush=True)

df = pd.read_parquet("/data/RAMS/tensor_ready.parquet")
temp_cols = [c for c in df.columns if c.startswith("temp_")]
meteo_cols = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
target_col = "conc_0.5"
T, H = 24, 8

# ========== 数据（同 P3/P4） ==========
def prep():
    X = df[temp_cols + meteo_cols].values.astype(np.float32)
    x_mu, x_sd = X.mean(0), X.std(0)+1e-8
    X = (X - x_mu) / x_sd
    y = df[target_col].values.astype(np.float32)
    y_mu, y_sd = y.mean(), y.std() + 1e-8
    y_n = (y - y_mu) / y_sd
    strat = df["strat_binary"].values.astype(np.int64)
    n = len(df)
    Xw, yw, sw = [], [], []
    for i in range(n - T - H):
        Xw.append(X[i:i+T]); yw.append(y_n[i+T:i+T+H]); sw.append(strat[i+T-1])
    Xw = np.stack(Xw); yw = np.stack(yw); sw = np.array(sw)
    nw = len(Xw); nt, nv = int(nw*0.7), int(nw*0.15)
    return (Xw[:nt], yw[:nt], sw[:nt]), (Xw[nt:nt+nv], yw[nt:nt+nv], sw[nt:nt+nv]), \
           (Xw[nt+nv:], yw[nt+nv:], sw[nt+nv:]), y_sd

(trX, try_, trs), (vaX, vay, vas), (teX, tey, tes), y_sd = prep()
print(f"train {len(trX)} val {len(vaX)} test {len(teX)}", flush=True)

# 单层输入
temp0 = [temp_cols[0]] + meteo_cols
Xs = df[temp0].values.astype(np.float32)
xs_mu, xs_sd = Xs.mean(0), Xs.std(0)+1e-8
Xs = (Xs - xs_mu) / xs_sd
n = len(df)
Xsw = []
for i in range(n - T - H):
    Xsw.append(Xs[i:i+T])
Xsw = np.stack(Xsw)
(trXs, _, _), (vaXs, _, _), (teXs, _, _) = \
    (Xsw[:int(len(Xsw)*0.7)], 0, 0), (Xsw[int(len(Xsw)*0.7):int(len(Xsw)*0.85)], 0, 0), (Xsw[int(len(Xsw)*0.85):], 0, 0)

class GRU(nn.Module):
    def __init__(self, in_dim, hidden=64, n_out=H, multi=False):
        super().__init__()
        self.rnn = nn.GRU(in_dim, hidden, batch_first=True)
        self.head1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
        self.multi = multi
        if multi:
            self.head2 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 2))
    def forward(self, x):
        out, _ = self.rnn(x)
        h = out[:, -1]
        if self.multi:
            return self.head1(h), self.head2(h)
        return self.head1(h)

def run(trX_, trY, trS, vaX_, vaY, vaS, teX_, teY, teS, multi, w1=1.0, w2=1.0, epochs=30, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    model = GRU(trX_.shape[2], multi=multi).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lm1 = nn.MSELoss(); lm2 = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.tensor(trX_), torch.tensor(trY), torch.tensor(trS)), batch_size=128, shuffle=False)
    for ep in range(epochs):
        model.train()
        for xb, yb, sb in dl:
            xb, yb, sb = xb.to(DEVICE), yb.to(DEVICE), sb.to(DEVICE)
            opt.zero_grad()
            if multi:
                rp, cp = model(xb); loss = w1*lm1(rp, yb) + w2*lm2(cp, sb)
            else:
                rp = model(xb); loss = lm1(rp, yb)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        if multi:
            rp, cp = model(torch.tensor(teX_).to(DEVICE))
            acc = (cp.argmax(1).cpu() == torch.tensor(teS)).float().mean().item()
        else:
            rp = model(torch.tensor(teX_).to(DEVICE)); acc = 0
        rmse = torch.sqrt(torch.mean((rp.cpu() - torch.tensor(teY))**2)).item() * y_sd
    return rmse, acc

print("\n===== M4: 3-seed 稳定性 =====", flush=True)
configs = {
    "单层(0.5m)":      (trXs, try_, trs, vaXs, vay, vas, teXs, tey, tes, False),
    "20层+气象":        (trX, try_, trs, vaX, vay, vas, teX, tey, tes, False),
    "20层+多任务(w1)":  (trX, try_, trs, vaX, vay, vas, teX, tey, tes, True),
    "20层+多任务(w2)":  (trX, try_, trs, vaX, vay, vas, teX, tey, tes, True),
}
w1_map = {"20层+多任务(w1)": 1.0, "20层+多任务(w2)": 1.0}
w2_map = {"20层+多任务(w1)": 1.0, "20层+多任务(w2)": 3.0}

results = {}
for name, args in configs.items():
    rmses, accs = [], []
    for seed in [0, 1, 2]:
        multi = args[9]
        w1 = 1.0
        w2 = w2_map.get(name, 1.0)
        rmse, acc = run(args[0], args[1], args[2], args[3], args[4], args[5],
                        args[6], args[7], args[8], multi, w1, w2, seed=seed)
        rmses.append(rmse); accs.append(acc)
    mean, std = np.mean(rmses), np.std(rmses)
    results[name] = (mean, std, np.mean(accs))
    print(f"  {name}: RMSE={mean:.3f}±{std:.3f} acc={np.mean(accs):.3f}", flush=True)

print("\n===== 最终架构决策建议 =====", flush=True)
base_single = results["单层(0.5m)"][0]
base_20 = results["20层+气象"][0]
print(f"垂直信息增益: 单层 {base_single:.3f} → 20层 {base_20:.3f} ({(base_single-base_20)/base_single*100:.0f}% 降误差)", flush=True)
for k in ["20层+多任务(w1)", "20层+多任务(w2)"]:
    m = results[k][0]
    print(f"{k}: {m:.3f} vs 20层单任务 {base_20:.3f} ({(base_20-m)/base_20*100:+.1f}%)", flush=True)
