# -*- coding: utf-8 -*-
"""P1 数据管线：构建时序预测张量 (B, T, D, C) + 真实分层标签

产出 parquet/npz 供 P2-P5 使用：
  - tensor_ready.parquet: 对齐后的 20 层完整长表（含分层指标特征）
  - 张量切分信息

分层标签（真实，从温度剖面计算）：
  - delta_T: 表层(0.5m) - 底层(10m) 温差 —— 分层强度主指标
  - thermo_grad: 温跃层最大梯度（深度方向）
  - strat_binary: 是否分层（delta_T > 阈值，阈值取训练段分位数）
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

print("读取数据 ...", flush=True)
algae = pd.read_parquet("/data/RAMS/algae_long.parquet")
meteo = pd.read_parquet("/data/RAMS/meteo.parquet")

# ========== 1. 20 层完整长表 ==========
print("合并 20 层 + 气象 ...", flush=True)
meteo_sorted = meteo.sort_values("timestamp")
# 气象 10min → 3h 重采样（对齐藻类 3h 间隔），再 merge_asof
meteo_3h = meteo_sorted.set_index("timestamp").resample("3h").mean(numeric_only=True).reset_index()
# 气象列保留
meteo_cols = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
meteo_3h = meteo_3h[["timestamp"] + meteo_cols]

algae = algae.sort_values("timestamp").reset_index(drop=True)
df = pd.merge_asof(
    algae, meteo_3h, on="timestamp", direction="nearest", tolerance=pd.Timedelta("3h"))

# 数值列类型
num_cols = ["water_temp", "transmittance", "green_conc", "cyano_conc", "diatom_conc",
            "crypto_conc", "total_conc", "cdom", "green_cells", "cyano_cells",
            "diatom_cells", "crypto_cells", "total_cells"] + meteo_cols
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(f"合并后: {df.shape}，时间 {df['timestamp'].min()} ~ {df['timestamp'].max()}", flush=True)

# ========== 2. 构建 20 层宽表（每时刻一行，20 层水温/浓度） ==========
print("构建 20 层宽表 ...", flush=True)
# 关键：各深度层时间戳有 4-5 分钟错位（0.5m=01:01:41, 10m=01:06:12），精确交集=0
# 正确做法：先把每个深度层 floor 到 3h 网格重采样（取均值），再透视成宽表
def to_3h_grid(sub):
    sub = sub.copy()
    sub["ts3h"] = sub["timestamp"].dt.floor("3h")
    return sub

df["ts3h"] = df["timestamp"].dt.floor("3h")
# 气象 10min → 3h 重采样已做过（meteo_3h），这里同样用 ts3h 对齐
# 聚合：同一 (ts3h, depth) 取均值
agg = df.groupby(["ts3h", "depth"]).agg(
    water_temp=("water_temp", "mean"),
    total_conc=("total_conc", "mean")).reset_index()
print(f"  (ts3h, depth) 网格: {agg.shape}, 唯一时刻 {agg['ts3h'].nunique()}", flush=True)

pivot_temp = agg.pivot_table(index="ts3h", columns="depth", values="water_temp", aggfunc="mean")
pivot_conc = agg.pivot_table(index="ts3h", columns="depth", values="total_conc", aggfunc="mean")
pivot_temp.columns = [float(c) for c in pivot_temp.columns]
pivot_conc.columns = [float(c) for c in pivot_conc.columns]
print(f"  pivot_temp: {pivot_temp.shape} (行=时刻数, 列=深度)", flush=True)
# 气象（按 ts3h 取唯一行）
meteo_ts = df.drop_duplicates("ts3h")[["ts3h"] + meteo_cols].set_index("ts3h")
print(f"  meteo_ts: {meteo_ts.shape}", flush=True)

# ========== 3. 分层指标（真实，从温度剖面） ==========
print("计算分层指标 ...", flush=True)
depths = sorted(pivot_temp.columns.tolist())
surface = pivot_temp[0.5].values
bottom = pivot_temp[10.0].values
delta_T = surface - bottom  # 表层-底层温差

# 温跃层最大梯度
grads = np.zeros(len(pivot_temp))
for i in range(len(depths) - 1):
    d1, d2 = depths[i], depths[i+1]
    g = (pivot_temp[d2].values - pivot_temp[d1].values) / (d2 - d1)
    grads = np.maximum(grads, g)

strat_df = pd.DataFrame({
    "ts3h": pivot_temp.index,
    "delta_T": delta_T,
    "thermo_grad": grads,
    "surface_temp": pivot_temp[0.5].values,
    "bottom_temp": pivot_temp[10.0].values,
}).set_index("ts3h")

# ========== 4. 拼接最终表 ==========
print("拼接最终表 ...", flush=True)
# 20 层水温（作为输入特征）
temp_wide = pivot_temp.add_prefix("temp_")
# 20 层总浓度（预测目标，M1）
conc_wide = pivot_conc.add_prefix("conc_")
# 分层指标（M2 标签）
strat_cols = strat_df[["delta_T", "thermo_grad", "surface_temp", "bottom_temp"]]

final = temp_wide.join(conc_wide).join(strat_cols).join(meteo_ts)
final = final.dropna()
print(f"最终宽表: {final.shape}，列数 {len(final.columns)}", flush=True)

# 分层二分类标签：用训练段的 delta_T 分位数作阈值
# 先按时间排序，训练段前 70%
final = final.sort_index()
n_tr = int(len(final) * 0.7)
train_dt = final["delta_T"].iloc[:n_tr]
threshold = train_dt.quantile(0.5)
final["strat_binary"] = (final["delta_T"] > threshold).astype(int)
print(f"分层阈值(delta_T p50): {threshold:.3f}，分层样本占比: {final['strat_binary'].mean():.3f}", flush=True)

# ========== 5. 保存 ==========
final.to_parquet("/data/RAMS/tensor_ready.parquet")
print(f"保存 tensor_ready.parquet: {final.shape}", flush=True)
print("列前缀分布:", {p: sum(c.startswith(p) for c in final.columns) for p in ["temp_", "conc_"]}, flush=True)
print("P1 完成", flush=True)
