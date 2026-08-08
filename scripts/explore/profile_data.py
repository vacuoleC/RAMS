# -*- coding: utf-8 -*-
"""探索性测试 · 数据画像：确认数据形态，只输出统计信息（不输出原始数值）"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd

algae = pd.read_parquet("/data/RAMS/algae_long.parquet")
meteo = pd.read_parquet("/data/RAMS/meteo.parquet")

print("=== 藻类长表 ===")
print(f"形状: {algae.shape}")
print(f"时间范围: {algae['timestamp'].min()} ~ {algae['timestamp'].max()}")
print(f"深度层数: {algae['depth'].nunique()} 层: {sorted(algae['depth'].unique())}")
# 每层行数
per_depth = algae.groupby("depth").size()
print(f"每层行数: min={per_depth.min()}, max={per_depth.max()}, 各层= {dict(per_depth.head(5))}...")
# 时间间隔
t = algae["timestamp"].sort_values()
dt = t.diff().dropna()
print(f"时间间隔(小时) 中位数: {dt.dt.total_seconds().median()/3600:.2f}, 唯一值: {sorted(dt.dt.total_seconds().unique())[:5]}")

print("\n=== 气象 ===")
print(f"形状: {meteo.shape}")
print(f"时间范围: {meteo['timestamp'].min()} ~ {meteo['timestamp'].max()}")
t2 = meteo["timestamp"].sort_values()
dt2 = t2.diff().dropna()
print(f"时间间隔(分钟) 中位数: {dt2.dt.total_seconds().median()/60:.1f}")

print("\n=== 关键列统计（不显示原始行）===")
for col in ["water_temp", "cyano_conc", "total_conc", "green_conc", "wind_speed", "air_temp", "rainfall"]:
    if col in algae.columns:
        s = algae[col].dropna()
        print(f"  {col}: 非空={len(s)} 均值={s.mean():.3f} 分位[p5={s.quantile(.05):.3f}, p50={s.quantile(.5):.3f}, p95={s.quantile(.95):.3f}]")
    if col in meteo.columns:
        s = meteo[col].dropna()
        print(f"  {col}(气象): 非空={len(s)} 均值={s.mean():.3f} 分位[p5={s.quantile(.05):.3f}, p50={s.quantile(.5):.3f}, p95={s.quantile(.95):.3f}]")

# 缺失率
print("\n=== 缺失率 >5% 的列 ===")
for df, name in [(algae, "藻类"), (meteo, "气象")]:
    na = df.isna().mean()
    for c in na[na > 0.05].index:
        print(f"  {name}.{c}: {na[c]*100:.1f}%")
