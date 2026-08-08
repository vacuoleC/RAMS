# -*- coding: utf-8 -*-
"""探索性测试 · 数据预处理：3 个 xlsx → 标准长表 parquet

产出（在 data/explore/ 下）：
  - algae_long.parquet   # 20 sheet 合并 + 中英列名，含 depth
  - meteo.parquet        # 气象原生 10min
  - standard_long.parquet # 合并后的标准长表（按 (timestamp, depth) 对齐）

注意：这是探索性测试的数据准备，字段契约沿用 01-数据规范.md。
只输出形状/列名/统计量，不输出原始数值行。
"""
import io
import sys
import os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np

DATA_DIR = "/data/RAMS"
OUT_DIR = "/data/RAMS"
os.makedirs(OUT_DIR, exist_ok=True)

CN2EN = {
    "站点编号": "site_id", "测量时间": "timestamp", "深度": "depth",
    "水温": "water_temp", "透光率": "transmittance",
    "绿藻浓度": "green_conc", "蓝藻浓度": "cyano_conc",
    "硅藻浓度": "diatom_conc", "隐藻浓度": "crypto_conc", "总浓度": "total_conc",
    "黄色物质": "cdom",
    "绿藻细胞数": "green_cells", "蓝藻细胞数": "cyano_cells",
    "硅藻细胞数": "diatom_cells", "隐藻细胞数": "crypto_cells", "总细胞数": "total_cells",
    "风速": "wind_speed", "风向": "wind_dir", "压力": "pressure",
    "温度": "air_temp", "湿度": "humidity", "降雨量": "rainfall",
}

def load_algae():
    """20 sheet 合并成藻类长表"""
    fn = os.path.join(DATA_DIR, "藻类检测数据_分深度整理.xlsx")
    wb = pd.read_excel(fn, sheet_name=None, engine="openpyxl")
    frames = []
    for name, df in wb.items():
        depth_val = float(name.rstrip("m"))
        # 先转列名，再删除可能存在的 depth（sheet 自带"深度"列时转后为 depth）
        df.columns = [c.replace("_x", "").replace("_y", "") for c in df.columns]
        df.rename(columns=CN2EN, inplace=True)
        if "depth" in df.columns:
            df = df.drop(columns=["depth"])
        df["depth"] = depth_val
        frames.append(df)
    algae = pd.concat(frames, ignore_index=True)
    return algae

def load_meteo():
    fn = os.path.join(DATA_DIR, "气象数据.xlsx")
    meteo = pd.read_excel(fn, sheet_name="Sheet1", engine="openpyxl")
    meteo.columns = [c.replace("_x", "").replace("_y", "") for c in meteo.columns]
    meteo.rename(columns=CN2EN, inplace=True)
    return meteo

def main():
    print("[1/4] 读取藻类 20 sheet ...", flush=True)
    algae = load_algae()
    print(f"  藻类: {algae.shape}", flush=True)

    print("[2/4] 读取气象 ...", flush=True)
    meteo = load_meteo()
    print(f"  气象: {meteo.shape}", flush=True)

    print("[3/4] 解析时间 ...", flush=True)
    # 列名转换已在 load_algae/load_meteo 内完成；这里统一解析时间
    for df in (algae, meteo):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print("[4/4] 保存 parquet ...", flush=True)
    algae.to_parquet(os.path.join(OUT_DIR, "algae_long.parquet"), index=False)
    meteo.to_parquet(os.path.join(OUT_DIR, "meteo.parquet"), index=False)
    print(f"  algae_long: {algae.shape} 列={list(algae.columns)[:10]}...", flush=True)
    print(f"  meteo: {meteo.shape} 列={list(meteo.columns)}", flush=True)
    print("完成", flush=True)

if __name__ == "__main__":
    main()
