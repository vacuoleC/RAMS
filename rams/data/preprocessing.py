# -*- coding: utf-8 -*-
"""RAMS 数据预处理（唯一接触原始 xlsx 的地方）

按 01-数据规范.md 契约，把 data/raw/ 的 3 个 xlsx 清洗成标准长表
`standard.parquet`（模型/训练/分析唯一数据源），只跑一次。

清洗流程：
  1. 藻类 20 个深度 sheet 纵向合并（加 depth 列）
  2. 气象 Sheet1 取 10min → 3h 重采样
  3. merge_asof 最近时刻对齐（藻类各深度时间戳有 4-5 分钟错位，需先 floor 到 3h 网格）
  4. 中英列名转换（英文 snake_case）
  5. 类型固化（float32 / datetime64）
  6. 输出 standard.parquet + norm_stats.json（只用训练段拟合）

注意事项（探索测试验证）：
  - 各深度层时间戳精确交集为 0，floor 到 3h 网格后 97.8% 对齐
  - 藻类采样间隔 1-5 小时（非文档声称的 3h），气象 10min
  - 部分 sheet 自带「深度」列，合并前需删除避免重复

用法：
  python -m rams.data.preprocessing --raw data/raw --out data/processed
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# ---- 中英列名映射（01-数据规范.md 第 3 节） ----
CN2EN: dict[str, str] = {
    "站点编号": "site_id",
    "测量时间": "timestamp",
    "深度": "depth",
    "水温": "water_temp",
    "透光率": "transmittance",
    "绿藻浓度": "green_conc",
    "蓝藻浓度": "cyano_conc",
    "硅藻浓度": "diatom_conc",
    "隐藻浓度": "crypto_conc",
    "总浓度": "total_conc",
    "黄色物质": "cdom",
    "绿藻细胞数": "green_cells",
    "蓝藻细胞数": "cyano_cells",
    "硅藻细胞数": "diatom_cells",
    "隐藻细胞数": "crypto_cells",
    "总细胞数": "total_cells",
    "风速": "wind_speed",
    "风向": "wind_dir",
    "压力": "pressure",
    "温度": "air_temp",
    "湿度": "humidity",
    "降雨量": "rainfall",
}

# 数值列（float32）
NUMERIC_COLS: list[str] = [
    "depth", "water_temp", "transmittance",
    "green_conc", "cyano_conc", "diatom_conc", "crypto_conc", "total_conc",
    "cdom", "green_cells", "cyano_cells", "diatom_cells", "crypto_cells", "total_cells",
    "wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall",
]

# 关键文件（不硬编码中文名之外的读取逻辑；文件名由调用方传入）
ALGAE_FILE = "藻类检测数据_分深度整理.xlsx"
METEO_FILE = "气象数据.xlsx"
MATCH_FILE = "藻类_气象_最近时刻匹配.xlsx"

RESAMPLE_PERIOD = "3h"  # 气象 10min → 3h 网格，对齐藻类层采样


def _rename_cols(df: pd.DataFrame) -> pd.DataFrame:
    """中英列名转换：去掉 _x/_y 后缀后映射为英文 snake_case。"""
    cols = [c.replace("_x", "").replace("_y", "") for c in df.columns]
    df = df.copy()
    df.columns = cols
    return df.rename(columns=CN2EN)


def load_algae(raw_dir: Path) -> pd.DataFrame:
    """读取藻类 20 sheet，纵向合并成含 depth 列的长表。"""
    fn = raw_dir / ALGAE_FILE
    wb = pd.read_excel(fn, sheet_name=None, engine="openpyxl")
    frames: list[pd.DataFrame] = []
    for name, df in wb.items():
        depth_val = float(name.rstrip("m"))
        df = _rename_cols(df)
        # 部分 sheet 自带「深度」列，转名后为 depth，删除避免与追加列重复
        if "depth" in df.columns:
            df = df.drop(columns=["depth"])
        df["depth"] = depth_val
        frames.append(df)
    algae = pd.concat(frames, ignore_index=True)
    algae["timestamp"] = pd.to_datetime(algae["timestamp"], errors="coerce")
    return algae


def load_meteo(raw_dir: Path) -> pd.DataFrame:
    """读取气象 Sheet1，重采样到 3h 网格（Sheet2/3 为空，忽略）。"""
    fn = raw_dir / METEO_FILE
    meteo = pd.read_excel(fn, sheet_name="Sheet1", engine="openpyxl")
    meteo = _rename_cols(meteo)
    meteo["timestamp"] = pd.to_datetime(meteo["timestamp"], errors="coerce")
    meteo = meteo.dropna(subset=["timestamp"]).sort_values("timestamp")
    # 10min → 3h 均值重采样
    meteo_3h = (
        meteo.set_index("timestamp")
        .resample(RESAMPLE_PERIOD)
        .mean(numeric_only=True)
        .reset_index()
    )
    return meteo_3h


def align_and_clean(algae: pd.DataFrame, meteo: pd.DataFrame) -> pd.DataFrame:
    """对齐 + 清洗，产出标准长表。

    核心：各深度层时间戳有 4-5 分钟错位（精确交集=0），
    先把藻类 timestamp floor 到 3h 网格，再与气象 merge_asof 最近时刻对齐。
    """
    # 1. 藻类 floor 到 3h 网格（对齐各深度层）
    algae = algae.copy()
    algae["ts3h"] = algae["timestamp"].dt.floor(RESAMPLE_PERIOD)

    # 2. merge_asof 对齐气象（nearest，容差 3h）
    meteo_cols = [c for c in meteo.columns if c != "timestamp"]
    meteo_align = meteo.sort_values("timestamp")
    algae_sorted = algae.sort_values("ts3h")
    df = pd.merge_asof(
        algae_sorted, meteo_align, left_on="ts3h", right_on="timestamp",
        direction="nearest", tolerance=pd.Timedelta("3h"),
    )
    # 合并后保留左侧 ts3h 为统一时间轴，丢弃右侧 timestamp 副本
    df = df.drop(columns=["timestamp_x", "timestamp_y"], errors="ignore")
    df = df.rename(columns={"ts3h": "timestamp"})
    return df


def _cast_types(df: pd.DataFrame) -> pd.DataFrame:
    """类型固化：数值列 float32，site_id 转字符串保留前导零。"""
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].astype(str).str.zfill(4)
    return df


def compute_norm_stats(df: pd.DataFrame, train_frac: float = 0.7) -> dict:
    """只用训练段拟合归一化参数（防数据泄漏），存 norm_stats.json。

    返回 {列名: {"mean": float, "std": float}}，仅含数值列。
    """
    df_sorted = df.sort_values("timestamp")
    n_tr = int(len(df_sorted) * train_frac)
    train_part = df_sorted.iloc[:n_tr]
    stats: dict = {}
    for c in NUMERIC_COLS:
        if c in train_part.columns:
            s = train_part[c].dropna()
            if len(s) > 0:
                stats[c] = {"mean": float(s.mean()), "std": float(s.std()) + 1e-8}
    return stats


def build_dataset(
    raw_dir: str | Path,
    out_dir: str | Path,
    train_frac: float = 0.7,
    verbose: bool = True,
) -> Path:
    """主入口：xlsx → standard.parquet + norm_stats.json。

    Args:
        raw_dir: data/raw/ 目录（含 3 个 xlsx）
        out_dir: data/processed/ 输出目录
        train_frac: 训练段占比，用于 norm_stats 拟合
    Returns:
        standard.parquet 路径
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("[1/4] 读取藻类 20 sheet ...", flush=True)
    algae = load_algae(raw_dir)
    if verbose:
        print(f"  藻类: {algae.shape}", flush=True)

    if verbose:
        print("[2/4] 读取气象并重采样 ...", flush=True)
    meteo = load_meteo(raw_dir)
    if verbose:
        print(f"  气象: {meteo.shape}", flush=True)

    if verbose:
        print("[3/4] 对齐 + 清洗 ...", flush=True)
    df = align_and_clean(algae, meteo)
    df = _cast_types(df)
    if verbose:
        print(f"  对齐后: {df.shape}", flush=True)

    if verbose:
        print("[4/4] 计算归一化参数 + 保存 ...", flush=True)
    stats = compute_norm_stats(df, train_frac)

    out_parquet = out_dir / "standard.parquet"
    df.to_parquet(out_parquet, index=False)

    out_stats = out_dir / "norm_stats.json"
    with open(out_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if verbose:
        print(f"  输出: {out_parquet} ({df.shape})", flush=True)
        print(f"  norm_stats: {len(stats)} 列", flush=True)
    return out_parquet


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAMS 数据预处理")
    parser.add_argument("--raw", default="data/raw", help="原始 xlsx 目录")
    parser.add_argument("--out", default="data/processed", help="输出目录")
    parser.add_argument("--train-frac", type=float, default=0.7)
    args = parser.parse_args()
    build_dataset(args.raw, args.out, args.train_frac)
