# -*- coding: utf-8 -*-
"""build_dataset CLI：xlsx → standard.parquet（唯一接触原始数据的地方）

用法：
  python -m scripts.build_dataset --raw data/raw --out data/processed
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="构建 RAMS 标准数据集")
    parser.add_argument("--raw", default="data/raw", help="原始 xlsx 目录")
    parser.add_argument("--out", default="data/processed", help="输出目录")
    parser.add_argument("--train-frac", type=float, default=0.7, help="训练段占比（norm_stats 拟合）")
    args = parser.parse_args()

    from rams.data.preprocessing import build_dataset

    out = build_dataset(args.raw, args.out, args.train_frac)
    print(f"\n✅ 标准数据集已生成: {out}")


if __name__ == "__main__":
    main()
