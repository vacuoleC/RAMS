#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1: 五头系统整合训练 —— 整合 M5 因果结论 + M3 选层结论 vs 基线

在真实数据上做「整合 vs 基线」正式对比实验（每个配置 3 seed，报告均值±std）：

  配置 A（基线）      ：20 层水温（temp_0.5~10.0）+ 6 气象            （当前特征）
  配置 B（M5 整合）   ：A + wind_u 短滞特征（风 u 分量）
                       M5 实证：藻类自回归主导；wind_u 短滞(≤2天)是唯一有信号的气象直驱；
                       **不加** 20-30 天滞后气象（季节共变非因果，会引入噪声）
  配置 C（M3 整合）   ：M3 跨 seed 共识最优 5 层（近表1 + 中层1 + 深3）+ 6 气象
                       （验证 5 层够不够 —— 若精度接近 B 但省 75% 输入则为工程价值）
  配置 D（M5+M3）    ：B + C

评估指标：M1 RMSE（表层浓度，还原尺度）、M2 分层 acc、M4 预警 acc、
          p10-p90 区间覆盖率。只输出统计量/结论，不打印原始数据数值行。

数据/训练细节：
  - 宽表复用 TensorBuilder._load_wide（同样的 3h 网格/聚合/dropna/时间轴），
    保证 A/B/C/D 用完全相同的 train/val/test 窗口，仅特征列不同（公平对比）。
  - wind_u = wind_speed * cos(wind_dir)：风向为环形量，先转 u 分量避免对角度取均值
    （与 M5 load_wide 一致）；作为每时刻特征通道加入，T=24(=3天) 窗口天然覆盖
    wind_u 的 0~2 天短滞效应。
  - M3 共识 5 层 = [1.5, 5.0, 8.5, 9.5, 10.0]m（近表 1.5 + 中层 5.0 + 深 8.5/9.5/10.0）。
  - 30 epoch × 3 seed，多任务 w_m1/w_m2/w_m4 = 1/3/2，M4 自动类别加权（与 trainer 默认一致）。

用法（算力机 /data/RAMS/proj 下）：
  python3 scripts/explore/t1_integration.py            # 全量 4 配置 × 3 seed × 30 epoch
  python3 scripts/explore/t1_integration.py --smoke     # 冒烟：4 配置 × 1 seed × 2 epoch
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from rams.data.tensor_builder import TensorBuilder, TensorConfig, METEO_COLS
from rams.models.rams_net import RamsNet
from rams.training.trainer import Trainer

T, H = 24, 8            # 回看 3 天 / 预测 24h
TRAIN_FRAC, VAL_FRAC = 0.7, 0.15
EPOCHS = 30
SEEDS = [0, 1, 2]

# M3 跨 seed 共识最优 5 层：近表层 1 层（1.5m）+ 中层 1 层（5.0m）+ 深层 3 层（8.5/9.5/10.0m）
M3_FIVE_DEPTHS = [1.5, 5.0, 8.5, 9.5, 10.0]


def load_wide(parquet: str):
    """读取宽表（复用 TensorBuilder 逻辑，保证与正式管线一致）。"""
    loader = TensorBuilder(TensorConfig(T=T, H=H))
    wide = loader._load_wide(Path(parquet))
    return wide.sort_index()


def build_dataset(wide, use_wind_u: bool, selected_depths, seed: int):
    """按配置构建 (X, y, strat, warn) 并 70/15/15 时间切分。

    selected_depths=None → 全 20 层；否则只保留选中的 temp_ 列。
    窗口/切分与 TensorBuilder._make_windows 逐一对齐（strat/warn 阈值用训练段，防泄漏）。
    """
    np.random.seed(seed)
    df = wide.copy()

    # M5：加 wind_u 短滞特征（风向 → u 分量）
    if use_wind_u:
        dir_rad = np.deg2rad(df["wind_dir"].values)
        df["wind_u"] = df["wind_speed"].values * np.cos(dir_rad)
    # M3：只保留选中深度层的 temp_ 列
    if selected_depths is not None:
        keep = {f"temp_{d:.1f}" for d in selected_depths}
        drop = [c for c in df.columns if c.startswith("temp_") and c not in keep]
        df = df.drop(columns=drop)

    n = len(df)
    n_tr = int(n * TRAIN_FRAC)

    temp_cols = [c for c in df.columns if c.startswith("temp_")]
    conc_cols = [c for c in df.columns if c.startswith("conc_")]
    feat_cols = temp_cols + list(METEO_COLS)
    if use_wind_u:
        feat_cols = feat_cols + ["wind_u"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    # 特征归一化（只用训练段拟合，防泄漏）
    tr = df.iloc[:n_tr]
    X = df[feat_cols].values.astype(np.float32)
    for i, c in enumerate(feat_cols):
        mu, sd = float(tr[c].mean()), float(tr[c].std()) + 1e-8
        X[:, i] = (X[:, i] - mu) / sd
    # 目标 = 表层浓度，训练段归一化
    yc = conc_cols[0]
    y_mu, y_sd = float(tr[yc].mean()), float(tr[yc].std()) + 1e-8
    y = (df[yc].values.astype(np.float32) - y_mu) / y_sd

    # 滑动窗口
    n_w = n - T - H
    Xw = np.stack([X[i:i + T] for i in range(n_w)]).astype(np.float32)
    yw = np.stack([y[i + T:i + T + H] for i in range(n_w)]).astype(np.float32)

    # M2 分层标签（窗口末时刻的分层状态，训练段中位数阈值）
    delta = df["delta_T"].values
    thr = float(np.median(delta[:n_tr]))
    strat = (delta > thr).astype(np.int64)
    strat_w = np.array([strat[i + T - 1] for i in range(n_w)])

    # M4 预警标签（未来 24h 峰值分级，训练段分位数阈值 p75/p90/p97）
    warn_val = yw.max(axis=1)
    n_win_tr = int(len(yw) * TRAIN_FRAC)
    qs = np.quantile(warn_val[:n_win_tr], [0.75, 0.90, 0.97])
    warn_w = np.searchsorted(qs, warn_val).astype(np.int64)

    # 切分（与 TensorBuilder 相同的窗口索引比例）
    n_trw, n_vaw = int(len(Xw) * TRAIN_FRAC), int(len(Xw) * VAL_FRAC)
    splits = {
        "train": (Xw[:n_trw], yw[:n_trw], strat_w[:n_trw], warn_w[:n_trw]),
        "val": (Xw[n_trw:n_trw + n_vaw], yw[n_trw:n_trw + n_vaw],
                strat_w[n_trw:n_trw + n_vaw], warn_w[n_trw:n_trw + n_vaw]),
        "test": (Xw[n_trw + n_vaw:], yw[n_trw + n_vaw:],
                 strat_w[n_trw + n_vaw:], warn_w[n_trw + n_vaw:]),
    }
    return splits, Xw.shape[2], y_sd


def run_config(wide, name, use_wind_u, selected_depths, epochs, seeds, smoke):
    """对单个配置跑 3 seed，返回各指标均值±std。"""
    rmses, accs, waccs, covs = [], [], [], []
    feat_dim = None
    for seed in seeds:
        splits, fd, y_sd = build_dataset(wide, use_wind_u, selected_depths, seed)
        feat_dim = fd
        (Xtr, ytr, str_tr, wr), (Xva, yva, str_va, wva), (Xte, yte, str_te, wte) = (
            splits["train"], splits["val"], splits["test"])

        torch.manual_seed(seed)
        np.random.seed(seed)
        model = RamsNet(feat_dim=fd, horizon=H, use_m4=True)
        trainer = Trainer(model)  # w_m1/w_m2/w_m4 = 1/3/2（与训练器默认一致）
        trainer.fit(Xtr, ytr, str_tr, Xva, yva, str_va,
                    warn_tr=wr, warn_va=wva, epochs=epochs, batch_size=128)
        res = trainer.evaluate(Xte, yte, str_te, wte, y_sd)
        rmses.append(res["rmse"])
        accs.append(res["acc"])
        waccs.append(res.get("warn_acc", np.nan))
        covs.append(res.get("coverage", np.nan))

    def ms(vals):
        return float(np.mean(vals)), float(np.std(vals))

    m_rmse, s_rmse = ms(rmses)
    m_acc, s_acc = ms(accs)
    m_wacc, s_wacc = ms(waccs)
    m_cov, s_cov = ms(covs)
    return {
        "feat_dim": int(feat_dim), "seeds": list(seeds), "epochs": epochs,
        "rmse_mean": m_rmse, "rmse_std": s_rmse,
        "acc_mean": m_acc, "acc_std": s_acc,
        "warn_acc_mean": m_wacc, "warn_acc_std": s_wacc,
        "coverage_mean": m_cov, "coverage_std": s_cov,
    }


def fmt(vals, nd=3):
    return f"{vals['rmse_mean']:.3f}±{vals['rmse_std']:.3f}"


def main():
    ap = argparse.ArgumentParser(description="T1 整合 M5/M3 结论 vs 基线（3-seed）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 seed × 2 epoch")
    ap.add_argument("--out-json", default="docs/t1_integration_results.json")
    args = ap.parse_args()

    t0 = time.time()
    print("== T1 整合 vs 基线（3-seed，30 epoch）==", flush=True)
    print("[1/2] 读取宽表 ...", flush=True)
    wide = load_wide(args.parquet)
    print(f"      宽表 {len(wide)} 时刻 × {wide.shape[1]} 列", flush=True)

    epochs = 2 if args.smoke else EPOCHS
    seeds = [0] if args.smoke else SEEDS
    configs = [
        ("A 基线20层+气象",     False, None),
        ("B A+wind_u(M5)",     True,  None),
        ("C M3最优5层(近表1+中层1+深3)", False, M3_FIVE_DEPTHS),
        ("D M5+M3",            True,  M3_FIVE_DEPTHS),
    ]

    print(f"[2/2] 训练 {len(configs)} 配置 × {len(seeds)} seed × {epochs} epoch ...", flush=True)
    results = {}
    for name, use_wu, depths in configs:
        r = run_config(wide, name, use_wu, depths, epochs, seeds, args.smoke)
        results[name] = r
        print(f"  {name:<22s} feat={r['feat_dim']:<3d} "
              f"RMSE={fmt(r)}  M2acc={r['acc_mean']:.3f}±{r['acc_std']:.3f} "
              f"M4acc={r['warn_acc_mean']:.3f}±{r['warn_acc_std']:.3f} "
              f"cov={r['coverage_mean']:.3f}±{r['coverage_std']:.3f}", flush=True)

    print("\n===== 对照表（均值±std，3 seed）=====", flush=True)
    print(f"  {'配置':<22s} {'feat':<5s} {'M1 RMSE':<16s} {'M2 acc':<16s} "
          f"{'M4 acc':<16s} {'覆盖p10-90':<16s}", flush=True)
    for name, r in results.items():
        print(f"  {name:<22s} {r['feat_dim']:<5d} {fmt(r):<16s} "
              f"{r['acc_mean']:.3f}±{r['acc_std']:.3f}      "
              f"{r['warn_acc_mean']:.3f}±{r['warn_acc_std']:.3f}      "
              f"{r['coverage_mean']:.3f}±{r['coverage_std']:.3f}", flush=True)

    # 差异表
    print("\n===== 关键对比（RMSE 差值 = 后项 - 前项，负=更好）=====", flush=True)
    pairs = [
        ("B vs A  (M5 wind_u 是否有用)", "A 基线20层+气象", "B A+wind_u(M5)"),
        ("C vs B  (5层 vs 20层)",        "B A+wind_u(M5)", "C M3最优5层(近表1+中层1+深3)"),
        ("D vs B  (M3 加到 20层+wind_u)", "B A+wind_u(M5)", "D M5+M3"),
        ("D vs C  (M5 加到 5层)",         "C M3最优5层(近表1+中层1+深3)", "D M5+M3"),
        ("D vs A  (整合 vs 基线)",        "A 基线20层+气象", "D M5+M3"),
    ]
    for label, k1, k2 in pairs:
        a, b = results[k1], results[k2]
        d = b["rmse_mean"] - a["rmse_mean"]
        d_std = np.sqrt(a["rmse_std"] ** 2 + b["rmse_std"] ** 2)
        print(f"  {label:<34s} RMSE {d:+.4f}±{d_std:.4f} ({d/max(a['rmse_mean'],1e-9)*100:+.1f}%) "
              f"M4acc {b['warn_acc_mean']-a['warn_acc_mean']:+.3f} "
              f"cov {b['coverage_mean']-a['coverage_mean']:+.3f}", flush=True)

    # 结论
    names = list(results)
    best = min(names, key=lambda k: results[k]["rmse_mean"])
    print("\n===== 结论 =====", flush=True)
    print(f"  最优配置（M1 RMSE 最低）: {best}", flush=True)
    print(f"  基线 A RMSE = {results[names[0]]['rmse_mean']:.3f}±{results[names[0]]['rmse_std']:.3f}", flush=True)
    print(f"  wind_u(M5) 增益: "
          f"{results['B A+wind_u(M5)']['rmse_mean']-results['A 基线20层+气象']['rmse_mean']:+.4f}", flush=True)
    print(f"  5层(M3) vs 20层: "
          f"{results['C M3最优5层(近表1+中层1+深3)']['rmse_mean']-results['B A+wind_u(M5)']['rmse_mean']:+.4f}", flush=True)
    print(f"  输入特征数: A {results['A 基线20层+气象']['feat_dim']} / "
          f"B {results['B A+wind_u(M5)']['feat_dim']} / "
          f"C {results['C M3最优5层(近表1+中层1+深3)']['feat_dim']} / "
          f"D {results['D M5+M3']['feat_dim']}", flush=True)
    print(f"  运行耗时: {(time.time()-t0)/60:.1f} 分钟", flush=True)

    # 存统计量（不含原始数据）
    out = Path(args.out_json)
    if args.smoke:
        out = out.with_name(out.stem + "_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {out}", flush=True)


if __name__ == "__main__":
    main()
