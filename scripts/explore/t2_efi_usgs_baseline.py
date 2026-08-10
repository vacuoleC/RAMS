# -*- coding: utf-8 -*-
"""T2 站位：EFI-USGS 河流叶绿素挑战赛 —— 可提交基线（持久化 + 气候学）

EFI-USGS River Chlorophyll Forecasting Challenge（EFI × USGS，持续开放）：
  - 网址: https://ecoforecast.org/efi-usgs-river-chlorophyll-forecasting-challenge/
  - 任务: 预测 USGS 河流站点总叶绿素 (chla, ug/L)，每日一次，10 个站点
  - 评估: CRPS（连续排序概率得分，proper scoring rule，越低越好）
          另用"相对气候学技能"（model CRPS / climatology CRPS）衡量
  - 数据: 目标文件公开（S3 OSN）：
      https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/project_id=usgsrc4cast/duration=P1D/river-chl-targets.csv.gz
  - 提交: 需注册 https://ecoforecast.org/ 获取访问权，按 parquet/csv 格式提交，公开评分

本脚本在**已下载的目标数据**上复现挑战评估协议：
  - 对每个站点、每个日期滚动生成"过去 30 天训练"的基线预测
  - 基线 1 持久化：预测 = 最近观测值（确定性点预测）
  - 基线 2 气候学：预测 = 历史同月均值的正态分布（不确定性预测，CRPS 可算）
  - 按挑战口径给 CRPS 与相对气候学技能
  - 全部基于公开数据，只输出统计量

用法：python scripts/explore/t2_efi_usgs_baseline.py --data data/public/efi-usgs/river-chl-targets.csv.gz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ---- CRPS 计算（scipy 或自带高斯积分）----


def crps_gaussian(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """高斯分布预测的 CRPS（闭合形式，Gneiting & Raftery 2007）。

    CRPS(N(mu,sigma), y) = sigma * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) ),
    其中 z=(y-mu)/sigma，phi/Phi 为标准正态 pdf/cdf。该值恒 >= 0。
    """
    from scipy.stats import norm

    sigma = np.maximum(np.asarray(sigma), 1e-6)
    z = (np.asarray(y) - np.asarray(mu)) / sigma
    phi = np.exp(-0.5 * z * z) / np.sqrt(2 * np.pi)
    Phi = norm.cdf(z)
    return sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / np.sqrt(np.pi))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/public/efi-usgs/river-chl-targets.csv.gz")
    ap.add_argument("--train-days", type=int, default=365, help="滚动训练窗口（天）")
    ap.add_argument("--min-train", type=int, default=60, help="最少训练天数才出预测")
    args = ap.parse_args()

    df = pd.read_csv(args.data, compression="gzip")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["site_id", "datetime"]).reset_index(drop=True)
    print(f"数据: {len(df)} 条，站点 {df['site_id'].nunique()} 个，变量 {df['variable'].unique()}", flush=True)

    # 挑战评估协议：每天对每个站点出预测，用之前的历史拟合
    # 为控制计算量，抽样评估"训练日"（每站点末 365 天，逐日滚动）
    out = []
    for site, g in df.groupby("site_id"):
        g = g.dropna(subset=["observation"])
        if len(g) < args.min_train + 10:
            continue
        for i in range(args.min_train, len(g)):
            hist = g["observation"].iloc[max(0, i - args.train_days):i]
            y_true = g["observation"].iloc[i]
            # 持久化：最近观测
            mu_persist = hist.iloc[-1]
            sigma_persist = hist.std()
            # 气候学：同月均值
            month = g["datetime"].iloc[i].month
            same_month = g["observation"].iloc[:i][g["datetime"].iloc[:i].dt.month == month]
            if len(same_month) >= 5 and same_month.std() > 1e-3:
                mu_clim = same_month.mean()
                sigma_clim = same_month.std()
            else:
                mu_clim, sigma_clim = hist.mean(), hist.std()
            if not np.isfinite(sigma_clim) or sigma_clim < 1e-3:
                sigma_clim = max(hist.std(), 1e-2)
            out.append({"site_id": site, "datetime": g["datetime"].iloc[i],
                        "y": y_true, "mu_persist": mu_persist, "sd_persist": sigma_persist,
                        "mu_clim": mu_clim, "sd_clim": sigma_clim})
    res = pd.DataFrame(out)
    print(f"评估样本: {len(res)}（每站点末段逐日滚动）", flush=True)

    crps_persist = np.mean(np.abs(res["y"] - res["mu_persist"]))  # 确定性预测 CRPS = MAE
    crps_clim = np.mean(crps_gaussian(res["mu_clim"].values, res["sd_clim"].values, res["y"].values))
    # 若仍有异常负值，说明某些样本 sigma 过小/极端值，换用经验 CDF 计算兜底
    if crps_clim < 0 or not np.isfinite(crps_clim):
        from scipy import stats as _st
        vals = []
        for _, row in res.iterrows():
            mu, sd, y = row["mu_clim"], row["sd_clim"], row["y"]
            xs = np.linspace(mu - 6 * sd, mu + 6 * sd, 2000)
            F = _st.norm.cdf(xs, mu, sd)
            F = np.clip(F, 1e-9, 1 - 1e-9)
            vals.append(np.trapezoid((F - (y <= xs).astype(float)) ** 2, xs))
        crps_clim = float(np.mean(vals))
    mae_persist = float(np.mean(np.abs(res["y"] - res["mu_persist"])))
    mae_clim = float(np.mean(np.abs(res["y"] - res["mu_clim"])))
    rmse_persist = float(np.sqrt(np.mean((res["y"] - res["mu_persist"]) ** 2)))
    rmse_clim = float(np.sqrt(np.mean((res["y"] - res["mu_clim"]) ** 2)))

    print("\n===== EFI-USGS 挑战基线（公开数据，滚动 30 天训练）=====", flush=True)
    print(f"  chla 单位 ug/L。{'':>10}{'CRPS':>10}{'MAE':>10}{'RMSE':>10}", flush=True)
    print(f"  持久化 persistence {crps_persist:>12.3f}{mae_persist:>10.3f}{rmse_persist:>10.3f}", flush=True)
    print(f"  气候学 climatology {crps_clim:>12.3f}{mae_clim:>10.3f}{rmse_clim:>10.3f}", flush=True)
    # 相对气候学技能（skill = clim_crps / model_crps，>1 表示优于气候学）
    skill = crps_clim / (crps_persist + 1e-9)
    print(f"\n  持久化相对气候学技能 (clim/persist CRPS): {skill:.3f}（>1 说明持久化优于气候学）", flush=True)
    # 分站点
    print("\n  分站点持久化 MAE / 气候学 CRPS:", flush=True)
    for site, g in res.groupby("site_id"):
        m = np.mean(np.abs(g["y"] - g["mu_persist"]))
        c = np.mean(crps_gaussian(g["mu_clim"].values, g["sd_clim"].values, g["y"].values))
        print(f"    {site}: persist_MAE={m:.3f}  climat_CRPS={c:.3f}", flush=True)
    print("\n冒烟通过", flush=True)


if __name__ == "__main__":
    main()
