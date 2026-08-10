# -*- coding: utf-8 -*-
"""对比基线：rLakeAnalyzer 温跃层深度 —— 日级全序列计算 + 协议内非冗余性验证

（mdl-baseline-compare · st-baseline-run 第 2 步）

目的：
  1. 在日级网格（与正式协议一致：1D 均值聚合）上计算每日温跃层深度
     thermo.depth（rLakeAnalyzer::thermo.depth，只需水温剖面，不需要 bathymetry）。
  2. 非冗余性验证（协议内）：thermo.depth 与现有分层代理 delta_T / thermo_grad
     的统计相关，以及它与目标 conc_0.5 / 藻华状态 / 未来 7 天目标的相关。
  3. 输出日级 thermo.depth 序列到算力机 /tmp（供 run_ml_thermo_ablation.py 做
     特征消融实验），中间文件只留在算力机、不同步回本地。

保密红线：只输出聚合统计量（相关 / N / 均值 / 标准差），不打印原始数据行；
     /tmp/rlake_daily.csv 只含 [date, thermo_depth] 两个派生物理量列，
     算力机跑完即用（特征消融），不落地本地、不同步。

用法（算力机 sensecore）：
  PYTHONPATH=/data/RAMS/proj python3 exp/baseline_feasibility/rlake_daily_thermo.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rams.data.tensor_builder import TensorBuilder, TensorConfig, BloomLabeler, DailyConfig  # noqa: E402

DEPTHS = [0.5 + 0.5 * i for i in range(20)]
CSV_IN = "/tmp/rlake_input.csv"
CSV_THERMO = "/tmp/rlake_daily.csv"  # date, thermo_depth（算力机本地，不同步）
OUT_JSON = "exp/baseline_feasibility/rlake_daily_thermo.json"


def main() -> None:
    parquet = "data/processed/standard.parquet"
    loader = TensorBuilder(TensorConfig())
    wide = loader._load_wide(Path(parquet)).sort_index()
    daily = wide.resample("1D").mean().dropna()
    print(f"日级宽表 {len(daily)} 天  {daily.index.min():%Y-%m-%d} → {daily.index.max():%Y-%m-%d}", flush=True)

    # 只取水温剖面 + 代理列（R 需要 wtr.name = 温度列）
    cols = [f"temp_{d}" for d in DEPTHS] + ["delta_T", "thermo_grad", "conc_0.5"]
    sub = daily[cols].reset_index()
    sub.columns = ["datetime"] + [f"wtr_{d}" for d in DEPTHS] + ["delta_T", "thermo_grad", "conc_0.5"]
    sub.to_csv(CSV_IN, index=False)
    print(f"中间表已生成 {CSV_IN}（{len(sub)} 天，仅水温剖面+代理，跑完即删）", flush=True)

    # R 脚本：逐日 thermo.depth + 输出日级序列 + 聚合统计量
    r_script = r"""
suppressMessages(library(rLakeAnalyzer))
df <- read.csv("/tmp/rlake_input.csv", check.names=FALSE)
depths <- seq(0.5, 10.0, by=0.5)
wtr_cols <- sprintf("wtr_%.1f", depths)
td <- numeric(nrow(df)) * NA_real_
n_ok <- 0L
for (i in seq_len(nrow(df))) {
  temps <- as.numeric(df[i, wtr_cols])
  td_i <- tryCatch(thermo.depth(temps, depths, Smin=0.1, seasonal=TRUE),
                   error=function(e) NA_real_, warning=function(w) NA_real_)
  if (!is.na(td_i) && td_i > 0) { td[i] <- td_i; n_ok <- n_ok + 1L }
}
# 写日级序列（date, thermo_depth）—— 供特征消融，仅算力机本地
out <- data.frame(date = df$datetime, thermo_depth = td)
write.csv(out, "/tmp/rlake_daily.csv", row.names=FALSE)
cat("N_TOTAL=", nrow(df), "\n", sep="")
cat("N_THERMO_OK=", n_ok, "\n", sep="")
cat("THERMO_MEAN=", round(mean(td, na.rm=TRUE), 3), "\n", sep="")
cat("THERMO_SD=", round(sd(td, na.rm=TRUE), 3), "\n", sep="")
for (proxy in c("delta_T", "thermo_grad", "conc_0.5")) {
  v <- df[[proxy]]
  mask <- !is.na(td) & td > 0 & !is.na(v)
  r <- cor(td[mask], v[mask])
  cat("CORR_", proxy, "=", round(r, 4), " N=", sum(mask), "\n", sep="")
}
# 季节相关：只在可算日上（与 Python 侧 mask 一致）
cat("CORR_THERMO_DOY=", round(cor(td, as.numeric(strftime(as.Date(df$datetime), "%j")), use="complete.obs"), 4), "\n", sep="")
"""
    r_out = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)
    print("---- R stdout ----", flush=True)
    print(r_out.stdout, flush=True)
    if r_out.stderr.strip():
        print("---- R stderr(截断) ----", flush=True)
        print("\n".join(r_out.stderr.strip().splitlines()[-8:]), flush=True)
    Path(CSV_IN).unlink(missing_ok=True)

    # ---- 协议内非冗余性 + 预测性（Python 侧，只出统计量）----
    th = pd.read_csv(CSV_THERMO, parse_dates=["date"]).set_index("date")["thermo_depth"]
    daily["thermo_depth"] = th
    # 藻华状态（整集拟合，N 定义）—— 只作相关性描述
    bloom_full = BloomLabeler(config=DailyConfig()).predict(daily)
    mask = daily["thermo_depth"].notna() & (daily["thermo_depth"] > 0)
    sub = daily[mask]
    stats = {
        "n_dates_total": int(len(daily)),
        "n_thermo_ok": int(mask.sum()),
        "thermo_mean": float(sub["thermo_depth"].mean()),
        "thermo_sd": float(sub["thermo_depth"].std()),
        "corr_thermo_delta_T": float(np.corrcoef(sub["thermo_depth"], sub["delta_T"])[0, 1]),
        "corr_thermo_thermo_grad": float(np.corrcoef(sub["thermo_depth"], sub["thermo_grad"])[0, 1]),
        "corr_thermo_conc0_5": float(np.corrcoef(sub["thermo_depth"], sub["conc_0.5"])[0, 1]),
        "corr_thermo_bloom": float(np.corrcoef(sub["thermo_depth"], bloom_full[mask])[0, 1]),
        "corr_thermo_doy": float(np.corrcoef(sub["thermo_depth"], sub.index.dayofyear)[0, 1]),
    }
    # 预测性（数据-only，无泄漏：thermo.depth_t vs 未来 7 天 conc）
    sub7 = daily.dropna(subset=["thermo_depth"]).copy()
    sub7["conc7"] = sub7["conc_0.5"].shift(-7)
    mm = sub7[["thermo_depth", "delta_T", "thermo_grad", "conc7"]].dropna()
    stats["corr_thermo_t_conc7"] = float(np.corrcoef(mm["thermo_depth"], mm["conc7"])[0, 1])
    stats["corr_deltaT_t_conc7"] = float(np.corrcoef(mm["delta_T"], mm["conc7"])[0, 1])
    stats["corr_thermo_grad_t_conc7"] = float(np.corrcoef(mm["thermo_grad"], mm["conc7"])[0, 1])

    print("\n===== 协议内非冗余性统计 =====", flush=True)
    for k, v in stats.items():
        print(f"  {k:<28} {v:.4f}" if isinstance(v, float) else f"  {k:<28} {v}", flush=True)

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUT_JSON).write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 统计量已写入 {OUT_JSON}（未含任何原始数据行）", flush=True)
    print(f"日级 thermo.depth 序列已存 {CSV_THERMO}（算力机本地，供特征消融，不同步回本地）", flush=True)


if __name__ == "__main__":
    main()
