# -*- coding: utf-8 -*-
"""对比基线可行性：rLakeAnalyzer 分层指数 —— 温跃层深度 vs 现有分层代理

目的（可行性验证，非正式实验）：
  1. 验证算力机 R + rLakeAnalyzer 是否可用（已装：R 3.6.3 + rLakeAnalyzer 1.11.4.1）
  2. 计算温跃层深度 thermo.depth（rLakeAnalyzer::thermo.depth，只需水温剖面，不需要 bathymetry）
  3. 与我们的分层代理 delta_T / thermo_grad 做相关分析（统计量），
     判断"是否需要 R 引入该指数做特征"，还是"现有代理已等价，可降级"

保密：只输出聚合统计量（相关性），不打印原始数据行。
     中间 CSV 只在本机（算力机）生成并删除，不落地本地、不同步。
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rams.data.tensor_builder import TensorBuilder, TensorConfig  # noqa: E402

DEPTHS = [0.5 + 0.5 * i for i in range(20)]


def main():
    parquet = "data/processed/standard.parquet"
    loader = TensorBuilder(TensorConfig())
    wide = loader._load_wide(Path(parquet)).sort_index()
    daily = wide.resample("24h").mean().dropna()

    # 只取水温剖面 + 分层代理列（R 需要 wtr.name = 温度列）
    cols = [f"temp_{d}" for d in DEPTHS] + ["delta_T", "thermo_grad"]
    sub = daily[cols].reset_index()
    sub.columns = ["datetime"] + [f"wtr_{d}" for d in DEPTHS] + ["delta_T", "thermo_grad"]

    csv_path = "/tmp/rlake_input.csv"
    sub.to_csv(csv_path, index=False)
    print(f"中间表已生成 {csv_path}（{len(sub)} 天，仅水温剖面+代理，将在 R 跑完后删除）", flush=True)

    # R 脚本：计算 thermo.depth + meta.depth，输出聚合统计量
    r_script = r"""
suppressMessages(library(rLakeAnalyzer))
df <- read.csv("/tmp/rlake_input.csv", check.names=FALSE)
depths <- seq(0.5, 10.0, by=0.5)
wtr_cols <- sprintf("wtr_%.1f", depths)
# 每行转温度向量
td <- numeric(nrow(df))
n_ok <- 0L
for (i in seq_len(nrow(df))) {
  temps <- as.numeric(df[i, wtr_cols])
  td_i <- tryCatch(thermo.depth(temps, depths, Smin=0.1, seasonal=TRUE),
                   error=function(e) NA_real_, warning=function(w) NA_real_)
  if (!is.na(td_i) && td_i > 0) { td[i] <- td_i; n_ok <- n_ok + 1L }
}
cat("N_DATES_TOTAL=", nrow(df), "\n", sep="")
cat("N_THERMO_OK=", n_ok, "\n", sep="")
cat("THERMO_MEAN=", round(mean(td, na.rm=TRUE), 3), "\n", sep="")
cat("THERMO_SD=", round(sd(td, na.rm=TRUE), 3), "\n", sep="")
# 与代理的相关（只统计量）
for (proxy in c("delta_T", "thermo_grad")) {
  v <- df[[proxy]]
  mask <- !is.na(td) & td > 0 & !is.na(v)
  r <- cor(td[mask], v[mask])
  cat("CORR_", proxy, "=", round(r, 4), " N=", sum(mask), "\n", sep="")
}
# 季节相关：周数（说明温跃层季节循环，作为描述性统计）
cat("CORR_THERMO_DOY=", round(cor(td, as.numeric(strftime(as.Date(df$datetime), "%j"))), 4),
    "\n", sep="")
"""
    r_out = subprocess.run(["Rscript", "-e", r_script], capture_output=True, text=True)
    print("---- R stdout ----", flush=True)
    print(r_out.stdout, flush=True)
    if r_out.stderr.strip():
        print("---- R stderr(截断) ----", flush=True)
        print("\n".join(r_out.stderr.strip().splitlines()[-8:]), flush=True)

    # 删除中间 CSV（只留统计量）
    Path(csv_path).unlink(missing_ok=True)
    print("中间 CSV 已删除，只输出统计量", flush=True)


if __name__ == "__main__":
    main()
