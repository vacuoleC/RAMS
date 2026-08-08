# M5: 气象-藻类 机理与时滞分析（PCMCI+，优化版）

- 数据: `standard.parquet`，降采样 12h 网格（均值聚合），3350 时刻（2021-03-01 00:00:00 ~ 2025-09-30 12:00:00）
- 方法: PCMCI+（tigramite parcorr），tau_max=60 （30 天），alpha=0.05
- ACF 定滞后上界: {'conc_surf': 149, 'cyano_surf': 179}（|ACF|<0.035），tau_max = min(192, 2*k) = 60
- 候选边剪枝: 全图初始 8640 条 → 扫描 2400 条 → PCMCI+ 运行于 全候选空间
- 约束: max_conds_dim=5, max_conds_px=5, max_conds_py=5
- 运行耗时: 7.4 分钟
- 变量: 12 个（季节编码 + 藻类 + 水温 + 气象）
- 目标: conc_surf, cyano_surf, temp_surf, temp_mean

## 显著因果边（变量 / 时滞 / MCI 相关 / p）

| from | to | τ(步) | 滞后 | MCI r | p |
|---|---|---:|---:|---:|---:|
| conc_surf | conc_surf | 1 | 0.5d | +0.619 | 0.0e+00 |
| temp_surf | temp_surf | 1 | 0.5d | +0.594 | 2.5e-303 |
| temp_mean | temp_mean | 1 | 0.5d | +0.590 | 3.5e-298 |
| cyano_surf | cyano_surf | 1 | 0.5d | +0.567 | 5.1e-271 |
| cyano_surf | cyano_surf | 2 | 1.0d | +0.373 | 2.5e-104 |
| temp_surf | temp_surf | 2 | 1.0d | +0.359 | 1.2e-95 |
| temp_mean | temp_mean | 2 | 1.0d | +0.309 | 9.2e-70 |
| temp_surf | temp_surf | 3 | 1.5d | -0.276 | 9.3e-55 |
| cyano_surf | cyano_surf | 5 | 2.5d | -0.241 | 2.1e-41 |
| temp_surf | temp_surf | 6 | 3.0d | +0.190 | 4.3e-25 |
| cyano_surf | cyano_surf | 10 | 5.0d | +0.174 | 5.0e-21 |
| temp_surf | temp_surf | 5 | 2.5d | -0.172 | 1.6e-20 |
| cyano_surf | cyano_surf | 14 | 7.0d | +0.164 | 1.5e-18 |
| temp_surf | temp_surf | 8 | 4.0d | +0.153 | 4.5e-16 |
| cyano_surf | cyano_surf | 8 | 4.0d | +0.147 | 7.4e-15 |
| temp_surf | temp_surf | 59 | 29.5d | -0.145 | 1.9e-14 |
| conc_surf | conc_surf | 2 | 1.0d | +0.144 | 3.6e-14 |
| temp_surf | temp_mean | 3 | 1.5d | -0.134 | 3.6e-12 |
| cyano_surf | cyano_surf | 22 | 11.0d | +0.129 | 2.7e-11 |
| temp_surf | temp_surf | 27 | 13.5d | -0.128 | 4.6e-11 |
| wind_u | temp_surf | 2 | 1.0d | +0.125 | 1.1e-10 |
| temp_surf | temp_surf | 11 | 5.5d | -0.119 | 1.2e-09 |
| temp_surf | temp_surf | 19 | 9.5d | -0.114 | 6.9e-09 |
| temp_surf | temp_surf | 15 | 7.5d | -0.114 | 8.3e-09 |
| cyano_surf | cyano_surf | 40 | 20.0d | +0.113 | 9.5e-09 |
| temp_surf | temp_surf | 33 | 16.5d | -0.113 | 9.7e-09 |
| temp_surf | temp_surf | 31 | 15.5d | -0.110 | 2.9e-08 |
| temp_surf | temp_surf | 39 | 19.5d | -0.110 | 3.2e-08 |
| cyano_surf | cyano_surf | 26 | 13.0d | +0.109 | 4.0e-08 |
| temp_surf | temp_surf | 41 | 20.5d | -0.109 | 4.3e-08 |
| temp_surf | temp_surf | 9 | 4.5d | -0.104 | 2.0e-07 |
| cyano_surf | cyano_surf | 34 | 17.0d | +0.103 | 2.8e-07 |
| temp_mean | temp_mean | 11 | 5.5d | -0.103 | 2.9e-07 |
| cyano_surf | cyano_surf | 38 | 19.0d | +0.100 | 7.5e-07 |
| cyano_surf | cyano_surf | 30 | 15.0d | +0.100 | 8.9e-07 |
| cyano_surf | cyano_surf | 42 | 21.0d | +0.097 | 1.8e-06 |
| cyano_surf | cyano_surf | 24 | 12.0d | +0.097 | 2.0e-06 |
| temp_surf | temp_surf | 35 | 17.5d | -0.096 | 2.3e-06 |
| temp_mean | temp_mean | 59 | 29.5d | -0.096 | 2.4e-06 |
| temp_surf | temp_mean | 33 | 16.5d | -0.096 | 2.8e-06 |
| cyano_surf | cyano_surf | 20 | 10.0d | +0.095 | 3.6e-06 |
| cyano_surf | cyano_surf | 32 | 16.0d | +0.094 | 4.8e-06 |
| temp_mean | temp_surf | 3 | 1.5d | -0.093 | 6.0e-06 |
| temp_mean | temp_mean | 60 | 30.0d | +0.089 | 1.6e-05 |
| temp_mean | temp_mean | 55 | 27.5d | -0.088 | 2.4e-05 |
| temp_surf | temp_surf | 7 | 3.5d | -0.087 | 3.3e-05 |
| temp_mean | temp_mean | 21 | 10.5d | -0.086 | 4.0e-05 |
| conc_surf | conc_surf | 10 | 5.0d | +0.083 | 9.1e-05 |
| temp_mean | temp_mean | 7 | 3.5d | -0.082 | 1.2e-04 |
| temp_surf | temp_surf | 17 | 8.5d | -0.079 | 2.6e-04 |
| temp_mean | temp_mean | 15 | 7.5d | -0.078 | 2.7e-04 |
| temp_mean | temp_mean | 33 | 16.5d | -0.078 | 2.9e-04 |
| temp_surf | temp_surf | 29 | 14.5d | -0.078 | 3.1e-04 |
| temp_mean | temp_mean | 58 | 29.0d | +0.077 | 3.5e-04 |
| cyano_surf | cyano_surf | 28 | 14.0d | +0.077 | 3.9e-04 |
| temp_surf | temp_surf | 13 | 6.5d | -0.076 | 4.6e-04 |
| temp_mean | temp_mean | 27 | 13.5d | -0.076 | 5.2e-04 |
| temp_mean | temp_mean | 19 | 9.5d | -0.076 | 5.2e-04 |
| conc_surf | conc_surf | 3 | 1.5d | -0.076 | 5.4e-04 |
| temp_mean | temp_mean | 51 | 25.5d | -0.075 | 5.9e-04 |
| temp_mean | temp_mean | 5 | 2.5d | -0.075 | 6.0e-04 |
| temp_mean | temp_mean | 30 | 15.0d | +0.075 | 6.4e-04 |
| temp_mean | temp_mean | 41 | 20.5d | -0.074 | 7.0e-04 |
| temp_mean | temp_mean | 17 | 8.5d | -0.074 | 7.4e-04 |
| wind_u | cyano_surf | 1 | 0.5d | +0.072 | 1.1e-03 |
| wind_u | temp_surf | 4 | 2.0d | +0.072 | 1.1e-03 |
| wind_u | temp_mean | 3 | 1.5d | -0.072 | 1.2e-03 |
| temp_surf | temp_mean | 31 | 15.5d | -0.072 | 1.3e-03 |
| temp_mean | temp_surf | 33 | 16.5d | -0.070 | 1.8e-03 |
| temp_mean | temp_mean | 28 | 14.0d | +0.068 | 2.6e-03 |
| cyano_surf | cyano_surf | 36 | 18.0d | +0.068 | 2.7e-03 |
| wind_u | temp_mean | 12 | 6.0d | +0.067 | 3.2e-03 |
| temp_mean | temp_mean | 34 | 17.0d | +0.067 | 3.2e-03 |
| temp_mean | temp_mean | 9 | 4.5d | -0.066 | 3.9e-03 |
| wind_u | temp_mean | 1 | 0.5d | -0.065 | 5.4e-03 |
| temp_mean | temp_mean | 36 | 18.0d | +0.063 | 7.0e-03 |
| temp_surf | temp_mean | 29 | 14.5d | -0.062 | 8.5e-03 |
| temp_mean | temp_mean | 31 | 15.5d | -0.062 | 8.6e-03 |
| humidity | temp_surf | 3 | 1.5d | +0.058 | 1.8e-02 |
| temp_mean | temp_mean | 23 | 11.5d | -0.057 | 2.2e-02 |
| temp_mean | temp_mean | 56 | 28.0d | +0.057 | 2.4e-02 |
| temp_mean | temp_mean | 39 | 19.5d | -0.056 | 2.6e-02 |
| temp_mean | temp_mean | 13 | 6.5d | -0.055 | 3.1e-02 |
| temp_mean | temp_mean | 35 | 17.5d | -0.054 | 3.6e-02 |
| wind_u | cyano_surf | 33 | 16.5d | +0.054 | 4.1e-02 |
| temp_mean | temp_mean | 49 | 24.5d | -0.053 | 4.2e-02 |
| temp_mean | temp_mean | 57 | 28.5d | -0.053 | 4.9e-02 |

## 说明
- 保密：仅报告变量名/时滞/统计量，不含原始数据值。
- 滞后 = τ×12h；1 天 = 2 个 τ。
- 文献参考滞后（降水 13-20d、风 20-29d、气温 25-30d）。
