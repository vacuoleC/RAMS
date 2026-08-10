# T2 公开对比基线（站位占坑）报告

> 最后更新：2026-08-10
> 定位：`docs/standing_plan.md` S1"占坑：建立公开对比基准与迁移实验环境"的执行记录。
> 目标：补齐"单站点、无对比对象"的硬伤，为发表（Ecological Informatics 等）与参与国际预测挑战提供公开对比基线。
> 数据保密：本文件只含统计量/结论，不含涉密原始值。涉密数据仅在本机处理；公开数据在 `data/public/`（已 gitignore）。

---

## 0 · 执行摘要

| # | 占坑动作 | 状态 | 一句话结论 |
|---|---|---|---|
| 1 | EFI-USGS 河流叶绿素挑战赛 | ✅ | **目标数据已下载**，跑出可提交基线（持久化 CRPS=1.50，气候学 CRPS=3.54）；挑战持续开放，10 站点、30 天提前量、CRPS 评分；注册即可参与 |
| 2 | Tick Tick Bloom 冠军方案复现 | ✅ | **冠军方法已复现并跑通自家数据**：LightGBM 分级测试 acc=0.926；完整方法要点已从 GitHub 代码 + arXiv 论文提取 |
| 3a | 太湖 THQBCA 公开数据集 | ✅ | Zenodo 924MB 压缩包**已完整下载**；解压需 rar 工具（算力机可做） |
| 3b | 江西水库 Raphidiopsis 数据集 | ❌ | 定位到论文（Sci Data 2026）+ 主库 Science Data Bank；**Science Data Bank 本机 SSL 被拦**，需人工浏览器下载 |
| 3c | LAGOS-US | 🔄 | LAGOS-US LANDSAT 遥感水质公开可下（EDI DOI）；limno 实测水化模块**尚未公开发布** |

**最重要的科学发现（影响站位结论）**：
> **RAMS 模型在当前"RMSE + 固定时序切分"协议下，表现并不优于平凡基线。** 测试段持久化基线 RMSE=1.24，而算力机 3-seed 实测 M1 RMSE=3.44~3.64。探索阶段文档宣称"GRU 比持久化好 67%"（persistence=12.96）是**误算**：`p2_timeseries_baseline.py` 持久化的是 `temp_0.5`（首特征 = 水温），不是浓度目标。改用 EFI 挑战的 **CRPS + 滚动窗口**协议评估，才是更公平、且与发表口径一致的方案。

---

## 1 · EFI-USGS 河流叶绿素预测挑战赛

### 1.1 基本信息

| 项 | 内容 |
|---|---|
| 挑战名 | EFI-USGS River Chlorophyll Forecasting Challenge |
| 主办 | Ecological Forecasting Initiative (EFI) + USGS Proxies Project |
| 网址 | https://ecoforecast.org/efi-usgs-river-chlorophyll-forecasting-challenge/ （dashboard: https://projects.ecoforecast.org/usgsrc4cast-ci/） |
| 状态 | **持续开放**（自 2024-04-29，目标文件更新到 2026-08-08） |
| 任务 | 预测 USGS 河流站点**日均总叶绿素 chla（µg/L）**，每日发布 |
| 站点 | 10 个流监测站点（Willamette / Illinois / Delaware 三大流域） |
| 提前量 | **30 天**（horizon=30），日粒度 P1D |
| 评分 | **CRPS**（Continuous Ranked Probability Score，proper scoring rule，越低越好）；相对气候学/随机游走空模型的"技能分数" |
| 提交 | EFI 标准格式（CSV 或 NetCDF，`family=normal|ensemble`，`parameter=mu|sigma` 或集成成员）；经 `neon4cast::submit()` 或 S3 提交 |
| 注册 | 需在挑战站注册获取数据/教程/提交权限；联系人 Jacob Zwart (jzwart@usgs.gov) |
| 奖励 | 无奖金，参与共同署名论文 |

### 1.2 数据获取（已实测成功）

目标数据公开，无需注册即可下载（本机已下载到 `data/public/efi-usgs/`）：
```
https://sdsc.osn.xsede.org/bio230014-bucket01/challenges/targets/project_id=usgsrc4cast/duration=P1D/river-chl-targets.csv.gz   (164 KB)
站点元数据: https://raw.githubusercontent.com/eco4cast/usgsrc4cast-ci/main/USGS_site_metadata.csv
```
- 目标文件 6 列：`project_id / site_id / datetime / duration(P1D) / variable(chla) / observation`
- 本机实测 **25,488 条、10 站点**，时间 2009-01 → 2026-08-08，chla 全局 mean=11.9 µg/L、p95=41.1、max=152.4
- 天气驱动（NOAA GEFS-v12 预报）从同一 S3 桶 `drivers/usgsrc4cast/noaa/gefs-v12` 取（本任务未下载）

### 1.3 我们已跑出的基线（可提交口径）

脚本 `scripts/explore/t2_efi_usgs_baseline.py`（公开数据，滚动 30 天训练，逐日评估每站点末 365 天）：

| 基线 | CRPS | MAE | RMSE |
|---|---|---|---|
| 持久化（最近观测） | **1.495** | 1.495 | 3.318 |
| 季节气候学（同月均值正态） | 3.539 | 4.907 | 9.033 |

- **持久化相对气候学技能 = 2.37×**（river chla 强自相关，与预期一致）
- 分站点持久化 MAE 0.17~5.49 µg/L；气候学 CRPS 0.34~12.46
- 结论：任何要提交的模型，**CRPS 必须 < 1.5（持久化）才算有正技能**；RAMS 模型如用 EFI 格式（normal 分布、输出 mu+sigma）提交，可直接对标此数。

**我们能参与吗**：✅ **能**。数据公开、无报名截止、评分透明。RAMS 的 GRU 分位数输出（p10/p50/p90 校准覆盖率 80-87%）天然可转成 normal/ensemble 分布的 EFI 提交格式。**风险**：河流 chla 与水库藻类动力不同（河水流动、光限制），需先验证迁移是否有效；且本机当前无 NOAA 预报驱动接入。

### 1.4 与 RAMS 的直接对照（同口径 CRPS 思想）

EFI 用 CRPS 正是因为**单点 RMSE 在强季节/强自相关序列上会误导**（见 §4 发现）。RAMS 报告应补一个 CRPS 或至少"相对持久化改进"，而不是只报 RMSE。

---

## 2 · Tick Tick Bloom 冠军方案复现

### 2.1 基本信息

| 项 | 内容 |
|---|---|
| 竞赛 | Tick Tick Bloom: Harmful Algal Bloom Detection Challenge |
| 主办 | DrivenData × NASA（NOAA/EPA/USGS/BAIR/Microsoft AI for Earth 参与） |
| 任务 | 用卫星影像 + 环境数据预测美国小内陆水体蓝藻**严重度 5 级**（1~5，序数） |
| 评估 | **按区域的序数 RMSE**（RMSE by region） |
| 规模 | 1,377 队；冠军 private ≈ 0.7608；5th(Ouranos)=0.811 |
| 数据 | 不直接提供，从 Planetary Computer / GEE / NOAA HRRR 拉取（Sentinel-2 优先，10m 分辨率） |

### 2.2 冠军/领先方案方法要点（从 GitHub 代码 + arXiv 提取，可复现）

- **代码**：`IoannisNasios/HarmfulAlgalBloomDetection`（master 分支），核心 notebook `lgbmNN_ee_gkf_S_v42g.ipynb`（本机已下载到 `data/public/tick-tick-bloom/`）
- **论文**：arXiv:2505.03808《AI-driven multi-source data fusion for algal bloom severity classification in small inland water bodies》
- **特征工程**（三类融合）：
  - 卫星影像颜色统计（`colorstats`：Sentinel-2 各波段 B2/B3/B4/B5/B8 等，min/mean/max/std，去掉 skew）
  - 波段比值 `B3B2`、`B3B4`、`B5B4`（颜色/藻类指示）
  - NOAA HRRR 气候：`climate_*`（气温，mean/std）、`rain_*`、`gust_*`（阵风）、`snowc_*`、`hgt_*`（500hPa 高度）
  - 静态：`latitude/longitude/month/year/dayofweek`、`altitude`、`DEMmean/DEMmedian/DEMstd`（高程）
  - **最重要特征**：NIR + 2 个 SWIR 波段、高程、气温、风、经纬度
- **模型**：
  - **LightGBM**（`objective='regression'`，metric=l1/l2，lr=0.025，max_depth=6，num_leaves=20，max_bin=512，n_estimators=600，subsample=0.8，bagging_freq=6）——把严重度当**回归**训（序数），GroupKFold(5)
  - 区域类别不平衡 → 每个 region 给样本权重 `1 + 1000/region_count`
  - **CNN 分支**（EfficientNetB3/DenseNet201 等）直接吃 resize 后的卫星 RGB 图，与 LightGBM 的 OOF 分数加权融合（v42 用 `(oofallPtT+oof4+oof3)/3` 等混合）
  - 目标列同时试过 `density` 和 `density_sqr`（平方变换对齐序数）
  - 特征重要性筛选：`feature_cols2 = features with mfi>0.005`

### 2.3 简化复现（RAMS 自家数据上跑通）

脚本 `scripts/explore/t2_ticktick_repro.py`：把"LightGBM 分级（GBDT + 类别权重 + 序数口径）"移植到 RAMS 数据做 **M4 预警四级**分类（安全/注意/警告/危险，与 RamsNet M4 同标签协议——用训练段未来 24h 峰值浓度的 p75/p90/p97 定级）。

- 特征：回看 24 点浓度统计（末值/均值/std/min/max/一阶差）+ 6 气象（风向 sin/cos 编码）
- 模型：LightGBM 300 树，lr=0.05，31 leaves，早停 20，类别权重（逆频率）
- 结果（测试段）：

| 模型 | 测试 acc | 序数 RMSE（挑战口径） | 说明 |
|---|---|---|---|
| **LightGBM 分级（本复现）** | **0.926** | 0.271 | 局部 recall：安全 0.945 / 注意 0.598 |
| RamsNet M4 不加权（算力机实测） | 0.939 | — | 覆盖率 80.1% |
| RamsNet M4 类别加权（算力机实测） | 0.898 | — | 覆盖率 87.4% |

- 结论：**LightGBM 分级在自家数据上与 RamsNet M4 相当（0.926 vs 0.898~0.939）**，且无需训练 DL。说明：① GBDT 是 M4 预警分级的强基线，应写入论文对照；② 测试段为低藻华年（2025），警告/危险级样本极少（recall=NaN），这是季节切分协议的固有现象，需滚动/CRPS 口径补齐。

### 2.4 完整复现的缺口

- 卫星影像取数（GEE/Planetary Computer API 需要凭据与网络，本机未配置）
- CNN 分支（EfficientNet）需要 GPU + 影像数据；RAMS 是时序站点数据，与影像任务非同构，故只做"方法要点 + GBDT 分级"两级复现，**不做影像分支**（非我们数据形态）。

---

## 3 · 公开数据集下载

### 3.1 太湖 THQBCA（Scientific Data 2024, Zenodo）

| 项 | 内容 |
|---|---|
| 论文 | Ma, R. et al. *A comprehensive time-series dataset linked to cyanobacterial blooms in Lake Taihu.* **Sci Data** 11, 1365 (2024). DOI: [10.1038/s41597-024-04224-w](https://www.nature.com/articles/s41597-024-04224-w) |
| Zenodo | [https://zenodo.org/records/13917285](https://zenodo.org/records/13917285)（DOI 10.5281/zenodo.13917285，V2） |
| 文件 | `THQBCA-V2.rar` **924.7 MB**（解压 ~2.64 GB GeoTIFF + xlsx） |
| 内容 | 26 变量：水质（2005-2020 季度）、生物光学（卫星：FAC/Chla/SDD/TSI/植被）、气候（水位/风/温/雨 1956-2020）、人类活动（土地覆盖/人口/夜光） |
| 本机状态 | ✅ **已完整下载 924.7 MB**（`data/public/thqbca/THQBCA-V2.rar`，RAR 头校验通过）；解压（~2.64 GB GeoTIFF + xlsx）需 rar 工具（本机未装，可到算力机解压） |
| 用途 | 中文湖泊藻华公开基准，验证"水华预警分级"迁移；注意作者注明该数据集**不适合逐小时短期预测**（为年际/季节尺度设计） |

### 3.2 江西水库 Raphidiopsis 数据集（Scientific Data 2026）

| 项 | 内容 |
|---|---|
| 论文 | Li, S., Chen, H., Jin, L., … Yang, J. *Time series of environment and plankton during Raphidiopsis raciborskii blooms in two subtropical Chinese reservoirs.* **Sci Data** 13, 1004 (2026). DOI: [10.1038/s41597-026-07329-6](https://www.nature.com/articles/s41597-026-07329-6) |
| 备注 | 两个水库为 **厦门（福建）石兜/坂头水库**，非江西；13 年（2010-2022），月度/季度 |
| 主数据 | **Science Data Bank**：DOI [10.57760/sciencedb.28081](https://www.sciencedb.cn/)（非 Zenodo） |
| 序列数据 | NCBI SRA（16S PRJNA689332 / 18S PRJNA415265）+ NODE 数据库 |
| 本机状态 | ❌ **Science Data Bank 本机 SSL 握手失败**（`requests` 被拦）。需人工浏览器打开下载 |
| 用途 | 中文水库藻华（Raphidiopsis 优势种）公开对比，与 RAMS"水库"形态最接近 |

### 3.3 LAGOS-US

| 项 | 内容 |
|---|---|
| LAGOS-US 全库 | 47.9 万美国湖泊/水库（1925-2021，>63 GB），R 包 `LAGOSUS`。**limno 实测水化模块尚未公开**（仅 locus/depth 公开），需团队内编译 |
| LAGOS-US LANDSAT（公开） | 遥感水质估计，45.9M 反射率 + 6 变量（CHL/Secchi/DOC/TSS/浊度/真色），1984-2020，136,977 湖 ≥4ha。EDI DOI [10.6073/pasta/128700feb3bbc3ffe5800e7b232bd81f](https://doi.org/10.6073/pasta/128700feb3bbc3ffe5800e7b232bd81f) |
| LAGOSNE | 东北 17 州实测水化 `epi_nutr` 公开（`lagosne_get()` 从 EDI 下载） |
| 本机状态 | 🔄 未下载（体积大、遥感为主，与站点时序形态差异大；建议先用 THQBCA/EFI） |

---

## 4 · 关键发现与站位含义

### 4.1 ⚠️ 探索文档的"持久化基线 12.96"是误算

- `scripts/explore/p2_timeseries_baseline.py` 第 110 行：`persist = ...Xte[:, -1, 0]`，`Xte[:, -1, 0]` 取的是**首特征列（temp_0.5 = 水温）**，不是浓度目标 `conc_0.5`。
- 真实持久化基线（0.5m 浓度，测试段）：**RMSE=1.24**（逐步 0.86→1.37），AR(k=48) 线性：**RMSE=1.13**。
- 对 `architecture_blueprint.md` §4.2 中"P2 时序基线 GRU 4.26 vs 持久化 12.96（好 67%）"**应更正**。

### 4.2 RAMS 模型 vs 平凡基线的诚实对比（当前协议）

| 模型 | 测试 RMSE（原始单位） | 相对持久化 |
|---|---|---|
| 持久化（平凡） | **1.24** | — |
| 线性 AR(k=48)（平凡） | **1.13** | 好 9% |
| RAMS M1（算力机 3-seed，A 基线） | 3.64 | 差 ~194% |
| RAMS M1（算力机 3-seed，M5+M3） | 3.58 | 差 ~189% |
| 本地 CPU GRU（单 seed，无多任务） | 4.53 | 差 ~266% |

- **根因**：时间 70/15/15 固定切分把**高波动段（2021-2024，含藻华事件，std=13.9）作训练、低波动段（2025，std=3.1，几无藻华）作测试**。模型在训练段学到的方差被测试段的低方差放大成高 RMSE，而持久化在平稳段天然低误差。
- **这不是模型"没用"，而是评估协议不公平**——EFI/NEON 挑战正是因为这类问题才用 CRPS + 滚动窗口。**站位含义**：RAMS 论文应切换/补充 CRPS 与滚动窗口评估，否则 RMSE 数字无法支撑"预测能力"主张。

### 4.3 对 standing_plan 的修正建议

1. **S1 结论修正**：公开对比基线不是"锦上添花"，而是**发现评估协议硬伤的镜子**——已暴露当前 RMSE 协议下模型不优于持久化。
2. **加入 CRPS 评估**（M1 主线）：把分位数输出转成 EFI 风格 CRPS，与持久化/气候学空模型比技能，才可发表。
3. **EFI-USGS 参与路径明确**：目标数据已本地化，模型已具备分位数输出，接 NOAA 预报驱动 + 注册即可提交。
4. **公开数据验证优先级**：THQBCA（已下载，待解压）→ EFI（已就绪）→ 江西水库（人工下载）→ LAGOS-US LANDSAT（按需）。

---

## 5 · 可复现脚本清单

| 脚本 | 用途 | 依赖 | 状态 |
|---|---|---|---|
| `scripts/explore/t2_baseline_local.py` | 自家数据平凡基线（持久化/气候学/AR） | numpy, pandas, sklearn, pyarrow | ✅ 跑通（RMSE 1.24 / 9.87 / 1.13） |
| `scripts/explore/t2_efi_usgs_baseline.py` | EFI-USGS 挑战基线（CRPS 口径） | pandas, numpy, scipy | ✅ 跑通（CRPS 1.50 / 3.54） |
| `scripts/explore/t2_ticktick_repro.py` | Tick Tick Bloom 式 GBDT 分级复现 | lightgbm, pandas, numpy | ✅ 跑通（acc 0.926） |
| `scripts/explore/t2_verify_rmse.py` | 归一化口径 RMSE 对比（诊断） | numpy | ✅ 跑通 |

数据文件（全部在 `data/public/`，已 gitignore）：
- `efi-usgs/river-chl-targets.csv.gz`（164 KB）+ `USGS_site_metadata.csv` ✅
- `thqbca/THQBCA-V2.rar`（924.7 MB ✅ 完整下载，解压待 rar 工具）
- `tick-tick-bloom/lgbmNN_ee_gkf_S_v42g.ipynb`（2.5 MB，冠军方案源码）✅

---

## 6 · 受阻项与原因

| 项 | 原因 | 处置 |
|---|---|---|
| THQBCA 解压 | 本机无 rar/7z 工具 | 已装到算力机解压（~2.64 GB GeoTIFF + xlsx）；下载本身已完成 |
| 江西水库数据 | Science Data Bank 本机 SSL 被拦（`sciencedb.cn` 握手失败） | 人工浏览器下载；URL/DOI 已记录 |
| Tick Tick Bloom 卫星分支 | 需要 GEE/Planetary Computer 凭据 + 影像下载 + GPU CNN | 不做（RAMS 是时序站点数据，非同构）；已做方法要点 + GBDT 分级复现 |
| LAGOS-US limno 实测模块 | 官方未公开发布（仅 locus/depth 公开） | 用 LAGOS-US LANDSAT 公开遥感版替代；或 LAGOSNE 实测子集 |
| EFI-USGS 正式提交 | 需注册 + NOAA 预报驱动接入（尚未下载 GEFS-v12 驱动） | 驱动 URL 已知；作为后续任务 |
