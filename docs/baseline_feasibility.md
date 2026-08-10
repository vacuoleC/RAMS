# 对比基线可行性验证（frozen-design 前决策依据）

> 日期：2026-08-10
> 目的：确认对比基线任务（`docs/standing_plan.md` M1"GP/KNN/XGBoost 全纳入 baseline"）中，
> 哪些基线能在算力机 sensecore 上跑，哪些跑不了（需降级方案），供 frozen-design 冻结前决策。
> 数据保密：本文件只含统计量 / CRPS / 技能 / 结论，不含任何涉密原始数据行。

---

## 1. 做了什么（执行记录）

| # | 动作 | 结果 |
|---|---|---|
| 1 | 环境摸底：sensecore（H100 80GB，Ubuntu 20.04，Python 3.8，root） | 无 R / 无 gfortran / 无 conda；sklearn 1.3.2 + xgboost 2.0.3 + lightgbm 4.1.0 可用；sudo+apt 可用；CRAN / GitHub 可达 |
| 2 | 统计 ML 基线：`exp/baseline_feasibility/run_ml_baselines.py`，8 个基线按正式协议（日级 1D 均值 + 滚动 730/90/45 + CRPS）跑 17 窗口 | ✅ 已跑通（冒烟 1 窗口 → 全量 17 窗口，本机 + 算力机）；结果见 §3 |
| 3 | 试装 R：`apt-get install r-base-core`（≤5 min） | ✅ **成功**：R 3.6.3（`/usr/bin/Rscript`，安装约 1 分钟） |
| 4 | 试装 rLakeAnalyzer：`install.packages`（CRAN，≤5 min） | ✅ **成功**：rLakeAnalyzer 1.11.4.1 |
| 5 | rLakeAnalyzer 实证：`exp/baseline_feasibility/run_rlake_demo.py`（温跃层深度 vs 现有分层代理） | ✅ **可算且非冗余**：thermo.depth 可用 1054/1638 天；与 delta_T 相关 r=0.40、与 thermo_grad r=-0.39（仅中等，含独立信息）；见 §4 |
| 6 | 试装 gfortran：`apt-get install gfortran` | ✅ **成功**：gfortran 9.4.0（`/usr/bin/gfortran`，安装约 1 分钟）；libnetcdf-dev / r-cran-rjags 亦在 apt 可用 |
| 7 | GLM-AED / CE-QUAL-W2 可编译性判断 | ⚠️ 工具链缺口已消除，但**缺 bathymetry（水库几何）数据** —— 决定性不可行，见 §5 |
| 8 | FLARE 判断 | ❌ R 环境缺失（现可装但耗时长）且**同样缺 bathymetry + 湖表/深水温度场驱动**，见 §5 |

---

## 2. 各基线可行性状态总表

| 基线 | 类型 | 可行性 | 证据 | 降级/替代 |
|---|---|---|---|---|
| 持久化（当前浓度当未来 7 天） | 统计 ML | ✅ 可跑 | 已按正式协议跑出 17 窗口 CRPS | — |
| 线性 AR / Ridge（过去 N 天浓度） | 统计 ML | ✅ 可跑 | 同上 | — |
| XGBoost / LightGBM（过去 7 天窗口→未来 7 天峰值/均值） | 统计 ML | ✅ 可跑 | 同上（含分位数回归口径） | — |
| rLakeAnalyzer 分层指数（thermo.depth 等） | 物理指数（R） | ✅ **可跑**（R+rLakeAnalyzer 已装） | 实测 thermo.depth 1054/1638 天；与代理中相关 | 不做也已有 delta_T / thermo_grad 代理（相关 0.4，代理≠等价，thermo.depth 有增量信息） |
| GLM-AED | 物理过程（Fortran） | ⚠️ 工具链可装但**数据不可行** | gfortran 已装，但**无 bathymetry 数据** | 降级：借鉴"分层/混合物理量做特征"，不跑完整物理模拟 |
| CE-QUAL-W2 | 物理过程（Fortran） | ❌ 不可行（同 GLM） | 需 bathymetry + 气驱动 + 初始/边界；无几何数据 | 同上；且 M3 曾计划用其"生成公开预训练数据"，需公开水库几何（不在本次范围） |
| FLARE | 物理-数据同化（R+Python） | ❌ 不可行 | 需 R + rjags/ncdf4 + bathymetry + 温度场驱动；缺几何数据 | 降级：借鉴"预报-同化滚动更新 + 概率集合"思想到统计/ML 预报协议 |
| Qiu 巢湖框架 / Pontchartrain 预警 | 方法借鉴 | ✅ 不跑代码，协议内复现思路 | 方法类，可在我们的协议里做"阈值预警 vs 概率预警对比" | 已在文档 §6 给出复现路径 |

---

## 3. 统计 ML 基线 —— 正式协议 CRPS 对比表（算力机全量 17 窗口）

**协议**：日级 1D 均值聚合；滚动窗口训练 730d / 测试 90d / 步长 45d（17 窗口）；目标 = 未来 7 天表层浓度 conc_0.5（原始浓度单位）；预测 = 分位数 p10/p50/p90；退化分布 CRPS=MAE（与 RamsNet/B7/D/L 逐视界 CRPS 退化口径一致）；指标 = CRPS（越低越好）+ p50 RMSE + [p10,p90] 覆盖率 + 相对持久化技能。

**基线定义**：持久化 = 当前浓度当未来 7 天；ar_ridge = Ridge（过去 30 天浓度/温度/气象统计→多输出 7 天）；xgb_abs/lgb_abs = GBDT 绝对量多输出（MultiOutput，每视界独立树，退化分布）；xgb_peak/lgb_peak = GBDT 预测 7 天峰值（任务口径）；xgb_q/lgb_q = GBDT **分位数回归**（alpha=0.1/0.5/0.9，逐视界），校准口径。

（注：冒烟 1 窗口与全量 17 窗口数值不同——全量以 §3 表为准。）

| 基线 | CRPS | 持久化 CRPS | 技能 vs 持久化 | p50 RMSE | 覆盖率[p10,p90] |
|---|---:|---:|---:|---:|---:|
| persist | **1.5428** | 1.5428 | — | 2.124 | 0.0% |
| ar_ridge | 1.5594 | 1.5428 | −1.1% | 2.078 | 0.0% |
| xgb_abs | 2.0669 | 1.5428 | −34.0% | 2.766 | 0.0% |
| lgb_abs | 1.9585 | 1.5428 | −26.9% | 2.674 | 0.0% |
| xgb_peak | 2.4615 | 1.5428 | −59.6% | 3.094 | 0.0% |
| lgb_peak | 2.4578 | 1.5428 | −59.3% | 3.049 | 0.0% |
| xgb_q | 1.4724 | 1.5428 | +4.6% | 2.624 | 57.6% |
| lgb_q | **1.3891** | 1.5428 | **+10.0%** | 2.503 | 55.4% |

> 读数：
> 1. **GBDT 的"分位数回归"口径（xgb_q / lgb_q）是唯一相对持久化有正技能的统计基线**（+4.6% / +10.0%），因为它输出概率（覆盖率 55-58%）；而确定性点基线（Ridge / xgb_abs / lgb_abs / xgb_peak / lgb_peak）全部 ≤ 持久化（退化分布 CRPS=|y−ŷ|，均值/峰值回归对 7 天慢扩散的目标不利）。
> 2. **峰值任务口径（xgb_peak / lgb_peak）最差**（−59%）：把"未来 7 天峰值"当单点回归，对藻华高方差目标几乎不可预测，且退化分布把不确定性压成 0——**峰值必须配概率/分级口径，不能退化分布**。
> 3. 全部统计基线覆盖率 [p10,p90] 远低于 80% 目标（退化分布 0%，分位数口径 55-58%）→ **统计 ML 无自带校准**，只有 RamsNet 分位数（L 探索覆盖 78%）能到 80% 附近。这支持"正式模型保留概率输出 + 统计 ML 作点精度对照"的定位。
> 4. **逐视界**（见 §3.1）：lgb_q 相对持久化的技能随视界**增大**——h1 次日为负（−32%，短视界当前浓度几乎就是最优），h3 起转正并一路升到 h7 +18%。这与 3h 尺度 RamsNet 不同（t4 里 3h 长视界持久化会回升）：日级 7 天目标下持久化随 h 单调恶化（h7 CRPS 2.005），概率基线在长视界的优势反而扩大。

### 3.1 逐视界 CRPS 与技能（h1=次日 … h7=第 7 天）

| 基线 | h1 | h2 | h3 | h4 | h5 | h6 | h7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| persist | 0.806 | 1.208 | 1.476 | 1.651 | 1.760 | 1.894 | 2.005 |
| ar_ridge | 0.835 | 1.249 | 1.507 | 1.665 | 1.769 | 1.895 | 1.995 |
| lgb_q | 1.068 | 1.213 | 1.328 | 1.418 | 1.490 | 1.569 | 1.637 |
| xgb_q | 1.217 | 1.332 | 1.426 | 1.497 | 1.552 | 1.613 | 1.670 |
| lgb_q 技能 vs persist | −32.6% | −0.4% | **+10.0%** | **+14.1%** | **+15.3%** | **+17.2%** | **+18.4%** |

（lgb_q 逐视界技能 = (persist_h − lgb_q_h)/persist_h；h1 次日负、h3 起转正，长视界概率口径优势更明显。）

**对照参考（同协议，L 探索已实测，RamsNet 正式模型）**：

| 模型 | CRPS | 持久化 CRPS | 技能 vs 持久化 | 说明 |
|---|---:|---:|---:|---|
| RamsNet（1D，3-seed 17 窗口） | 1.183 | 1.543 | **+23.4%** | `exp/model_enhancement/l_daily_scale/results.json` |

---

## 4. rLakeAnalyzer —— 可装可用，指数有增量信息

- **可行性**：R 3.6.3 + rLakeAnalyzer 1.11.4.1 已装好（各 ≤1 min），**完全可跑**。
- **实证（`run_rlake_demo.py`，只出统计量）**：
  - `thermo.depth`（温跃层深度，rLakeAnalyzer::thermo.depth，seasonal=True）在 1638 天中 **1054 天可算**（64%），均值 5.3m、SD 4.2m；
  - 与现有分层代理的统计相关（**仅中等，非冗余**）：thermo.depth vs delta_T **r=0.40**、vs thermo_grad **r=-0.39**；
  - 即：**rLakeAnalyzer 能跑，且温跃层深度 ≠ 我们的 delta_T/thermo_grad 代理**，直接做特征有增量信息。
- **判定**：作为 M2 分层状态基线/特征**可进正式项目**；若想省 R 依赖，也可用现有代理降级（但不推荐完全等价的替代——相关仅 0.4）。
- 备注：meta.depth / Schmidt 稳定性（L&O 标准）均可由同温度剖面算出（不需 bathymetry）；**需 bathymetry 的指标（如 Schmidt stability 的严格体积加权版）受限于数据，只能做近似**。

---

## 5. 物理模型 —— 诚实结论：数据不可行（非工具链）

### 5.1 GLM-AED / CE-QUAL-W2（Fortran）

- **工具链**：原本无 gfortran（决定性）；本次实测 `apt-get install gfortran` **已成功**（9.4.0，约 1 min），libnetcdf-dev 亦在 apt 可用（GLM 可选依赖）。即 Fortran 编译缺口**已消除**。
- **决定性障碍 = 数据**：GLM 和 CE-QUAL-W2 都是**一维垂向/二维水动力模型，强制需要 bathymetry（库区几何：水深-面积-体积 / hypsographic 曲线）**。经全库检索（`/data/RAMS` 及仓库），**项目没有任何 bathymetry / 库容 / 高程-面积数据**。没有库区几何，GLM/CE-QUAL-W2 **无法初始化、无法跑**——这不是编译问题，是输入数据缺失。
- **结论**：**❌ 不可行（数据层面）**，即使能编译。诚实表述：不应为了"物理基线"而去下载公开水库几何（与涉密水库不符，且算出的结果对"本项目水库"无意义）。

### 5.2 FLARE（R + Python 预报-同化）

- FLARE 是 EFI 生态的湖泊生态预报框架（数据同化 + 集合预报），需要：R 环境 + rjags/ncdf4/GLM 底层模拟 + bathymetry + 湖表/深水温度场初始化驱动。
- **障碍**：R 现可装（本次已装 3.6.3），但 FLARE 对 R 包版本有要求（focal 3.6.3 偏旧，rjags 等需额外编译/安装）；且**与 GLM 共享 bathymetry 硬缺**；还需气象预报驱动（NOAA GEFS 等，需外部下载）。**整体投入产出比极低**。
- **结论**：**❌ 不可行**。降级为借鉴其"概率集合 + 滚动同化更新"的预报思想，落到我们的统计/ML 滚动协议里（已具备 CRPS 概率评估）。

### 5.3 降级方案（建议进正式项目）

> 物理模型**不跑完整模拟**，改为"借鉴其方法：用我们的物理量做特征"，这与文献一致（EFDC+ConvLSTM 物理耦合、WRO-Water 物理约束 DL 都是"物理先验转特征/正则"而非全仿真）。

1. **分层物理量特征**：rLakeAnalyzer 温跃层深度（已验证可用，§4）+ 现有 delta_T / thermo_grad；
2. **混合物理指数**：从温度剖面推近似 Schmidt 稳定性、势能异常（PE anomaly）——纯数据可算，作为 M2/特征；
3. **水动力-气象滞后**：文献 Qiu/Pontchartrain 都强调风/降水滞后（M5 已实证风 20-29 天、降水 13-20 天），物理模型做不到，但滞后窗口特征可以在 ML 里直接表达（已有 J 方向慢变量先例）。

---

## 6. PDF 方法基线（方法借鉴类，不需要跑代码）

PDF 中的 Qiu 2025 巢湖框架、Pontchartrain 预警系统属于"方法/方案"，建议**不进"跑代码基线"名单，进"协议内复现其思路"**：

| 文献思路 | 核心方法 | 在我们的协议里的复现路径 | 状态 |
|---|---|---|---|
| Qiu 2025 巢湖 | 多源驱动 + 预警框架（驱动因子→暴发预测→分级） | 把"驱动因子滞后窗特征 + 未来 7 天峰值/均值 + M4 分级"作为正式协议的任务口径（已具备）；出"分级混淆矩阵 + 提前量"而非只 CRPS | ✅ 可在正式项目实现 |
| Pontchartrain | 阈值预警系统（实测 Chl-a + 气象阈值→预警等级） | 复现为**"阈值预警 vs 概率预警"对比**：纯阈值规则（如 conc 超 p90 阈值即预警）作为 M4 的规则基线，与 RamsNet 概率预警（分位数+CRPS）对照，量化概率预警相对规则预警的增益 | ✅ 可在正式项目实现 |
| FLARE（同上） | 概率集合 + 滚动同化 | 降级为"滚动重训练 + 集合/分位数"（已具备） | ✅ |

---

## 7. 结论 —— 对比基线任务最终范围建议

### 进正式项目（✅ 可跑，直接实现）
1. **统计 ML 基线 8 项**（持久化 / Ridge / XGBoost / LightGBM × 均值-峰值-分位数口径）——已按正式协议跑通 17 窗口，正式项目按此实现并出 CRPS 对比表（含 3-seed 正式要求）；
2. **rLakeAnalyzer 温跃层深度**——可装可用且非冗余（相关仅 0.4），作为 M2 分层指数基线/特征；
3. **概率 vs 阈值预警对比**（Qiu/Pontchartrain 思路复现）——作为 M4 基线，不需要物理模型。

### 降级（⚠️/❌ 不跑完整物理模拟，借鉴方法）
4. **GLM-AED / CE-QUAL-W2 / FLARE**：工具链（gfortran/R）可装，但**缺 bathymetry 数据，物理模拟数据不可行**。降级为"借鉴其方法：物理量做特征"（分层指数 + 滞后窗特征 + 概率集合滚动更新），不做完整物理基线。诚实表述：物理模拟基线在数据上不成立，论文/汇报口径用"方法借鉴"而非"物理模型对照"。

### 不进正式项目（不建议）
5. 任何需要 bathymetry / 外部气象预报驱动 / R 生态重依赖（rjags/ncdf4）的物理模拟。

**一句话**：**对比基线任务 = 统计 ML（已跑通）+ rLakeAnalyzer 分层指数（已装好）+ 概率 vs 阈值预警对比（方法复现）；物理模型因缺 bathymetry 数据降级为方法借鉴。** 这 3 类构成"有代码实证的基线 + 有方法学对照的基线"，足以支撑 frozen-design 的对比设计，且不依赖不可行的物理模拟。

---

## 附：复现命令

```bash
# 算力机 sensecore（root，Ubuntu 20.04）
# 1) 统计 ML 基线（17 窗口全量）
cd /data/RAMS/proj
PYTHONPATH=/data/RAMS/proj python3 exp/baseline_feasibility/run_ml_baselines.py --out-json exp/baseline_feasibility/ml_baselines_results.json
# 冒烟：加 --smoke（1 窗口 × 60 树）

# 2) rLakeAnalyzer 温跃层深度实证（只出统计量，中间 CSV 自动删除）
PYTHONPATH=/data/RAMS/proj python3 exp/baseline_feasibility/run_rlake_demo.py

# 环境（已装，记录备查）
apt-get install -y --no-install-recommends gfortran r-base-core      # ~1 min each
Rscript -e 'install.packages("rLakeAnalyzer", repos="https://cloud.r-project.org")'
```
