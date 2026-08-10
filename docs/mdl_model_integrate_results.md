# mdl-model-integrate 模块结果（RAMS 0.2.0 · 五头整合正式模型）

> 模块：`modules/mdl-model-integrate/module_design.yaml`（UID 242d4bb6-2f8d-4414-a8d5-014ff9a883d5）
> 日期：2026-08-10 ｜ 冒烟通过，交主会话评审
> 数据保密：本报告只含形状 / 统计量 / 验收结论，不含任何涉密原始数据行。

## 1. 做了什么

重建磁盘清理时删除的正式模型与训练代码（`rams/models/`、`rams/training/`），
把五头整合探索结论落地为可训练代码，接上 mdl-data-scale 已就位的日级数据管线。

设计全部来自冻结的探索结论：
- **增量目标**（B1）：M1 预测 Δ=conc_{t+h}−conc_t，评估时还原 conc_t+Δ。
- **q9 分位数**（G）：M1 输出 9 个固定分位数 [0.05,0.10,0.20,0.35,0.50,0.65,0.80,0.90,0.95]，
  比 3 分位 CRPS +2%（点数效应）。
- **多任务**（A）：M1+M2+M4，w=1/3/2；M4 标签支持藻华状态（N 定义）与峰值分位两口径。
- **两阶段**（K）：Stage1 单任务 M1（默认 20ep）→ Stage2 冻结 backbone 微调多头（默认 10ep，ts_freeze）。
- **GRU backbone**（框架比较）：最优，不换；hidden 可配（默认 64）。
- **评估协议**（T4）：滚动窗口 730/90/45 + CRPS（q9 分段线性闭合形式 / 3 分位兼容）+ 覆盖率。

## 2. 重建的文件

| 文件 | 内容 |
|---|---|
| `rams/models/rams_net.py` | `SharedGRU`（GRU backbone，hidden 可配）/ `M1Head`（增量分位数，q9 默认，3 分位兼容）/ `M2Head`（分层 2 类）/ `M4Head`（预警 4 级）/ `RamsNet`（三头多任务）+ `count_parameters` |
| `rams/models/__init__.py` | 包导出（RamsNet / SharedGRU / 头 / QUANTILE_LEVELS / QUANTILES / WARN_LEVELS） |
| `rams/training/trainer.py` | `QuantileLoss`（pinball，任意分位数）/ `MultiTaskLoss`（w=1/3/2 + M4 逆频率类别权重）/ `Trainer`（`fit` 单阶段、`fit_two_stage` / `fit_m1_only` / `fit_multi` 两阶段、`evaluate`、`predict_m1`）/ `crps_cdf_pline`（任意结 CRPS）/ `crps_quantiles`（3 分位兼容）/ `make_m4_labels`（peak_quantile / bloom）/ 独立 `__main__` 冒烟 |
| `rams/training/__init__.py` | 包导出（Trainer / 损失 / CRPS / M4 标签 / 权重常量） |
| `rams/training/smoke_roll.py` | 全量数据滚动窗口冒烟脚本（1 窗口 × 两阶段 fast_dev_run，只输出形状/统计量） |
| `rams/__init__.py` | 包入口（补 `models`/`training` 模块图） |
| `tests/test_model_integrate.py` | 15 项单元测试（合成小数据）：q9 形状 / quantile_matrix / predict_mean / predict_interval / 3 分位兼容 / 参数量预算 / 损失 / 两阶段 / CRPS / M4 标签 |
| `pyproject.toml` | ruff per-file-ignore 追加 `models/rams_net.py`（N806）、`training/trainer.py`（N803/N806）、`training/smoke_roll.py`（N806）（张量命名约定，与 `data/tensor_builder.py` 一致） |

## 3. 冒烟结果（合成小数据 + standard.parquet 全量首窗口）

### 3.1 模型前向形状（q9）
- `RamsNet(feat_dim=29, horizon=7, n_quantiles=9)`：M1 `(B, 63)`（9 结 × 7 视界）、M2 `(B, 2)`、M4 `(B, 4)`
- `quantile_matrix` → `(B, 9, 7)`；`predict_mean` 取结 0.50 → `(B, 7)`；`predict_interval` 取结 0.10/0.90 → `(B, 7)`×2
- 参数量 35,205（hidden=64）；3 分位兼容契约（0.1.0）形状 `(B, 3H)` 通过
- **兼容**：既有 `tests/test_smoke.py`（旧 3 分位接口）7/7 通过，未破坏 0.1.0 契约

### 3.2 两阶段训练循环（fast_dev_run）
- 合成小数据：Stage1 loss 0.4123→0.4628 有限；Stage2 loss 5.3701→5.3232 有限；CRPS 均值 0.7794；覆盖率 [p10,p90]∈[0,1]
- 全量首窗口（762 样本，n_train=675，delta_scale=9.03）：Stage1 loss 0.1823→0.1675、Stage2 4.88→4.87（随机初始化，loss 有限且下降）
- 冻结正确：Stage2 `backbone.requires_grad=False`，训练结束后恢复 `True`

### 3.3 覆盖率 / CRPS 概念验证（首窗口测试段 87 样本，随机初始化）
- q9 输出 `(87, 9, 7)`；还原 conc 单位 `(87, 9, 7)`
- CRPS 3.31 vs 持久化 3.59，相对技能 **+7.8%**（随机初始化即超持久化，机制接线正确；训练收敛后应达探索的 +20% 量级）
- p50 RMSE 4.93；覆盖率 [p10,p90] 0.113（随机初始化区间未校准，属正常；训练后由多任务 Stage2 抬升）

### 3.4 M4 标签（日级协议）
- bloom 模式（N 定义藻华状态）：[689, 73]（正例 73/762，9.6%，与 data-scale 全量 7.0% 同口径量级）
- peak_quantile 模式（探索 A/K 协议 4 级）：[544, 150, 49, 19]

### 3.5 单元测试
- `tests/test_model_integrate.py`：**15/15 通过**（合成小数据）
- 全量套件：`tests/` 共 **30/30 通过**（含 data-scale 8 项、旧 smoke 7 项）
- ruff lint + format：`rams/` 全通过（正式代码风格：类型注解 + I/O 形状 docstring）

### 3.6 保密红线
- `data/` 只读；冒烟只输出形状/统计量，未打印任何原始数据行

## 4. 验收标准达成情况

| 冻结验收标准 | 达成 |
|---|---|
| M1 增量Δ + q9 分位数 | ✅ M1 头输出 9 固定分位数（0.05–0.95），增量 Δ 口径（评估还原 conc_t+Δ） |
| M2/M4 多任务头（M4 用藻华状态） | ✅ M2 分层 2 类 + M4 预警 4 级；M4 标签支持 bloom（N 定义）/ peak_quantile 双口径 |
| 两阶段训练（ts_freeze） | ✅ Stage1 单任务 M1（默认 20ep）→ Stage2 冻结 backbone 微调多头（默认 10ep），`fit_two_stage` 一键；冻结/解冻可配 |
| CRPS 相对持久化 >+20% | 🔶 代码链路就绪（CRPS 闭合形式 + 相对技能 + 持久化基线）；冒烟随机初始化 +7.8% 验证机制接线正确。**正式收敛值需算力机训练**（探索实证 ts_freeze 为 +28.4%，K 报告） |
| 覆盖率达标 | 🔶 代码链路就绪（[p10,p90] 覆盖率）；正式值需训练后评估（探索实证多任务 0.806、ts_freeze 0.704） |

> 说明：CRPS>+20% 与覆盖率属**训练后评估指标**，冒烟阶段只验证评估链路正确（随机初始化已 +7.8% 验证机制）。两项正式数字待 sensecore H100 跑 3-seed 滚动窗口后由 st-train-v020 补齐。

## 5. 下一步建议

1. **算力机正式训练**（st-train-v020）：用本模块 `DailyTensorBuilder` + `RamsNet(q9)` + `fit_two_stage` 跑 3-seed × 17 窗口，验证 CRPS 相对持久化 >+20% 与覆盖率（冒烟已验证链路，剩余为算力/时间）。
2. **M4 事件评估**（mdl-m4-warning）：M4 头用 `make_m4_labels(..., mode="bloom")` 产出的藻华标签训练后，按 N 定义 5 事件评估召回率 + 提前量（模块依赖本模块 trained-model）。
3. **端到端接入**：`fit` 单阶段多任务 + `fit_two_stage` 两阶段两口径均可用；训练脚本可在 `scripts/train.py` 重建时直接复用（增量归一化 + q9 + M4 bloom 标签 + 滚动窗口）。
4. **校准补强（可选）**：ts_freeze 覆盖率距 80% 目标仍差 ~10pp（探索结论），如需达标可试更长 Stage2 或 Stage2 只训 M1 分位头。
