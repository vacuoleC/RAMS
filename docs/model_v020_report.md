# RAMS 0.2.0 落地总结报告

> 2026-08-11（东八区）· RAMS-Algae-Monitoring 项目 0.2.0 版本落地完成
> 项目 UID：`47077f0c-1772-47c8-ae2a-b99a38928335`
> 治理框架：build-supervised-project（Controller Skill `rams-algae-controller`）

---

## 1. 落地范围

按冻结设计（frozen-design 5 模块）完成全部模块落地：

| 模块 | 交付物 | 验收 |
|---|---|---|
| mdl-data-scale | 日级数据管线 + 藻华标签 | ✅ 张量(B,30,D,C) + BloomLabeler |
| mdl-model-integrate | 五头整合正式模型 | ✅ CRPS +22.1% |
| mdl-m4-warning | 藻华预警评估 | ✅ 提前 14.5 天 |
| mdl-baseline-compare | 对比基线报告 | ✅ 比最强基线好 13.4% |
| mdl-test-package | 全链路测试 + 打包 | ✅ 53/53 测试，wheel 构建成功 |

## 2. 核心科学成果

### M1 藻类浓度预测（增量Δ + q9 分位 + 多任务 + 日级）
- **CRPS 相对持久化 +22.1%**（3-seed × 17 窗口，日级 T=30/H=7）
- 覆盖率 [p10,p90] = **0.766**（接近 80% 目标）
- 全部 7 视界技能 >+20%，15/17 窗口 >+15%
- 比最强统计基线（lgb_q +10.0%）好 **13.4%**，唯一有概率校准

### M4 藻华预警（概率预警）
- **命中事件平均提前 14.5 天**（13-16 天），与 N 探索"爬升提前量 12-27 天"一致
- **概率预警 vs 阈值预警**：模型用一半误报（6 段 vs 14 段）换 **3.6 倍提前量**（14.5 天 vs 4 天）——核心卖点
- 阈值敏感性：θ=0.5-0.6 为推荐工作点

### 对比基线
- RamsNet +22.1% vs 持久化；比 lgb_q（+10%）好 13.4pp
- rLakeAnalyzer thermo.depth 非冗余（r=0.40）但边际（+0.57%）
- 物理模型（GLM/CE-QUAL-W2/FLARE）缺 bathymetry 降级为方法借鉴

## 3. 落地过程中修复的真实 bug

| Bug | 影响 | 修复 |
|---|---|---|
| DailyTensorBuilder 数据切分：`tr_ts` 缺失时泄漏训练行到测试段（窗口12） | 测试集污染（n_test 232 vs 正确 87） | 改行计数 `searchsorted`，回归测试 |
| stdout wrapper 被 pytest import 包住 → teardown 崩溃 | 全站测试崩溃 | `__main__` 守卫 |
| .gitignore `data/` 裸规则误忽略 `rams/data/` 源码 | 源码无法追踪 | `/data/` 锚定 |

## 4. 工程资产

```
rams/
├── data/tensor_builder.py    # 日级管线 + BloomLabeler + 滚动窗口
├── data/preprocessing.py     # 唯一接触 xlsx 的清洗（85% 测试覆盖）
├── models/rams_net.py        # GRU + M1增量q9 + M2分层 + M4预警（95% 覆盖）
├── training/trainer.py       # 分位数损失 + 多任务 + 两阶段ts_freeze（91% 覆盖）
scripts/
├── train_v020.py             # 正式训练验收脚本
├── eval_m4_warning.py        # 藻华预警评估
tests/                        # 53 项测试，83% 分支覆盖
docs/
├── mdl_data_scale_results.md
├── mdl_model_integrate_results.md
├── mdl_m4_warning_results.md
├── baseline_comparison.md
├── test_report.md
exp/                          # 探索记录 + 正式训练结果
```

## 5. 验收标准达成

| 冻结验收标准 | 状态 |
|---|---|
| CRPS 相对持久化 >+20% | ✅ +22.1%（3-seed 全超线） |
| M4 藻华事件召回 + 提前量 | ✅ 提前 14.5 天，召回 2/5（θ=0.5） |
| pytest 全链路 | ✅ 53/53，83% 覆盖 |
| 报告含探索证据链 + 对比基线 | ✅ 全部文档产出 |

## 6. 后续方向（0.2.1+ / 报告/部署）

- **M4 改进**：标签改 3h 网格恢复 12 事件口径；加垂直联动确认门控压秋冬误报
- **覆盖达标 80%**：ts_freeze + 事后共形（F 探索 α=0.2 cqr 已验证）
- **报告/论文**：核心卖点 = 概率预警 3.6 倍提前量 + 增量预测超持久化 + 五头整合
- **部署（Stage 7）**：ONNX 导出、FastAPI（4 个 CLI 入口待建）
- **找数据**：M 探针证明接近数据上限，需多站/营养盐数据突破

## 7. 治理记录

- 5 个模块全部按 Controller Skill 执行，每模块子代理 + 冒烟 + 提交
- 钩子审计：每次工具调用记录（events.jsonl），状态机正常
- 数据保密：全程未打印原始数据行，data/ 只读
