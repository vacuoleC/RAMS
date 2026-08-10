# RAMS Paper Figures — Manifest

> 生成日期：2026-08-11 ｜ 目标期刊：Ecological Informatics 风格（英文标注、300 dpi、可发表）
> 产出目录：`docs/paper_figs/` ｜ 生成脚本：`scripts/paper_figs/make_figures.py`
> 保密红线：所有图仅使用标准化（z-score）/去标识化统计量，未显示任何原始浓度行或具体站点值。

---

## 图清单

| # | 文件名 | 一句话说明 | 数据来源 |
|---|--------|-----------|----------|
| 1 | `fig1_data_overview.png` | 研究区域概览：日级表层浓度与水温标准化时间序列，红色底纹标注 12 个 N 定义藻华事件。 | `data/processed/standard.parquet`（仅统计：日级均值→z-score）；`exp/model_enhancement/n_bloom_identify/results.json`（事件区间） |
| 2 | `fig2_bloom_events.png` | 12 个藻华事件的日历分布（甘特条带），红色=2021 主导（N1–N4），蓝色=2022–2025。 | `exp/model_enhancement/n_bloom_identify/results.json`（`final_event_list`） |
| 3 | `fig3_stratification.png` | 垂直分层结构：20 层浓度与水温周均 z-score 热力图（时间×深度），浓度色标裁剪至 +3σ 突出藻华层。 | `data/processed/standard.parquet`（仅统计：逐层周均→z-score） |
| 4 | `fig4_architecture.png` | RAMS-Net 五头整合架构图（复用现有英文架构图：共享 GRU backbone + M1/M2/M4 头 + 多任务损失）。 | `docs/rams_architecture.png`（原图），架构说明见 `docs/architecture_blueprint.md` |
| 5 | `fig5_increment_vs_baseline.png` | 增量预测 vs 基线：CRPS 逐视界（1–7 天）对比 RAMS-Net（增量+q9）与持久化/lgb_q/xgb_q，全视界超持久化 +21%。 | `exp/model_enhancement/st-train-v020/results.json`（`per_horizon`）；`exp/baseline_feasibility/ml_baselines_results.json`（`crps_h`） |
| 6 | `fig6_coverage.png` | 预测区间覆盖率对比：RAMS-Net [p10,p90] 覆盖率接近名义 80%，远高于统计基线；增量分位数(0.824)优于绝对分位数(0.669)。 | `exp/baseline_feasibility/baseline_comparison_results.json`；`exp/model_enhancement/b2_increment_quantile/results.json` |
| 7 | `fig7_m4_threshold.png` | M4 藻华预警阈值敏感性：召回 vs 误报权衡（θ=0.3–0.8 扫描），与带 p75 阈值规则基线对照。 | `exp/mdl_m4_warning/results.json`（`sweep_n_events`、`baseline_p75`） |
| 8 | `fig8_lead_time.png` | 预警提前量对比：概率预警（θ=0.5/0.6/0.7）中位提前 14–21 天 vs 规则阈值基线仅 4 天。 | `exp/mdl_m4_warning/results.json`（`sweep_n_events`、`baseline_p75`） |
| 9 | `fig9_causal_graph.png` | M5 PCMCI+ 因果结构（重绘）：6 变量定向边（实线/虚线/点线=边强度 |MCI| 分级），节点标注自回归 τ=1 强度。 | `docs/m5_pcmci_edges.json`；参数见 `docs/m5_pcmci_results.md` |
| 10 | `fig10_ablations.png` | 关键消融：(a) 单任务/多任务/两阶段 CRPS+覆盖率；(b) 增量 vs 绝对 vs 持久化 RMSE（B1）。 | `exp/model_enhancement/k_two_stage/results.json`；`exp/model_enhancement/b1_increment/results.md` |
| 11 | `fig11_framework_compare.png` | 框架比较：12 模型测试 RMSE 条形图（RamsNet 多任务最优，3.64）。 | `docs/framework_compare_results.json`；`docs/framework_compare.md` |
| 12 | `fig12_distribution_acf.png` | 数据分布：(a) 表层浓度 z-score 直方图 vs N(0,1)；(b) 日级自相关函数（长拖尾、强自回归）。 | `data/processed/standard.parquet`（仅统计：z-score、ACF） |

---

## 关键结论图标注（供论文正文引用）

- **Fig. 5**：RAMS-Net 增量预测相对持久化 CRPS 全 7 视界 +21.1% ~ +23.0%。
- **Fig. 6**：正式协议覆盖率 0.766（名义 80%）；探索 3h 协议增量分位数 0.824 vs 绝对 0.669。
- **Fig. 7/8**：θ=0.5 召回 2/5、中位提前 14.5 天、误报 6 段；规则基线召回 5/5 但仅提前 4 天、误报 14 段 —— **提前量 3.6 倍、误报减半**。
- **Fig. 9**：自回归主导（conc τ=1 MCI≈0.62），气象短滞直驱弱（|MCI|<0.13），20–30 天滞后为季节共变。
- **Fig. 11**：固定切分下所有学习框架显著优于其余框架（RamsNet 3.643 vs 次优 DLinear 5.519，-51.5%）；持久化 1.23 为平凡基线。

## 复现

```bash
D:/enviranment/Python313/python.exe scripts/paper_figs/make_figures.py
```

依赖：matplotlib ≥3.7、pandas、numpy、networkx、pyarrow（读 parquet 仅用于统计）。
