# RAMS-Net: Reservoir Algal Monitoring System

Multi-task deep learning for vertically stratified reservoir algal forecasting and bloom early warning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](environment.yml)

---

## Overview

RAMS-Net is a multi-task deep learning framework for monitoring reservoir algae, developed on a single stratified reservoir (2021–2025, 20 depth layers at 0.5 m intervals, 3-hour sampling). It integrates five monitoring tasks into a shared GRU backbone:

| Task | Description | Type |
|---|---|---|
| **M1** | Algal concentration forecasting | Regression (9-quantile) |
| **M2** | Thermal stratification detection | Classification |
| **M3** | Monitoring depth selection | GAT + greedy |
| **M4** | Bloom early warning | Probabilistic |
| **M5** | Causal driver / time-lag analysis | PCMCI+ |

---

## Key Results (v0.2.0)

| Metric | Result |
|---|---|
| **CRPS skill vs persistence** | **+22.1%** (3 seeds × 17 rolling windows, all horizons >+20%) |
| **Interval coverage [p10,p90]** | **0.766** (approaching 80% target) |
| **Bloom warning lead time** | **14.5 days** (3.6× over threshold baseline) |
| **Bloom false positives** | Half of threshold baseline |
| **vs best statistical baseline** | **+13.4%** better than LightGBM quantile |
| **Bloom events identified** | 12 (2021-dominant, 19.5-day median warning window) |

<p align="center">
  <img src="docs/paper_figs/fig5_increment_vs_baseline.png" width="600" alt="Forecasting skill vs baselines">
  <img src="docs/paper_figs/fig8_lead_time.png" width="450" alt="Warning lead time">
</p>

**Methodological contributions:**
1. **Incremental prediction** — predicting the change Δ = conc_{t+h} − conc_t rather than absolute concentration addresses the strong autocorrelation regime and outperforms persistence across all horizons.
2. **Multi-task shared backbone** — forecasting / stratification / warning heads jointly improve interval calibration.
3. **Daily-scale process matching** — daily sampling with 7-day horizon aligns with algal growth timescales (1–3 day doubling).
4. **Probabilistic bloom warning** — calibrated probability output with threshold tuning provides 3.6× lead time with half the false positives.

---

## Architecture

```
Input (B, T=30, D=20, C)
    └── Shared GRU Backbone (hidden=64)
          ├── M1: Incremental forecast (9 quantiles, H=7 days)
          ├── M2: Stratification classification
          └── M4: Bloom warning (probability)
```

<p align="center">
  <img src="docs/paper_figs/fig4_architecture.png" width="600" alt="RAMS-Net Architecture">
</p>

Two-stage training: Stage 1 single-task M1 (20 epochs) → Stage 2 freeze backbone, fine-tune multi-head (10 epochs). Multi-task loss L = w1·L_q(M1) + w2·CE(M2) + w4·CE(M4), w=(1,3,2).

---

## Installation

```bash
# conda environment (GPU PyTorch)
conda create -n rams python=3.10 -y
conda activate rams
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e .
```

## Quick Start

```bash
# Build standard dataset from raw xlsx (one-time)
python scripts/build_dataset.py --raw data/raw --out data/processed

# Train formal model (daily scale, incremental target, multi-task)
python scripts/train_v020.py --fast-dev       # smoke test
python scripts/train_v020.py                  # full 3-seed × 17-window training

# Evaluate bloom warning (recall / lead time / false positives)
python scripts/eval_m4_warning.py --smoke
```

---

## Data

Confidential reservoir monitoring data: 258,542 algae observations across 20 depth layers and 232,067 meteorological observations (10-min sampling). **Data is not distributed** due to confidentiality; the pipeline expects a processed `standard.parquet`.

**Bloom event definition**: surface-band (0.5–3.0 m) median > band-p90 AND ≥3 layers in 0.5–5.0 m > their p90, sustained ≥2 days.

---

## Repository Structure

```
├── rams/               # core package
│   ├── data/           # tensor builder, preprocessing
│   ├── models/         # RAMS-Net (GRU + M1/M2/M4 heads)
│   └── training/       # two-stage trainer, losses
├── scripts/            # CLI entry points
├── tests/              # pytest suite (53 tests, 83% coverage)
├── docs/               # papers, results, figures
├── exp/                # exploration records
└── environment.yml     # conda dependencies
```

## Results & Figures

See `docs/` for the full exploration report and paper drafts (English and Chinese) with 12 figures covering forecasting performance, calibration, bloom warning, causality, ablations, and framework comparison.

---

## Citation

If you use this work, please cite:

```bibtex
@article{rams2026,
  title={Multi-task Deep Learning with Incremental Prediction for Reservoir Algal Forecasting and Bloom Early Warning},
  journal={to be submitted},
  year={2026}
}
```

## License

MIT (see LICENSE).

## Acknowledgments

Data provided under confidentiality agreement; site anonymized. Training performed on NVIDIA H100.

---
---

# RAMS-Net：水库藻类智能监测系统

面向垂向分层水库藻类预测与藻华预警的多任务深度学习框架。

---

## 概述

RAMS-Net 是一个面向水库藻类监测的多任务深度学习框架，基于单座分层水库实测数据（2021–2025 年，0.5–10 m 共 20 个水深分层，3 小时采样间隔）开发。它将五项监测任务整合进共享 GRU 主干：

| 任务 | 描述 | 类型 |
|---|---|---|
| **M1** | 藻类浓度预测 | 回归（9 分位） |
| **M2** | 热分层识别 | 分类 |
| **M3** | 监测点位优化 | GAT + 贪心 |
| **M4** | 藻华预警 | 概率化 |
| **M5** | 因果驱动及时滞分析 | PCMCI+ |

---

## 关键成果（v0.2.0）

| 指标 | 结果 |
|---|---|
| **CRPS 相对持久性基线技巧** | **+22.1%**（3 种子 × 17 滚动窗口，全视界 >+20%） |
| **预测区间覆盖率 [p10,p90]** | **0.766**（接近 80% 目标） |
| **藻华预警提前量** | **14.5 天**（阈值法的 3.6 倍） |
| **藻华误报** | 阈值法的一半 |
| **对比最强统计基线** | **+13.4%**（优于 LightGBM 分位数） |
| **识别的藻华事件** | 12 次（2021 年为主，预警窗口 19.5 天） |

<p align="center">
  <img src="docs/paper_figs/fig5_increment_vs_baseline.png" width="600" alt="预测技巧对比基线">
  <img src="docs/paper_figs/fig8_lead_time.png" width="450" alt="预警提前量">
</p>

**方法贡献：**
1. **增量预测** — 预测变化量 Δ = conc_{t+h} − conc_t 而非绝对值，有效应对强自相关情境，全视界优于持久性基线。
2. **多任务共享主干** — 预测/分层/预警任务头联合提升区间校准。
3. **日尺度过程匹配** — 日尺度采样配合 7 天视界，匹配藻类生长时间尺度（翻倍 1–3 天）。
4. **概率化藻华预警** — 校准后的概率输出配合阈值调优，提供 3.6 倍提前量，误报减半。

---

## 架构

```
输入 (B, T=30, D=20, C)
    └── 共享 GRU 主干 (hidden=64)
          ├── M1: 增量预测（9 分位，H=7 天）
          ├── M2: 分层分类
          └── M4: 藻华预警（概率）
```

<p align="center">
  <img src="docs/paper_figs/fig4_architecture.png" width="600" alt="RAMS-Net 架构">
</p>

两阶段训练：第一阶段单任务训练 M1（20 轮）→ 第二阶段冻结主干、微调多任务头（10 轮）。多任务损失 L = w1·L_q(M1) + w2·CE(M2) + w4·CE(M4)，w=(1,3,2)。

---

## 安装

```bash
# conda 环境（GPU 版 PyTorch）
conda create -n rams python=3.10 -y
conda activate rams
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e .
```

## 快速开始

```bash
# 构建标准数据集（一次性，从原始 xlsx）
python scripts/build_dataset.py --raw data/raw --out data/processed

# 训练正式模型（日尺度、增量目标、多任务）
python scripts/train_v020.py --fast-dev       # 冒烟测试
python scripts/train_v020.py                  # 全量 3 种子 × 17 窗口训练

# 藻华预警评估（召回 / 提前量 / 误报）
python scripts/eval_m4_warning.py --smoke
```

---

## 数据

涉密水库监测数据：258,542 条藻类观测（20 层）+ 232,067 条气象观测（10 分钟采样）。**因保密协议不公开数据**；管线需要已处理的 `standard.parquet`。

**藻华事件定义**：表层带（0.5–3.0 m）浓度中位数 > 带 p90，且 0.5–5.0 m 内 ≥3 层 > 各自 p90，持续 ≥2 天。

---

## 仓库结构

```
├── rams/               # 核心包
│   ├── data/           # 张量构建与预处理
│   ├── models/         # RAMS-Net（GRU + M1/M2/M4 头）
│   └── training/       # 两阶段训练器、损失
├── scripts/            # 命令行入口
├── tests/              # 测试套件（53 项测试，83% 覆盖率）
├── docs/               # 论文、结果、图表
├── exp/                # 探索记录
└── environment.yml     # conda 依赖
```

## 结果与图表

完整探索报告与论文初稿（中英文）见 `docs/`，含 12 张图，覆盖预测性能、校准、藻华预警、因果、消融与框架对比。

---

## 引用

如使用本工作，请引用：

```bibtex
@article{rams2026,
  title={Multi-task Deep Learning with Incremental Prediction for Reservoir Algal Forecasting and Bloom Early Warning},
  journal={to be submitted},
  year={2026}
}
```

## 许可

MIT（见 LICENSE）。

## 致谢

数据依保密协议提供，站点已匿名；训练在 NVIDIA H100 上完成。
