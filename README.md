# RAMS-Net: Reservoir Algal Monitoring System
# RAMS-Net：水库藻类智能监测系统

Multi-task deep learning for vertically stratified reservoir algal forecasting and bloom early warning.
面向垂向分层水库藻类预测与藻华预警的多任务深度学习框架。

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](environment.yml)

---

## Overview / 概述

RAMS-Net is a multi-task deep learning framework for monitoring reservoir algae, developed on a single stratified reservoir (2021–2025, 20 depth layers at 0.5 m intervals, 3-hour sampling). It integrates five monitoring tasks into a shared GRU backbone:

RAMS-Net 是一个面向水库藻类监测的多任务深度学习框架，基于单座分层水库实测数据（2021–2025 年，0.5–10 m 共 20 个水深分层，3 小时采样间隔）开发，将五项监测任务整合进共享 GRU 主干：

| Task | Description | 描述 | Type |
|---|---|---|---|
| **M1** | Algal concentration forecasting | 藻类浓度预测 | Regression (9-quantile) |
| **M2** | Thermal stratification detection | 热分层识别 | Classification |
| **M3** | Monitoring depth selection | 监测点位优化 | GAT + greedy |
| **M4** | Bloom early warning | 藻华预警 | Probabilistic |
| **M5** | Causal driver / time-lag analysis | 因果驱动及时滞分析 | PCMCI+ |

---

## Key Results / 关键成果 (v0.2.0)

| Metric | Result | 指标 | 结果 |
|---|---|---|---|
| **CRPS skill vs persistence** | **+22.1%** | CRPS 相对持久性基线技巧 | **+22.1%**（3 种子 × 17 滚动窗口，全视界 >+20%） |
| **Interval coverage [p10,p90]** | **0.766** | 预测区间覆盖率 | **0.766**（接近 80% 目标） |
| **Bloom warning lead time** | **14.5 days** | 藻华预警提前量 | **14.5 天**（阈值法的 3.6 倍） |
| **Bloom false positives** | Half of baseline | 藻华误报 | 阈值法的一半 |
| **vs best statistical baseline** | **+13.4%** | 对比最强统计基线 | **+13.4%**（优于 LightGBM 分位数） |
| **Bloom events identified** | 12 | 识别的藻华事件 | 12 次（2021 年为主，预警窗口 19.5 天） |

**Methodological contributions / 方法贡献：**
1. **Incremental prediction / 增量预测** — predicting the change Δ = conc_{t+h} − conc_t rather than absolute concentration addresses the strong autocorrelation regime and outperforms persistence across all horizons. 预测变化量而非绝对值，有效应对强自相关情境，全视界优于持久性基线。
2. **Multi-task shared backbone / 多任务共享主干** — forecasting / stratification / warning heads jointly improve interval calibration. 预测/分层/预警任务头联合提升区间校准。
3. **Daily-scale process matching / 日尺度过程匹配** — daily sampling with 7-day horizon aligns with algal growth timescales. 日尺度采样配合 7 天视界，匹配藻类生长时间尺度。
4. **Probabilistic bloom warning / 概率化藻华预警** — calibrated probability output provides 3.6× lead time with half the false positives. 校准后的概率输出提供 3.6 倍提前量，误报减半。

---

## Architecture / 架构

```
Input (B, T=30, D=20, C)
    └── Shared GRU Backbone (hidden=64) / 共享 GRU 主干
          ├── M1: Incremental forecast (9 quantiles, H=7 days) / 增量预测（9 分位，7 天）
          ├── M2: Stratification classification / 分层分类
          └── M4: Bloom warning (probability) / 藻华预警（概率）
```

Two-stage training / 两阶段训练: Stage 1 single-task M1 (20 epochs) → Stage 2 freeze backbone, fine-tune multi-head (10 epochs). Multi-task loss L = w1·L_q(M1) + w2·CE(M2) + w4·CE(M4), w=(1,3,2).
第一阶段单任务训练 M1（20 轮）→ 第二阶段冻结主干、微调多任务头（10 轮）。

---

## Installation / 安装

```bash
# conda environment (GPU PyTorch) / conda 环境（GPU 版 PyTorch）
conda create -n rams python=3.10 -y
conda activate rams
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -e .
```

## Quick Start / 快速开始

```bash
# Build standard dataset (from raw xlsx, one-time) / 构建标准数据集（一次性）
python scripts/build_dataset.py --raw data/raw --out data/processed

# Train formal model (daily scale, incremental target, multi-task) / 训练正式模型
python scripts/train_v020.py --fast-dev       # smoke test / 冒烟测试
python scripts/train_v020.py                  # full 3-seed × 17-window training / 全量训练

# Evaluate bloom warning (recall / lead time / false positives) / 藻华预警评估
python scripts/eval_m4_warning.py --smoke
```

---

## Data / 数据

Confidential reservoir monitoring data: 258,542 algae observations across 20 depth layers and 232,067 meteorological observations (10-min sampling). **Data is not distributed** due to confidentiality; the pipeline expects a processed `standard.parquet`.
涉密水库监测数据：258,542 条藻类观测（20 层）+ 232,067 条气象观测（10 分钟采样）。**因保密协议不公开数据**；管线需要已处理的 `standard.parquet`。

**Bloom event definition / 藻华事件定义**: surface-band (0.5–3.0 m) median > band-p90 AND ≥3 layers in 0.5–5.0 m > their p90, sustained ≥2 days.
表层带（0.5–3.0 m）浓度中位数 > 带 p90，且 0.5–5.0 m 内 ≥3 层 > 各自 p90，持续 ≥2 天。

---

## Repository Structure / 仓库结构

```
├── rams/               # Core package / 核心包
│   ├── data/           # tensor builder, preprocessing / 张量构建与预处理
│   ├── models/         # RAMS-Net (GRU + M1/M2/M4 heads) / 模型
│   └── training/       # two-stage trainer, losses / 训练器与损失
├── scripts/            # CLI: build_dataset / train_v020 / eval_m4_warning / 命令行入口
├── tests/              # pytest suite (53 tests, 83% coverage) / 测试套件
├── docs/               # papers (EN/CN), results, figures / 论文与结果
├── exp/                # exploration records / 探索记录
└── environment.yml     # conda dependencies / 依赖
```

## Results & Figures / 结果与图表

See `docs/` for the full exploration report and paper drafts (English and Chinese) with 12 figures covering forecasting performance, calibration, bloom warning, causality, ablations, and framework comparison.
完整探索报告与论文初稿（中英文）见 `docs/`，含 12 张图，覆盖预测性能、校准、藻华预警、因果、消融与框架对比。

---

## Citation / 引用

If you use this work, please cite / 如使用本工作，请引用：

```bibtex
@article{rams2026,
  title={Multi-task Deep Learning with Incremental Prediction for Reservoir Algal Forecasting and Bloom Early Warning},
  journal={to be submitted},
  year={2026}
}
```

## License / 许可

MIT (see LICENSE / 见 LICENSE).

## Acknowledgments / 致谢

Data provided under confidentiality agreement; site anonymized. Training performed on NVIDIA H100.
数据依保密协议提供，站点已匿名；训练在 NVIDIA H100 上完成。
