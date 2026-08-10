# RAMS-Net: Reservoir Algal Monitoring System

Multi-task deep learning for vertically stratified reservoir algal forecasting and bloom early warning.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](environment.yml)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)](environment.yml)

## Overview

RAMS-Net is a multi-task deep learning framework for monitoring reservoir algae, developed on a single stratified reservoir (2021–2025, 20 depth layers at 0.5 m intervals, 3-hour sampling). It integrates five monitoring tasks into a shared GRU backbone:

| Task | Description | Type |
|---|---|---|
| **M1** | Algal concentration forecasting | Regression (9-quantile) |
| **M2** | Thermal stratification detection | Classification |
| **M3** | Monitoring depth selection | GAT + greedy |
| **M4** | Bloom early warning | Probabilistic |
| **M5** | Causal driver / time-lag analysis | PCMCI+ |

## Key Results (v0.2.0)

| Metric | Result |
|---|---|
| **CRPS skill vs persistence** | **+22.1%** (3 seeds × 17 rolling windows, all horizons >+20%) |
| **Interval coverage [p10,p90]** | **0.766** (approaching 80% target) |
| **Bloom warning lead time** | **14.5 days** (3.6× over threshold baseline) |
| **Bloom false positives** | Half of threshold baseline |
| **vs best statistical baseline** | **+13.4%** better than LightGBM quantile |
| **Bloom events identified** | 12 (2021-dominant, 19.5-day median warning window) |

**Methodological contributions:**
1. **Incremental prediction** — formulating the target as Δ = conc_{t+h} − conc_t (rather than absolute concentration) addresses the strong autocorrelation regime and outperforms persistence across all horizons.
2. **Multi-task shared backbone** — forecasting / stratification / warning heads jointly improve interval calibration.
3. **Daily-scale process matching** — daily sampling with 7-day horizon aligns with algal growth timescales (1–3 day doubling).
4. **Probabilistic bloom warning** — calibrated probability output with threshold tuning provides 3.6× lead time with half the false positives.

## Architecture

```
Input (B, T=30, D=20, C)
    └── Shared GRU Backbone (hidden=64)
          ├── M1: Incremental forecast (9 quantiles, H=7 days)
          ├── M2: Stratification classification
          └── M4: Bloom warning (probability)
```

Two-stage training: Stage 1 single-task M1 (20 epochs) → Stage 2 freeze backbone, fine-tune multi-head (10 epochs). Multi-task loss L = w1·L_q(M1) + w2·CE(M2) + w4·CE(M4), w=(1,3,2).

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
# Build standard dataset (from raw xlsx, one-time)
python scripts/build_dataset.py --raw data/raw --out data/processed

# Train formal model (daily scale, incremental target, multi-task)
python scripts/train_v020.py --fast-dev       # smoke test
python scripts/train_v020.py                  # full 3-seed × 17-window training

# Evaluate bloom warning (recall / lead time / false positives)
python scripts/eval_m4_warning.py --smoke
```

## Data

Confidential reservoir monitoring data: 258,542 algae observations across 20 depth layers and 232,067 meteorological observations (10-min sampling). Data is **not distributed** due to confidentiality; the pipeline expects a processed `standard.parquet`.

**Bloom event definition**: surface-band (0.5–3.0 m) median > band-p90 AND ≥3 layers in 0.5–5.0 m > their p90, sustained ≥2 days.

## Repository Structure

```
├── rams/               # Core package
│   ├── data/           # tensor builder, preprocessing
│   ├── models/         # RAMS-Net (GRU + M1/M2/M4 heads)
│   └── training/       # two-stage trainer, losses
├── scripts/            # CLI: build_dataset / train_v020 / eval_m4_warning
├── tests/              # pytest suite (53 tests, 83% coverage)
├── docs/               # papers (EN/CN), results, figures
├── exp/                # exploration records & results
└── environment.yml     # conda dependencies
```

## Results & Figures

See `docs/` for the full exploration report and the paper drafts (English and Chinese) with 12 figures covering forecasting performance, calibration, bloom warning, causality, ablations, and framework comparison.

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
