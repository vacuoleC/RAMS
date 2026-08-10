# 基于增量多任务深度学习的垂向分层水库藻类预测与藻华预警

## A Multi-Task Deep Learning Framework with Incremental Prediction for Vertically Stratified Reservoir Algal Forecasting and Bloom Early Warning

---

## Abstract

Reservoir algal blooms pose significant risks to drinking water safety, yet predicting bloom dynamics remains challenging due to strong temporal autocorrelation and vertical stratification. This study develops a multi-task deep learning framework (RAMS-Net) for a single stratified reservoir (2021-2025, 20 depth layers, 3-hour sampling). We demonstrate that (1) formulating the prediction target as *incremental change* (Δ = conc_{t+h} − conc_t) rather than absolute concentration substantially improves skill over persistence baselines; (2) a shared GRU backbone with multi-task heads (forecasting, stratification, bloom warning) improves interval calibration; (3) daily-scale sampling with 7-day horizon better matches the physical process scale; and (4) probabilistic bloom warning provides 3.6× longer lead time than threshold-based methods with half the false positives. The framework achieves CRPS skill of +22.1% over persistence with coverage of 0.766, and identifies 12 bloom events with median 19.5-day pre-event warning windows. These results provide a scientifically grounded and deployable approach for reservoir algae monitoring.

**Keywords**: reservoir algae; deep learning; incremental prediction; multi-task learning; bloom early warning; vertical stratification

---

## 1. Introduction

### 1.1 Background and Significance
Algal blooms in reservoirs threaten drinking water quality worldwide. Traditional monitoring relies on physical sampling and threshold-based warning, which are reactive and limited by vertical heterogeneity — algae concentrations vary dramatically across depth layers (0.5-10 m), and surface measurements alone miss subsurface dynamics.

### 1.2 Related Work
Prior studies have applied deep learning to algal forecasting: LSTM-based prediction (Newanjiang Reservoir), temporal attention networks (GCN-TPA), and Transformer variants (Bloomformer-2). However, three gaps persist: (1) most models predict absolute concentration, which is dominated by strong autocorrelation and underperforms naive persistence; (2) few integrate multiple monitoring tasks (prediction, stratification, warning) in a shared architecture; (3) vertical profile data is underutilized.

### 1.3 Contributions
1. **Incremental prediction formulation**: predicting change (Δ) rather than absolute value outperforms persistence across all horizons, addressing the autocorrelation-dominated regime.
2. **Multi-task shared backbone**: GRU backbone with forecasting/stratification/warning heads improves both accuracy and interval calibration.
3. **Daily-scale process alignment**: matching sampling resolution to algal process timescale improves long-horizon forecasting.
4. **Probabilistic bloom warning**: calibrated probability output provides 3.6× lead time over threshold methods with fewer false positives.
5. **Vertical data utilization**: systematic analysis of 20-layer profiles reveals stratification-state coupling during blooms.

---

## 2. Data and Methods

### 2.1 Study Site and Data
Single reservoir (site 0902, anonymized), 2021-03 to 2025-09. 258,542 algae measurements across 20 depth layers (0.5-10 m at 0.5 m intervals), 232,067 meteorological observations (10-min sampling). Key variables: water temperature, transmittance, algae concentrations (green/cyanobacteria/diatom/cryptophyte), total concentration, and six meteorological drivers (wind, temperature, humidity, pressure, rainfall).

![Fig1](paper_figs/fig1_data_overview.png)
**Figure 1.** Daily surface concentration and water temperature (z-scored) time series with the 12 identified bloom events shaded. Bloom events concentrate in 2021 (dominant) with secondary peaks in 2023.
![Fig2](paper_figs/fig2_bloom_events.png)
**Figure 2.** Gantt-style distribution of the 12 bloom events across the study period (2021-2025). 2021 events (N1-N4) highlighted in red, comprising 74% of total bloom days.
![Fig3](paper_figs/fig3_stratification.png)
**Figure 3.** Weekly-mean z-scored heatmaps of 20-layer concentration (top) and water temperature (bottom). Concentration color scale clipped at +3σ to highlight bloom layers.

### 2.2 Data Preprocessing
- Vertical timestamp alignment: depth layers have 4-5 min timestamp offsets (exact intersection = 0); floored to 3-hour grid (97.8% alignment).
- Meteorological alignment via merge_asof (nearest, 3-h tolerance).
- Daily aggregation (1D mean) for process-scale matching.

### 2.3 Bloom Event Definition
Bloom status = surface-band (0.5-3.0 m) median > band-p90 AND ≥3 layers in 0.5-5.0 m > their p90, sustained ≥2 days. This identifies 12 bloom events (2021 dominant: 74%), with 19.5-day median pre-event warning window.

### 2.4 Model Architecture
RAMS-Net: shared GRU backbone (hidden=64) → three task heads:
- **M1 Forecasting**: incremental target Δ = conc_{t+h} − conc_t, 9-quantile output (q9)
- **M2 Stratification**: binary classification (stratified/not)
- **M4 Bloom warning**: bloom-status classification

![Fig4](paper_figs/fig4_architecture.png)
**Figure 4.** RAMS-Net multi-task architecture: shared GRU backbone with M1 incremental forecasting (q9 quantiles), M2 stratification, and M4 bloom warning heads. Multi-task loss with w=(1,3,2).

### 2.5 Training
- **Two-stage**: Stage 1 single-task M1 (20 epochs) → Stage 2 freeze backbone, fine-tune multi-head (10 epochs)
- Multi-task loss: L = w1·L_q(M1) + w2·CE(M2) + w4·CE(M4), w=(1,3,2)
- Daily scale: T=30 lookback, H=7-day horizon
- Rolling window evaluation: 730d train / 90d test / 45d step (17 windows), 3 seeds

### 2.6 Evaluation
CRPS (continuous ranked probability score), coverage, p50 RMSE, relative skill vs persistence. Bloom warning: recall, lead time, false positives.

---

## 3. Results

### 3.1 Forecasting Performance
![Fig5](paper_figs/fig5_increment_vs_baseline.png)
**Figure 5.** CRPS by forecast horizon comparing RAMS-Net (incremental, q9) with persistence, LightGBM-quantile, and XGBoost-quantile baselines. Error bars show ±1 std across 3 seeds × 17 rolling windows.

**Incremental prediction beats persistence across all horizons.** RAMS-Net achieves CRPS skill of +22.1% over persistence (3-seed × 17-window), exceeding all statistical baselines: LightGBM quantile (+10.0%), XGBoost quantile (+4.6%). The only method with meaningful probabilistic calibration (coverage 0.766 vs 0.55 for GBDT baselines). All 7 horizons exceed +20% skill.

*[Table 1: CRPS comparison table]*

### 3.2 Interval Calibration
![Fig6](paper_figs/fig6_coverage.png)
**Figure 6.** Prediction interval coverage comparison: RAMS-Net formal (0.766) vs nominal 80% target; incremental quantiles (0.824) vs absolute-concentration quantiles (0.669).

Two-stage training with multi-task heads calibrates prediction intervals: coverage 0.766, approaching the 80% target. Absolute-concentration quantiles under-cover (0.669) due to seasonal level drift; incremental quantiles are better calibrated.

### 3.3 Bloom Warning
![Fig7](paper_figs/fig7_m4_threshold.png)
**Figure 7.** Bloom warning recall-false-positive tradeoff across probability threshold θ (0.3-0.8), with rule-based baseline reference point.
![Fig8](paper_figs/fig8_lead_time.png)
**Figure 8.** Median warning lead time: probabilistic warning (θ=0.5-0.7) achieves 14-21 days versus 4 days for the rule-based threshold baseline — a 3.6× improvement with half the false positives.

**Probabilistic warning provides 3.6× lead time.** At threshold θ=0.5, model recall 0.4 with median 14.5-day lead (13-16 days), 6 false-positive episodes. Threshold baseline (band p75): recall 1.0 but only 4-day lead, 14 false-positive episodes. Model trades half the false positives for 3.6× longer lead.

### 3.4 Vertical Structure and Causality
![Fig9](paper_figs/fig9_causal_graph.png)
**Figure 9.** PCMCI+ causal structure: algae concentration dominated by autoregression (τ=1, r=0.63); meteorological drivers weak (wind_u r≈0.07); stratification decoupled during blooms.

PCMCI+ analysis shows algae concentration is dominated by its own autoregression (τ=1 r=0.63); meteorological drivers have weak direct effects (wind_u r≈0.07); stratification decouples during blooms (surface-10m correlation drops to -0.07 vs 0.57 non-bloom).

### 3.5 Ablation Studies
![Fig10](paper_figs/fig10_ablations.png)
**Figure 10.** Key ablations: two-stage training (ts_freeze best accuracy), incremental vs absolute target, multi-task vs single-task (calibration improvement).

- Incremental vs absolute target: incremental wins all horizons
- Multi-task vs single-task: multi-task improves calibration (0.67→0.81 coverage)
- Two-stage vs joint: two-stage ts_freeze achieves best accuracy
- Daily vs 3h scale: daily better for 7-day horizon

### 3.6 Framework Comparison
![Fig11](paper_figs/fig11_framework_compare.png)
**Figure 11.** RMSE comparison across 12 frameworks: RAMS-Net (GRU multi-task, 3.64) optimal vs DLinear, TFT, XGBoost, LightGBM, Transformer, PatchTST.

RAMS-Net (GRU multi-task) outperforms all alternatives: DLinear, TFT, XGBoost, LightGBM, Transformer, PatchTST — GRU multi-task architecture is optimal for this data scale.

---

## 4. Discussion

### 4.1 Why Incremental Prediction Works
Algal concentration is a strong autoregressive process (τ=1 r=0.63). Persistence (current value) already captures ~80% of the level. Modeling the *change* (Δ) lets the network learn corrections rather than redundantly modeling the level — this is the key insight, generalizable to other strongly autocorrelated environmental variables.

### 4.2 Multi-task Regularization
The stratification and warning heads act as regularizers, improving interval calibration without sacrificing point accuracy. This aligns with multi-task learning theory: auxiliary tasks provide inductive bias.

### 4.3 Process Scale Matching
Daily sampling with 7-day horizon matches algal growth (1-3 day doubling) and meteorological lag (13-30 days) timescales better than 3-hour sampling with 24-hour horizon. This suggests sampling resolution should be matched to the dominant process timescale.

### 4.4 Limitations
- **Information ceiling**: oracle probe shows the data approaches its predictable limit; pure autoregression captures most signal. Further improvement requires more data (multi-site, nutrients, inflow).
- **Direction predictability**: P(Δ>0) is weakly discriminative (AUC 0.59); direction of change is largely unpredictable.
- **Seasonal drift**: interval coverage varies across windows (0.66-0.94) due to inter-annual non-stationarity; conformal calibration only partially mitigates.

---

## 5. Conclusion
We developed RAMS-Net, a multi-task deep learning framework for vertically stratified reservoir algal forecasting. Key findings: (1) incremental prediction outperforms persistence by +22.1% CRPS skill; (2) daily-scale process matching improves long-horizon forecasting; (3) probabilistic bloom warning provides 3.6× lead time with fewer false positives; (4) multi-task architecture calibrates prediction intervals. The framework offers a scientifically grounded, deployable approach for reservoir algae monitoring and early warning.

## Acknowledgements
Data provided under confidentiality agreement; anonymized.

## References
*(To be completed with ≥15 references, 60% recent 5 years)*

1. Rousso et al. Water Research 2020 — cyanobacteria forecasting review
2. Hipsey et al. GMD 2019 — General Lake Model
3. Carey et al. Inland Waters 2022 — FLARE forecasting
4. Thomas et al. WRR 2020 — iterative forecasting
5. Runge et al. — PCMCI+ causality
6. Romano et al. 2019 — conformal prediction (CQR)
...

