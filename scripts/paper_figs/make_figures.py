"""Generate publication figures for RAMS (Reservoir Algae Monitoring System) paper.

Target journal style: Ecological Informatics (English labels, publication-quality, 300 dpi).
All figures are saved to docs/paper_figs/ as PNG.

Secrecy: figures show only standardized (z-score) values or aggregated statistics —
no raw concentration values are printed or plotted. Raw parquet is used only to
compute statistical summaries (daily means, standardized series, correlations).

Run:  D:/enviranment/Python313/python.exe scripts/paper_figs/make_figures.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

ROOT = Path(r"D:\coding\26AIendwork")
OUT = ROOT / "docs" / "paper_figs"
OUT.mkdir(parents=True, exist_ok=True)

PARQUET = ROOT / "data" / "processed" / "standard.parquet"

# ----------------------------------------------------------------------------
# Shared style (publication / Ecological Informatics feel)
# ----------------------------------------------------------------------------
plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
    }
)

C_RAMS = "#1f77b4"
C_PERSIST = "#d62728"
C_LGB = "#2ca02c"
C_XGB = "#ff7f0e"
C_GREY = "#7f7f7f"


def _load_json(rel: str) -> dict:
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def _daily_surface() -> pd.DataFrame:
    """Daily-mean surface (0.5 m) concentration + water temperature, standardized."""
    df = pd.read_parquet(PARQUET, columns=["timestamp", "depth", "total_conc", "water_temp"])
    surf = df[df["depth"] == 0.5].copy()
    surf["date"] = surf["timestamp"].dt.normalize()
    daily = surf.groupby("date").agg(
        conc=("total_conc", "mean"),
        temp=("water_temp", "mean"),
    ).reset_index()
    daily["conc_z"] = (daily["conc"] - daily["conc"].mean()) / daily["conc"].std()
    daily["temp_z"] = (daily["temp"] - daily["temp"].mean()) / daily["temp"].std()
    return daily


def _weekly_profile(metric: str) -> tuple[pd.DataFrame, list[float]]:
    """Weekly-mean (time x depth) matrix for a metric ('conc' or 'temp'), standardized."""
    df = pd.read_parquet(PARQUET, columns=["timestamp", "depth", "total_conc", "water_temp"])
    df["date"] = df["timestamp"].dt.normalize()
    pivot = df.pivot_table(index="date", columns="depth", values=metric, aggfunc="mean")
    pivot = pivot.sort_index()
    weekly = pivot.resample("7D").mean()
    depths = sorted(pivot.columns.tolist())
    z = (weekly.values - weekly.values.mean()) / weekly.values.std()
    return pd.DataFrame(z, index=weekly.index, columns=[float(d) for d in depths]), depths


def _bloom_events() -> list[dict]:
    d = _load_json("exp/model_enhancement/n_bloom_identify/results.json")
    evs = d.get("final_event_list", [])
    out = []
    for i, e in enumerate(evs, 1):
        out.append(
            {
                "n": i,
                "start": pd.Timestamp(e["start"]),
                "end": pd.Timestamp(e["end"]),
                "duration_days": e["duration_days"],
                "peak": e["peak"],
            }
        )
    return out


# ============================================================================
# Fig 1  —  Study-area overview: standardized surface concentration & temperature
#          time series with N-defined bloom events shaded
# ============================================================================
def fig1():
    daily = _daily_surface()
    evs = _bloom_events()
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.4), sharex=True)
    for ax, col, ylab, c in (
        (axes[0], "conc_z", "Chl-a conc. (z-score)", C_RAMS),
        (axes[1], "temp_z", "Water temp. (z-score)", C_RAMS),
    ):
        ax.plot(daily["date"], daily[col], lw=0.6, color=c, alpha=0.85)
        ax.set_ylabel(ylab)
        ax.grid(True, which="major")
    for e in evs:
        for ax in axes:
            ax.axvspan(e["start"], e["end"], color="red", alpha=0.12, lw=0)
    axes[0].set_title(
        "Fig. 1 — Daily surface observations at the monitoring station "
        "(2021-03 to 2025-09; shaded: 12 N-defined bloom events)",
        fontsize=9,
    )
    axes[1].set_xlabel("Date")
    axes[1].xaxis.set_major_locator(matplotlib.dates.YearLocator())
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig1_data_overview.png")
    plt.close(fig)


# ============================================================================
# Fig 2  —  Bloom-event temporal distribution (12 events, 2021 dominant)
# ============================================================================
def fig2():
    evs = _bloom_events()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    # Gantt-style timeline: one row per event, x = calendar time
    yticks, labels = [], []
    for i, e in enumerate(evs):
        year = e["start"].year
        ax.barh(i, (e["end"] - e["start"]).days, left=e["start"], height=0.62,
                color="#e85d75" if year == 2021 else "#4c72b0", alpha=0.9)
        yticks.append(i)
        labels.append(f"N{e['n']} ({year})")
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    ax.set_xlabel("Date")
    ax.set_xlim(pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31"))
    ax.set_title("Fig. 2 — Temporal distribution of the 12 N-defined bloom events", fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#e85d75", label="2021 (dominant)"),
                       Patch(color="#4c72b0", label="2022–2025")], loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_bloom_events.png")
    plt.close(fig)


# ============================================================================
# Fig 3  —  Vertical stratification: standardized concentration / temperature
#          heatmap (time x depth), weekly resolution
# ============================================================================
def fig3():
    data_c, depths = _weekly_profile("total_conc")
    data_t, _ = _weekly_profile("water_temp")
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 4.6), sharex=True)
    for ax, Z, met, cmap, vmax in (
        (axes[0], data_c, "Chl-a concentration", "YlGnBu", 3.0),
        (axes[1], data_t, "Water temperature", "RdYlBu_r", None),
    ):
        im = ax.imshow(Z.T, aspect="auto", origin="lower", cmap=cmap, vmin=Z.values.min(), vmax=vmax,
                       extent=[0, len(Z), min(depths), max(depths)])
        ax.set_ylabel("Depth (m)")
        ax.set_yticks(np.arange(0, 10.5, 2.0))
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("z-score")
        ax.set_title(f"{met} (weekly mean, standardized)", fontsize=8, loc="left")
    n_weeks = len(data_c)
    axes[0].set_xticks(np.linspace(0, n_weeks - 1, 5))
    axes[0].set_xticklabels([])
    xt = np.linspace(0, n_weeks - 1, 5).astype(int)
    axes[1].set_xticks(xt)
    axes[1].set_xticklabels([data_c.index[int(i)].strftime("%Y-%m") for i in xt], rotation=20, fontsize=7)
    axes[1].set_xlabel("Time (weekly)  |  2021-03 to 2025-09")
    fig.suptitle("Fig. 3 — Vertical stratification of the water column", fontsize=10, y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_stratification.png")
    plt.close(fig)


# ============================================================================
# Fig 4  —  Model architecture (RAMS-Net, five-task integration)
#           Reuses the existing English architecture diagram
# ============================================================================
def fig4():
    from PIL import Image
    img = Image.open(ROOT / "docs" / "rams_architecture.png")
    w, h = img.size
    fig, ax = plt.subplots(figsize=(9.0, 9.0 * h / w))
    ax.imshow(np.asarray(img))
    ax.axis("off")
    ax.set_title("Fig. 4 — RAMS-Net architecture (shared GRU backbone with M1/M2/M4 heads)",
                 fontsize=9, pad=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_architecture.png")
    plt.close(fig)


# ============================================================================
# Fig 5  —  Incremental forecast vs baselines: CRPS by horizon (daily, H=7)
# ============================================================================
def fig5():
    st = _load_json("exp/model_enhancement/st-train-v020/results.json")
    ml = _load_json("exp/baseline_feasibility/ml_baselines_results.json")
    ph = {p["h"]: p for p in st["per_horizon"]}
    hs = sorted(ph.keys())
    crps_rams = [ph[h]["crps_mean"] for h in hs]
    crps_std = [ph[h]["crps_std"] for h in hs]
    crps_persist = [ph[h]["crps_p"] for h in hs]
    crps_lgb = ml["baselines"]["lgb_q"]["crps_h"]
    crps_xgb = ml["baselines"]["xgb_q"]["crps_h"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    x = np.arange(len(hs))
    ax.plot(x, crps_persist, "o--", color=C_PERSIST, lw=1.3, ms=4, label="Persistence")
    ax.plot(x, crps_lgb, "s-", color=C_LGB, lw=1.1, ms=4, label="LightGBM-quantile")
    ax.plot(x, crps_xgb, "^-", color=C_XGB, lw=1.1, ms=4, label="XGBoost-quantile")
    ax.errorbar(x, crps_rams, yerr=crps_std, fmt="o-", color=C_RAMS, lw=1.6, ms=4.5,
                capsize=2, label="RAMS-Net (incremental, q9)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{h} d" for h in hs])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel("CRPS (conc. units)")
    ax.set_title("Fig. 5 — Incremental forecast skill vs baselines by horizon", fontsize=9)
    ax.legend(frameon=False, loc="upper left")
    ax.set_ylim(0, 2.2)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_increment_vs_baseline.png")
    plt.close(fig)


# ============================================================================
# Fig 6  —  Coverage / calibration: interval-coverage of quantile predictions
#           vs nominal 80% target
# ============================================================================
def fig6():
    # (a) formal daily protocol  (baseline_comparison_results.json)
    bc = _load_json("exp/baseline_feasibility/baseline_comparison_results.json")
    b = bc["statistical_ml_baselines"]
    cov_formal = {
        "RAMS-Net (q9+Δ)": bc["ramsnet"]["coverage_mean"],
        "LightGBM-q": b["lgb_q"]["coverage_mean"],
        "XGBoost-q": b["xgb_q"]["coverage_mean"],
        "Persistence": b["persist"]["coverage_mean"],
    }
    # (b) exploration 3h protocol  (b2_increment_quantile)
    b2 = _load_json("exp/model_enhancement/b2_increment_quantile/results.json")
    cov_expl = {
        "Δ-quantile": b2["agg_coverage"]["inc"],
        "absolute-quantile": b2["agg_coverage"]["abs"],
        "zero": b2["agg_coverage"]["zero"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
    for ax, cov, title in (
        (axes[0], cov_formal, "(a) Daily protocol (rolling 730/90/45)"),
        (axes[1], cov_expl, "(b) Exploration 3-h protocol"),
    ):
        names = list(cov.keys())
        vals = [cov[n] for n in names]
        cols = [C_RAMS if "RAMS" in n or "Δ" in n else C_GREY for n in names]
        ax.bar(range(len(names)), vals, color=cols, width=0.62)
        ax.axhline(0.80, color="k", ls="--", lw=1.0)
        ax.text(len(names) - 0.5, 0.815, "nominal 80%", ha="right", va="bottom", fontsize=7.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Coverage [p10, p90]")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=8)
    fig.suptitle("Fig. 6 — Predictive-interval coverage of quantile forecasts", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig6_coverage.png")
    plt.close(fig)


# ============================================================================
# Fig 7  —  M4 bloom-warning threshold sweep: recall vs false-alarm trade-off
#          (θ sensitivity), with rule-based baseline for context
# ============================================================================
def fig7():
    m4 = _load_json("exp/mdl_m4_warning/results.json")
    sweep = m4["sweep_n_events"]
    th = [s["theta"] for s in sweep]
    rec = [s["recall"] for s in sweep]
    fpr = [s["fpr_days"] for s in sweep]
    lead = [s["lead_median"] for s in sweep]
    bl = m4["baseline_p75"][0]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(fpr, rec, "o-", color=C_RAMS, lw=1.5, ms=5, label="Probabilistic M4 (θ sweep)")
    for t, r, f, l in zip(th, rec, fpr, lead):
        ax.annotate(f"θ={t}", (f, r), textcoords="offset points",
                    xytext=(-2, 7), fontsize=7, color=C_RAMS)
    ax.plot(bl["fpr_days"], bl["recall"], "P", color=C_PERSIST, ms=8,
            label="Rule baseline (band p75)")
    ax.set_xlabel("False-positive rate (fraction of evaluation days)")
    ax.set_ylabel("Event recall")
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fig. 7 — M4 bloom-warning threshold sensitivity (recall vs false alarms)",
                 fontsize=9)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig7_m4_threshold.png")
    plt.close(fig)


# ============================================================================
# Fig 8  —  Lead-time comparison: probabilistic warning vs threshold warning
# ============================================================================
def fig8():
    m4 = _load_json("exp/mdl_m4_warning/results.json")
    sweep = {s["theta"]: s for s in m4["sweep_n_events"]}
    bl = m4["baseline_p75"][0]
    groups = [
        ("Prob. θ=0.5 (rec 0.40)", sweep[0.5]["lead_median"], C_RAMS),
        ("Prob. θ=0.6 (rec 0.40)", sweep[0.6]["lead_median"], C_RAMS),
        ("Prob. θ=0.7 (rec 0.20)", sweep[0.7]["lead_median"], C_RAMS),
        ("Rule baseline (rec 1.00)", bl["lead_median"], C_PERSIST),
    ]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    names = [g[0] for g in groups]
    vals = [g[1] for g in groups]
    cols = [g[2] for g in groups]
    bars = ax.bar(range(len(names)), vals, color=cols, width=0.62)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{vals[i]:.0f} d",
                ha="center", fontsize=8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("Median warning lead time (days)")
    ax.set_ylim(0, 26)
    ax.set_title("Fig. 8 — Warning lead time: probabilistic vs threshold-based warning", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_lead_time.png")
    plt.close(fig)


# ============================================================================
# Fig 9  —  M5 PCMCI+ causal structure (redrawn: 6 nodes, cross-variable edges
#           weighted by mean |MCI|; self-lag strength annotated)
# ============================================================================
def fig9():
    import networkx as nx
    edges = _load_json("docs/m5_pcmci_edges.json")
    # aggregate cross-variable edges (ignore self-lags) by mean |MCI corr|
    pair_strength: dict[tuple[str, str], list[float]] = {}
    self_lag: dict[str, float] = {}
    for e in edges:
        f, t, r = e["from"], e["to"], e["mci_corr"]
        if f == t:
            self_lag[f] = max(self_lag.get(f, 0.0), abs(r))
        else:
            pair_strength.setdefault((f, t), []).append(abs(r))
    nodes = sorted({e["from"] for e in edges} | {e["to"] for e in edges})
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    for (f, t), rs in pair_strength.items():
        w = float(np.mean(rs))
        G.add_edge(f, t, weight=w)
    ws = [G[u][v]["weight"] for u, v in G.edges()]
    wmax = max(ws) if ws else 1.0
    # absolute strength classes (cross-variable causal links are all weak
    # relative to the self-lags shown in the node boxes)
    wstrong = 0.090
    wweak = 0.065
    # manual layout: meteo (left) -> thermal (center) -> algal (right)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    pos = {
        "wind_u": (-1.0, 0.85),
        "humidity": (-1.0, -0.85),
        "temp_surf": (0.0, 0.75),
        "temp_mean": (0.0, -0.75),
        "cyano_surf": (1.0, 0.85),
        "conc_surf": (1.0, -0.85),
    }
    # node boxes with self-lag strength
    for n in nodes:
        x, y = pos[n]
        if n in self_lag:
            lab = f"{n}\nself τ=1: {self_lag[n]:.2f}"
        else:
            lab = n
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - 0.24, y - 0.09), 0.48, 0.18, boxstyle="round,pad=0.01",
            fc="#eef2ff", ec="#4c72b0", lw=1.0))
        ax.text(x, y, lab, ha="center", va="center", fontsize=6.8)
    # reciprocal edges get opposite curvature
    rad = {}
    for u, v in G.edges():
        rad[(u, v)] = 0.15 if (v, u) in G.edges() else 0.06
    for u, v in G.edges():
        w = G[u][v]["weight"]
        if w >= wstrong:
            lw = 1.2 + 3.0 * (w - wstrong) / max(wmax - wstrong, 1e-6)
            ls = "-"
        elif w >= wweak:
            lw = 1.0
            ls = (0, (4, 2))
        else:
            lw = 0.8
            ls = (0, (1, 2))
        r = -rad[(u, v)] if (v, u) in G.edges() else rad[(u, v)]
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=[(u, v)], width=lw,
                               edge_color="#888888", arrows=True, arrowstyle="-|>",
                               arrowsize=13, connectionstyle=f"arc3,rad={r}", style=ls)
    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.15, 1.15)
    # edge weights legend
    ax.plot([], [], color="#888888", lw=2.2, label="strong (|MCI| ≥ 0.09)")
    ax.plot([], [], color="#888888", lw=1.0, ls=(0, (4, 2)), label="medium (0.065–0.09)")
    ax.plot([], [], color="#888888", lw=0.8, ls=(0, (1, 2)), label="weak (< 0.065)")
    ax.legend(loc="lower center", ncol=3, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.07))
    ax.axis("off")
    ax.set_title("Fig. 9 — M5 PCMCI+ causal structure (12-h grid, τ_max=60, α=0.05)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig9_causal_graph.png")
    plt.close(fig)


# ============================================================================
# Fig 10 — Key ablations: (a) single-task vs multi-task vs two-stage CRPS
#           (b) incremental vs absolute vs persistence RMSE
# ============================================================================
def fig10():
    k = _load_json("exp/model_enhancement/k_two_stage/results.json")
    arms = {
        "single": ("Single-task", k["arms"]["single"]["crps"], k["arms"]["single"]["coverage"]),
        "multi": ("Multi-task", k["arms"]["multi"]["crps"], k["arms"]["multi"]["coverage"]),
        "ts_freeze": ("Two-stage (freeze)", k["arms"]["ts_freeze"]["crps"], k["arms"]["ts_freeze"]["coverage"]),
        "ts_full": ("Two-stage (full)", k["arms"]["ts_full"]["crps"], k["arms"]["ts_full"]["coverage"]),
    }
    # B1 increment vs absolute vs persistence RMSE (from b1_increment/results.md, seed 0, 28-fold)
    horizons = [1, 4, 8]
    inc = [1.4949, 2.3160, 1.7858]
    abs_ = [1.923, 2.205, 2.115]
    persist = [1.5349, 2.3801, 1.8223]

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.3))
    # (a)
    ax = axes[0]
    names = [arms[k_][0] for k_ in arms]
    crps = [arms[k_][1] for k_ in arms]
    cov = [arms[k_][2] for k_ in arms]
    bars = ax.bar(range(len(names)), crps, color=["#4c72b0", "#dd8452", C_RAMS, "#8172b3"], width=0.62)
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                f"{crps[i]:.3f}\n(cov {cov[i]:.2f})", ha="center", fontsize=6.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=18, ha="right", fontsize=7)
    ax.set_ylabel("CRPS (conc. units)")
    ax.set_title("(a) Training scheme (3h protocol)", fontsize=8)
    # (b)
    ax = axes[1]
    ax.plot(horizons, inc, "o-", color=C_RAMS, lw=1.5, label="Incremental Δ")
    ax.plot(horizons, abs_, "s-", color="#dd8452", lw=1.3, label="Absolute conc.")
    ax.plot(horizons, persist, "o--", color=C_PERSIST, lw=1.3, label="Persistence")
    ax.set_xlabel("Horizon (3-h steps)")
    ax.set_xticks(horizons)
    ax.set_ylabel("RMSE (conc. units)")
    ax.set_title("(b) Incremental vs absolute target (B1)", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.suptitle("Fig. 10 — Key ablation comparisons", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig10_ablations.png")
    plt.close(fig)


# ============================================================================
# Fig 11 — Framework comparison (12 models, test RMSE, fixed-split protocol)
# ============================================================================
def fig11():
    fc = _load_json("docs/framework_compare_results.json")
    models = fc["models"]
    names = list(models.keys())
    rmse = [models[n]["rmse_mean"] for n in names]
    std = [models[n]["rmse_std"] for n in names]
    # map Chinese-free labels
    label_map = {
        "\u6301\u4e45\u5316": "Persistence",
        "RamsNet(\u5f53\u524d\u67b6\u6784,\u591a\u4efb\u52a1)": "RamsNet (multi-task)",
        "GRU(\u5f53\u524d\u67b6\u6784,\u5355\u4efb\u52a1)": "GRU (single-task)",
        "TFT(\u7b80\u7248)": "TFT (lite)",
        "XGBoost": "XGBoost",
        "LightGBM": "LightGBM",
        "LinearRegression": "LinearRegression",
        "Ridge": "Ridge",
        "DLinear": "DLinear",
        "TSMixer": "TSMixer",
        "Transformer": "Transformer",
        "PatchTST": "PatchTST",
    }
    labels = [label_map.get(n, n) for n in names]
    order = np.argsort(rmse)
    names_s = [labels[i] for i in order]
    rmse_s = [rmse[i] for i in order]
    std_s = [std[i] for i in order]
    cols = []
    for n in names_s:
        if "RamsNet" in n:
            cols.append(C_RAMS)
        elif "Persist" in n:
            cols.append(C_PERSIST)
        else:
            cols.append("#7f8fa6")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    y = np.arange(len(names_s))
    ax.barh(y, rmse_s, xerr=std_s, color=cols, height=0.62, capsize=2)
    ax.set_yticks(y)
    ax.set_yticklabels(names_s, fontsize=7.5)
    ax.set_xlabel("Test RMSE (conc. units, fixed 70/15/15 split)")
    ax.invert_yaxis()
    ax.set_title("Fig. 11 — Framework comparison across 12 forecasting models", fontsize=9)
    ax.set_xlim(0, 9.5)
    fig.subplots_adjust(left=0.30, right=0.97, top=0.90, bottom=0.13)
    fig.savefig(OUT / "fig11_framework_compare.png")
    plt.close(fig)


# ============================================================================
# Fig 12 — Data distribution: histogram of standardized concentration + ACF
# ============================================================================
def fig12():
    daily = _daily_surface()
    conc_z = daily["conc_z"].values
    conc = daily["conc"].values
    # ACF of daily surface concentration
    n = len(conc)
    x = conc - conc.mean()
    var = np.dot(x, x) if np.dot(x, x) > 0 else 1.0
    maxlags = 60
    acf = np.array([np.dot(x[: n - l], x[l:]) / var for l in range(maxlags + 1)])
    # 95% CI band
    ci = 1.96 / np.sqrt(n)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    ax = axes[0]
    ax.hist(conc_z, bins=60, color=C_RAMS, alpha=0.85, density=True)
    xs = np.linspace(conc_z.min(), conc_z.max(), 200)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), "k--", lw=1.2, label="N(0,1)")
    ax.set_xlabel("Surface concentration (z-score)")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.set_title("(a) Distribution", fontsize=8)
    ax = axes[1]
    ax.vlines(np.arange(maxlags + 1), 0, acf, color=C_RAMS, lw=0.8)
    ax.plot(np.arange(maxlags + 1), acf, "o", ms=2, color=C_RAMS)
    ax.fill_between(np.arange(maxlags + 1), -ci, ci, color="k", alpha=0.08)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("(b) Autocorrelation function", fontsize=8)
    fig.suptitle("Fig. 12 — Surface concentration distribution and persistence", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig12_distribution_acf.png")
    plt.close(fig)


def main():
    funcs = [
        ("fig1_data_overview", fig1),
        ("fig2_bloom_events", fig2),
        ("fig3_stratification", fig3),
        ("fig4_architecture", fig4),
        ("fig5_increment_vs_baseline", fig5),
        ("fig6_coverage", fig6),
        ("fig7_m4_threshold", fig7),
        ("fig8_lead_time", fig8),
        ("fig9_causal_graph", fig9),
        ("fig10_ablations", fig10),
        ("fig11_framework_compare", fig11),
        ("fig12_distribution_acf", fig12),
    ]
    for name, fn in funcs:
        try:
            fn()
            print(f"OK  {name}.png")
        except Exception as e:  # noqa: BLE001
            print(f"ERR {name}.png  ->  {e}")
    print(f"\nOutput directory: {OUT}")


if __name__ == "__main__":
    main()
