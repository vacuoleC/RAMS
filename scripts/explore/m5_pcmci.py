#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5: 气象-藻类 机理与时滞分析（PCMCI+）— 优化版

在 standard.parquet 的 12h 网格序列上，用 tigramite 的 PCMCI+ 做因果发现。
（3h 对水库过程是过采样；12h 均值聚合保持物理尺度并大幅压缩滞后数。）

性能优化（对照 tigramite 官方 discussion #208：总成本 ∝ 链接数 × 条件数）：
  1. 降采样到 12h（`df.resample('12h').mean()`，均值聚合）。
  2. ACF 定 tau_max：先算关键目标（表层总浓度/蓝藻）自相关衰减到噪声水平的
     滞后 k，取 tau_max = min(max_lag, 2*k)。
  3. 两阶段粗筛 + link_assumptions 剪枝：先用便宜的条件偏相关扫描每对
     (源, 目标) 在 1..tau_max 上的信号，每个目标只保留 top 5 源 × top 3 滞后
     作为候选父链接，link_assumptions 之外的链接视为不存在（候选边从 2.8 万
     级砍到几百级）。
  4. 约束参数：max_conds_dim=5, max_conds_px=5, max_conds_py=5。
  5. 季节变量无条件作为全部目标的候选父链接（吸收季节混杂）。

变量集：季节编码(sin/cos) + 表层总浓度 + 表层蓝藻 + 表层水温 + 20 层平均水温
       + 风(u/v 分量) + 气温 + 湿度 + 降雨 + 气压。
目标变量：表层总浓度、表层蓝藻、表层水温、平均水温。
条件独立性检验：ParCorr（线性、快）；可选 --method cmiknn（非线性、慢）。

保密红线：只输出变量名 / 时滞 / 统计量（MCI 相关），绝不打印原始数据数值行。

用法（在算力机上）：
  python3 scripts/explore/m5_pcmci.py --step-h 12               # 默认，ACF 定 tau_max
  python3 scripts/explore/m5_pcmci.py --step-h 12 --tau-max 60  # 强制 60 步(=30天)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 无显示环境
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tigramite import plotting as tp
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.independence_tests.cmiknn import CMIknn

DEPTHS = [0.5 + 0.5 * i for i in range(20)]  # 0.5 ~ 10.0 m
SURF = 0.5
GRID = "3h"

METEO_COLS = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]

# 变量顺序（0-index 与结果矩阵对齐）
VAR_ORDER = [
    "season_sin",    # 季节正弦
    "season_cos",    # 季节余弦
    "conc_surf",     # 表层总浓度（目标）
    "cyano_surf",    # 表层蓝藻浓度（目标）
    "temp_surf",     # 表层水温（目标）
    "temp_mean",     # 20 层平均水温（目标）
    "wind_u",        # 风东西分量
    "wind_v",        # 风南北分量
    "air_temp",      # 气温
    "humidity",      # 湿度
    "rainfall",      # 降雨
    "pressure",      # 气压
]
SELECTED = ["conc_surf", "cyano_surf", "temp_surf", "temp_mean"]
METEO = ["wind_u", "wind_v", "air_temp", "humidity", "rainfall", "pressure"]
SEASON = ["season_sin", "season_cos"]
# 粗筛参数
TOP_SOURCES = 5     # 每个目标保留的源数
TOP_LAGS = 3        # 每个 (源,目标) 保留的显著滞后数
NOISE_CL = 2.0      # 噪声阈值系数（2/sqrt(N)）
STEP_H = 12         # 运行时步长（小时）


def load_wide(parquet_path: str, step_h: int) -> pd.DataFrame:
    """standard.parquet 长表 → step_h 小时网格宽表（均值聚合）。

    返回列：temp_surf, temp_mean, conc_surf, cyano_surf + wind_u/v + 4 气象列。
    """
    df = pd.read_parquet(parquet_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["ts3h"] = df["timestamp"].dt.floor(GRID)

    agg = (
        df.groupby(["ts3h", "depth"])
        .agg(
            water_temp=("water_temp", "mean"),
            total_conc=("total_conc", "mean"),
            cyano_conc=("cyano_conc", "mean"),
        )
        .reset_index()
    )
    pv_t = agg.pivot_table(index="ts3h", columns="depth", values="water_temp")
    pv_c = agg.pivot_table(index="ts3h", columns="depth", values="total_conc")
    pv_cy = agg.pivot_table(index="ts3h", columns="depth", values="cyano_conc")
    pv_t.columns = [float(c) for c in pv_t.columns]

    wide = pd.DataFrame(index=pv_t.index)
    wide["temp_surf"] = pv_t[SURF]
    wide["temp_mean"] = pv_t.mean(axis=1)  # 20 层平均水温
    wide["conc_surf"] = pv_c[SURF]
    wide["cyano_surf"] = pv_cy[SURF]

    meteo = df.drop_duplicates("ts3h")[["ts3h"] + METEO_COLS].set_index("ts3h")
    wide = wide.join(meteo)

    # 先补全 3h 网格再插值，之后降采样到 step_h（均值聚合）
    full = pd.date_range(wide.index.min(), wide.index.max(), freq=GRID)
    wide = wide.reindex(full)
    wide = wide.interpolate(method="time", limit_direction="both")

    # 风向为环形量，先在 3h 转 u/v 分量再聚合，避免对角度做简单平均
    dir_rad = np.deg2rad(wide["wind_dir"])
    wide["wind_u"] = wide["wind_speed"] * np.cos(dir_rad)
    wide["wind_v"] = wide["wind_speed"] * np.sin(dir_rad)
    wide = wide.drop(columns=["wind_speed", "wind_dir"])

    wide = wide.resample(f"{step_h}h").mean().dropna()
    return wide


def add_season(wide: pd.DataFrame) -> pd.DataFrame:
    """在降采样后的索引上做季节循环编码（sin/cos of 小数 doy）。"""
    out = wide.copy()
    t = pd.Series(out.index)
    frac_doy = t.dt.dayofyear + (t.dt.hour + t.dt.minute / 60.0) / 24.0
    # frac_doy 是 RangeIndex 的 Series，直接赋值会因索引不对齐变成全 NaN，用 .values
    out["season_sin"] = np.sin(2 * np.pi * frac_doy / 365.25).values
    out["season_cos"] = np.cos(2 * np.pi * frac_doy / 365.25).values
    return out


def preprocess(wide: pd.DataFrame, log_conc: bool = True) -> pd.DataFrame:
    """可选 log1p 浓度 → 逐列 z-score 标准化（ParCorr 需要同尺度、线性更稳）。"""
    out = wide.copy()
    if log_conc:
        for c in ("conc_surf", "cyano_surf"):
            out[c] = np.log1p(out[c].clip(lower=0.0))
    mu, sd = out.mean(), out.std()
    out = (out - mu) / (sd + 1e-12)
    return out


def acf_est_tau_max(data2d: np.ndarray, names, max_lag: int):
    """用关键目标的自相关衰减估计滞后上界 k；返回 (tau_max, ks, thresh)。"""
    N = data2d.shape[0]
    thresh = NOISE_CL / np.sqrt(N)
    ks = {}
    for name in ("conc_surf", "cyano_surf"):
        i = names.index(name)
        x = data2d[:, i]
        xc = x - x.mean()
        acf = np.correlate(xc, xc, "full")[N - 1: N + max_lag] / np.dot(xc, xc)
        below = np.where(np.abs(acf[1:]) < thresh)[0] + 1
        ks[name] = int(below[0]) if len(below) else int(max_lag)
    k_est = max(ks.values())
    tau_max = min(int(max_lag), 2 * k_est)
    return tau_max, ks, thresh


def _partial_corr(ry: np.ndarray, rx: np.ndarray) -> float:
    """两残差序列的相关系数（已各自对同一条件集回归取残差）。"""
    denom = np.sqrt(np.dot(rx, rx) * np.dot(ry, ry))
    if denom == 0:
        return 0.0
    return float(np.dot(rx, ry) / denom)


def screen_candidates(data2d: np.ndarray, names, tau_max: int,
                      targets=SELECTED) -> tuple[dict, dict]:
    """粗筛：对每个目标，条件偏相关扫描所有 (源, 滞后) 候选。

    对齐约定：回归只取 t = tau_max..N-1（避免越界），条件集固定为
      {1, 目标自身过去 1/2 滞后, 季节 sin/cos}。
    每个目标保留 top 源 × top 显著滞后 作为滞后父链接候选；
    季节变量无条件全滞后保留（吸收季节混杂）。
    返回 (cands, stats)，cands = {j: {(i, -tau)}} 仅含滞后候选（不含 lag0）。
    """
    N = data2d.shape[0]
    thresh = NOISE_CL / np.sqrt(N)
    ti_season = [names.index(s) for s in SEASON]
    ti_selected = [names.index(s) for s in targets]
    n_vars = data2d.shape[1]
    M = N - tau_max  # 有效回归行数

    cands: dict[int, set] = {}
    stats = {"screened_edges": 0, "kept_edges": 0}

    for j in range(n_vars):
        if j not in ti_selected:
            continue
        y = data2d[:, j]
        y_reg = y[tau_max:]                      # t = tau_max..N-1
        # 固定条件集（t = tau_max..N-1）：自身过去 1/2 滞后 + 季节
        C = np.column_stack([
            y[tau_max - 1: N - 1],               # y[t-1]
            y[tau_max - 2: N - 2],               # y[t-2]
            data2d[tau_max:, ti_season[0]],      # season_sin[t]
            data2d[tau_max:, ti_season[1]],      # season_cos[t]
        ])
        Xc = np.column_stack([np.ones(M), C])
        ry = y_reg - Xc @ np.linalg.lstsq(Xc, y_reg, rcond=None)[0]

        scores = []  # (max|pc|, i, [(tau, pc)])
        for i in range(n_vars):
            if i in ti_season:
                continue
            x = data2d[:, i]
            tau_pcs = []
            for tau in range(1, tau_max + 1):
                stats["screened_edges"] += 1
                x_reg = x[tau_max - tau: N - tau]    # x[t-tau], t=tau_max..N-1
                rx = x_reg - Xc @ np.linalg.lstsq(Xc, x_reg, rcond=None)[0]
                pc = _partial_corr(ry, rx)
                if abs(pc) > thresh:
                    tau_pcs.append((tau, pc))
            if not tau_pcs:
                continue
            maxpc = max(abs(pc) for _, pc in tau_pcs)
            tau_pcs.sort(key=lambda tp: -abs(tp[1]))
            scores.append((maxpc, i, tau_pcs[:TOP_LAGS]))
        scores.sort(key=lambda s: -s[0])

        cands[j] = set()
        for maxpc, i, tps in scores[:TOP_SOURCES]:
            for tau, pc in tps:
                cands[j].add((i, -tau))
        # 季节变量：全滞后候选（吸收季节混杂）
        for s in ti_season:
            for tau in range(1, tau_max + 1):
                cands[j].add((s, -tau))
        # 自身滞后 1-2 强制候选（强自相关必须能建模）
        cands[j].add((j, -1))
        cands[j].add((j, -2))
        stats["kept_edges"] += len(cands[j])

    return cands, stats


def build_link_assumptions(cands, names, tau_max,
                           selected=SELECTED, meteo=METEO, season=SEASON):
    """由粗筛候选构造完整 link_assumptions（所有变量作为 target）。

    约定（link_assumptions 之外的链接一律视为不存在）：
      - 所有 target：同层边 (i,0)='o?o' 对全部 i（PCMCI+ 自行定向）。
      - 生物目标（SELECTED）：滞后候选 = 粗筛 cands（含季节/自身）。
      - 气象目标：滞后候选 = 自身 -1/-2 + 季节全滞后（外生、内部结构最小）。
      - 季节目标（season_sin/cos）：无任何父链接（纯外生）。
    """
    n_vars = len(names)
    i_selected = {names.index(s) for s in selected}
    i_meteo = {names.index(m) for m in meteo}
    i_season = {names.index(s) for s in season}

    link_assumptions: dict[int, dict] = {}
    for j in range(n_vars):
        d = {}
        for i in range(n_vars):
            d[(i, 0)] = "o?o"           # 同层边全部候选
        if j in i_selected:
            for (i, tau) in cands[j]:
                d[(i, tau)] = "-?>"
        elif j in i_meteo:
            d[(j, -1)] = "-?>"
            d[(j, -2)] = "-?>"
            for s in i_season:
                for tau in range(1, tau_max + 1):
                    d[(s, -tau)] = "-?>"
        # else: season 目标无父链接（纯外生）
        link_assumptions[j] = d
    return link_assumptions


def run_pcmci(data2d: np.ndarray, tau_min, tau_max, alpha, method,
              link_assumptions, max_samples=None):
    """运行 PCMCI+（link_assumptions=None 时为全候选空间）。

    注意：link_assumptions 用 '-?>' 时，tigramite 5.2 会在最终 graph 里保留
    未经过 MCI 检验的边（val=0, p=1 的伪边）。因此默认跑全候选空间
    （link_assumptions=None），保证每条边都有真实 MCI 统计量。
    """
    if max_samples is not None:
        data2d = data2d[-max_samples:]
        print(f"[info] 仅用最后 {max_samples} 个时刻")
    dframe = pp.DataFrame(data2d)  # v5.2：全部连续变量无需 data_type

    if method == "parcorr":
        cond = ParCorr(significance="analytic")
    else:
        cond = CMIknn(significance="shuffle_test", knn=0.2)
    pcmci = PCMCI(dataframe=dframe, cond_ind_test=cond, verbosity=1)

    results = pcmci.run_pcmciplus(
        link_assumptions=link_assumptions,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=alpha,
        fdr_method="fdr_bh",
        max_conds_dim=5,
        max_conds_py=5,
        max_conds_px=5,
    )
    return pcmci, results


def collect_edges(results, var_names, step_h, alpha=0.05) -> list[dict]:
    """解析 PCMCI+ graph（string 数组）中的显著滞后边（'-->'）。

    只保留经过 MCI 检验且显著（p<alpha）的边，过滤 link_assumptions
    可能引入的 val=0/p=1 伪边。
    """
    graph = results["graph"]
    val = results["val_matrix"]
    pmat = results["p_matrix"]
    edges = []
    for j in range(graph.shape[0]):
        if var_names[j] not in SELECTED:
            continue
        for i in range(graph.shape[0]):
            for tau in range(1, graph.shape[2]):
                if graph[i, j, tau] == "-->" and pmat[j, i, tau] < alpha:
                    edges.append(
                        {
                            "from": var_names[i],
                            "to": var_names[j],
                            "lag": int(tau),
                            "lag_hours": int(tau * step_h),
                            "mci_corr": float(val[j, i, tau]),
                            "p": float(pmat[j, i, tau]),
                        }
                    )
    edges.sort(key=lambda e: -abs(e["mci_corr"]))
    return edges


def print_edges(edges, step_h):
    print("\n" + "=" * 78)
    print(f"显著因果边（MCI 相关 |r| 降序）  [τ = 时滞；{step_h}h/步；PCMCI+ FDR 已剪枝]")
    print("=" * 78)
    for e in edges:
        print(f"  {e['from']:<10s} -> {e['to']:<10s}  τ={e['lag']:>4d} "
              f"({e['lag_hours']:>4d}h = {e['lag_hours']/24:6.1f}d)  "
              f"r={e['mci_corr']:+.3f}  p={e['p']:.2e}")


def summarize_drivers(edges, var_names):
    targets = {}
    for e in edges:
        targets.setdefault(e["to"], []).append(e)
    print("\n" + "=" * 78)
    print("驱动因子排序（按目标变量分组）")
    print("=" * 78)
    for tname in SELECTED:
        es = targets.get(tname, [])
        print(f"\n◆ {tname}  <-  {len(es)} 个显著滞后驱动因子:")
        for i, e in enumerate(es[:12], 1):
            flag = " [水温中介]" if e["from"] in ("temp_surf", "temp_mean") else ""
            flag += " [气象直驱]" if e["from"] in METEO else ""
            flag += " [季节]" if e["from"] in SEASON else ""
            print(f"   {i:>2}. {e['from']:<10s} τ={e['lag']:>4d} "
                  f"({e['lag_hours']/24:5.1f}d)  r={e['mci_corr']:+.3f}{flag}")


def save_report(out_dir, edges, var_names, args, tau_max, meta):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M5: 气象-藻类 机理与时滞分析（PCMCI+，优化版）",
        "",
        f"- 数据: `standard.parquet`，降采样 {meta['step_h']}h 网格（均值聚合），"
        f"{meta['n']} 时刻（{meta['t0']} ~ {meta['t1']}）",
        f"- 方法: PCMCI+（tigramite {args.method}），tau_max={tau_max} "
        f"（{tau_max*meta['step_h']/24:.0f} 天），alpha={args.alpha}",
        f"- ACF 定滞后上界: {meta['acf_ks']}（|ACF|<{meta['thresh']:.3f}），"
        f"tau_max = min({args.max_lag}, 2*k) = {tau_max}",
        f"- 候选边剪枝: 全图初始 {meta['full_edges']} 条 → 扫描 "
        f"{meta['screened']} 条 → PCMCI+ 运行于 "
        f"{'link_assumptions 剪枝（' + str(meta['kept']) + ' 条）' if args.use_screened else '全候选空间'}",
        f"- 约束: max_conds_dim=5, max_conds_px=5, max_conds_py=5",
        f"- 运行耗时: {meta['runtime_min']:.1f} 分钟",
        f"- 变量: {len(var_names)} 个（季节编码 + 藻类 + 水温 + 气象）",
        f"- 目标: {', '.join(SELECTED)}",
        "",
        "## 显著因果边（变量 / 时滞 / MCI 相关 / p）",
        "",
        "| from | to | τ(步) | 滞后 | MCI r | p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for e in edges:
        lines.append(
            f"| {e['from']} | {e['to']} | {e['lag']} | "
            f"{e['lag_hours']/24:.1f}d | {e['mci_corr']:+.3f} | {e['p']:.1e} |"
        )
    lines += ["", "## 说明",
              "- 保密：仅报告变量名/时滞/统计量，不含原始数据值。",
              f"- 滞后 = τ×{meta['step_h']}h；1 天 = {24//meta['step_h']} 个 τ。",
              "- 文献参考滞后（降水 13-20d、风 20-29d、气温 25-30d）。"]
    (out_dir / "m5_pcmci_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out_dir / "m5_pcmci_edges.json").write_text(
        json.dumps(edges, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report] 已写入 {out_dir / 'm5_pcmci_results.md'} 与 "
          f"{out_dir / 'm5_pcmci_edges.json'}")


def plot_graph(results, var_names, tau_max, step_h, out_dir):
    try:
        graph = results["graph"]
        val = results["val_matrix"]
        idx = [var_names.index(v) for v in SELECTED]
        graph_s = graph[np.ix_(idx, idx)]
        val_s = val[np.ix_(idx, idx)]
        names = [var_names[i] for i in idx]

        fig, ax = plt.subplots(figsize=(16, 6))
        tp.plot_time_series_graph(
            graph=graph_s, val_matrix=val_s, var_names=names,
            fig_ax=(fig, ax), label_fontsize=9, arrow_linewidth=1.2,
        )
        ax.set_title(f"PCMCI+ time-lag causal graph (targets: algae/water-temp; "
                     f"tau_max={tau_max}={tau_max*step_h/24:.0f}d)", fontsize=9)
        out = Path(out_dir) / "m5_pcmci_graph.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[plot] 因果图已保存 {out}")
    except Exception as exc:
        print(f"[warn] 绘图跳过: {exc}")


def main():
    ap = argparse.ArgumentParser(description="M5 PCMCI+ 因果时滞分析（优化版）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--step-h", type=int, default=12, help="降采样步长(小时)")
    ap.add_argument("--tau-min", type=int, default=1,
                    help="最小滞后；默认 1（只做滞后因果，聚焦时滞效应；"
                         "设 0 会额外检验同层边，显著更慢）")
    ap.add_argument("--max-lag", type=int, default=192,
                    help="ACF 扫描上限与 tau_max 上限（12h 步 = 96 天）")
    ap.add_argument("--tau-max", type=int, default=None,
                    help="强制 tau_max（覆盖 ACF 估计值），如 60 = 30 天")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--method", choices=["parcorr", "cmiknn"], default="parcorr")
    ap.add_argument("--max-samples", type=int, default=None, help="快速验证时截断样本")
    ap.add_argument("--use-screened", action="store_true",
                    help="用粗筛+link_assumptions 剪枝（更快但可能引入伪边，默认关闭）")
    ap.add_argument("--no-log-conc", action="store_true", help="不对浓度取 log1p")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args()
    t_start = time.time()

    print(f"[1/5] 读取、插值并降采样到 {args.step_h}h ...")
    wide = load_wide(args.parquet, args.step_h)
    wide = add_season(wide)
    data = preprocess(wide, log_conc=not args.no_log_conc)
    data = data[VAR_ORDER]
    print(f"      宽表: {data.shape[0]} 时刻 × {data.shape[1]} 变量, "
          f"范围 {data.index.min()} ~ {data.index.max()}")

    print("[2/5] ACF 估计滞后上界 ...")
    tau_max, acf_ks, thresh = acf_est_tau_max(
        data.values.astype(np.float32), VAR_ORDER, args.max_lag)
    print(f"      ACF 衰减滞后 k={acf_ks}（阈值 {thresh:.3f}），tau_max={tau_max} "
          f"（{tau_max*args.step_h/24:.0f} 天）")
    if args.tau_max is not None:
        tau_max = args.tau_max
        print(f"      [override] tau_max={tau_max}（{tau_max*args.step_h/24:.0f} 天）")

    print("[3/5] 粗筛（仅信息性，默认不用于剪枝） ...")
    cands, stats = screen_candidates(
        data.values.astype(np.float32), VAR_ORDER, tau_max)
    link_assumptions = None if not args.use_screened else \
        build_link_assumptions(cands, VAR_ORDER, tau_max)
    n_vars = data.shape[1]
    full_edges = n_vars * n_vars * tau_max
    if args.use_screened:
        kept = sum(len(d) for d in link_assumptions.values())
        print(f"      全图候选 {full_edges} → 扫描 {stats['screened_edges']} → "
              f"link_assumptions 保留 {kept}（剪枝 {full_edges/kept:.1f}×）")
    else:
        print(f"      全候选空间 PCMCI+（{full_edges} 条滞后候选，"
              f"粗筛仅报告 {stats['screened_edges']} 条扫描量）")
        kept = full_edges

    print("[4/5] 运行 PCMCI+ ...")
    pcmci, results = run_pcmci(
        data.values.astype(np.float32),
        args.tau_min, tau_max, args.alpha, args.method, link_assumptions,
        args.max_samples)

    print("[5/5] 收集结果 + 报告 ...")
    edges = collect_edges(results, VAR_ORDER, args.step_h, args.alpha)
    print_edges(edges, args.step_h)
    summarize_drivers(edges, VAR_ORDER)

    meta = dict(
        step_h=args.step_h, n=len(data), t0=str(data.index.min()),
        t1=str(data.index.max()), acf_ks=acf_ks, thresh=thresh,
        full_edges=int(full_edges), screened=stats["screened_edges"],
        kept=kept,
        runtime_min=(time.time() - t_start) / 60.0,
    )
    save_report(args.out_dir, edges, VAR_ORDER, args, tau_max, meta)
    plot_graph(results, VAR_ORDER, tau_max, args.step_h, args.out_dir)

    print(f"\n== 完成。M5 结论见 docs/m5_pcmci_results.md "
          f"（耗时 {meta['runtime_min']:.1f} 分钟）==")


if __name__ == "__main__":
    main()
