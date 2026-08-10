# -*- coding: utf-8 -*-
"""N 方向：藻华事件识别与定义（bloom event identification & definition）

核心问题：怎么区分"真藻华事件"和"数据尖峰/噪声"？
  这决定了"事件预警"的目标对象。

关键检验假设：真藻华会整层联动（表层 + 相邻深度同时升高），
  数据尖峰只出现在单层（异常）——用 20 层数据的邻域联动性来区分。

分析角度：
  A. 垂直联动性：尖峰时刻是否同时出现在相邻深度层？真藻华应整层联动，单层尖峰是数据异常
  B. 时间结构：真藻华应有多天持续（上升-平台-回落），尖峰是瞬时（≤1 个采样点）
  C. 气象/水温背景：真藻华通常伴随特定条件（水温、风、分层），尖峰无
  D. 阈值敏感性：不同阈值（p90/p95/p99/绝对浓度）下事件数怎么变

产出：一个"藻华事件定义"（可计算判定规则），并用它重新统计：
  - 数据里有多少个"真藻华事件"？
  - 时间分布（是否还集中在 2021？）
  - 每个事件的前置窗口（暴发前多久有信号？）

保密：数据涉密只输出统计量，不打印原始数据行。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed/standard.parquet")
OUT = Path("exp/model_enhancement/n_bloom_identify")


def fmt(x: float, nd: int = 3) -> float:
    return round(float(x), nd)


def load_pivot() -> pd.DataFrame:
    df = pd.read_parquet(DATA)
    pv = df.pivot_table(index="timestamp", columns="depth", values="total_conc").sort_index()
    return pv


def run() -> dict:
    pv = load_pivot()
    deps = pv.columns.tolist()
    S = pv[0.5]  # 表层
    res: dict = {}

    # ---------- 0. 基本统计 ----------
    res["n_timestamps"] = int(len(pv))
    res["n_depths"] = int(len(deps))
    res["depths"] = deps
    res["time_range"] = [str(pv.index.min()), str(pv.index.max())]

    # 每层统计量
    layer_stats = {}
    for d in deps:
        col = pv[d].dropna()
        layer_stats[float(d)] = {
            "mean": fmt(col.mean()),
            "p50": fmt(col.quantile(0.50)),
            "p90": fmt(col.quantile(0.90)),
            "p95": fmt(col.quantile(0.95)),
            "p99": fmt(col.quantile(0.99)),
            "max": fmt(col.max()),
        }
    res["layer_stats"] = layer_stats

    # ---------- 1. 垂直联动性分析 ----------
    # 对每个时刻，计算"该层是否超过该层 p95"的层数
    # 考察：表层超阈值的时刻，深层同时超阈值的比例 = 联动性
    TOP = 6  # 表层+相邻 5 层（0.5~3.0m）作为"邻近联动带"
    near_deps = deps[:TOP]
    deep_deps = deps[TOP:]

    thresh_p95 = {d: pv[d].quantile(0.95) for d in deps}
    thresh_p90 = {d: pv[d].quantile(0.90) for d in deps}
    thresh_p99 = {d: pv[d].quantile(0.99) for d in deps}

    exceed_p95 = pd.DataFrame({d: (pv[d] > thresh_p95[d]).astype(float) for d in deps})
    exceed_p90 = pd.DataFrame({d: (pv[d] > thresh_p90[d]).astype(float) for d in deps})
    exceed_p99 = pd.DataFrame({d: (pv[d] > thresh_p99[d]).astype(float) for d in deps})

    # 表层超阈值时刻的层联动数分布
    surf_ex_p95 = exceed_p95.index[exceed_p95[0.5] == 1]
    surf_ex_p90 = exceed_p90.index[exceed_p90[0.5] == 1]
    surf_ex_p99 = exceed_p99.index[exceed_p99[0.5] == 1]

    res["surface_exceed_counts"] = {
        "p90": int(len(surf_ex_p90)),
        "p95": int(len(surf_ex_p95)),
        "p99": int(len(surf_ex_p99)),
    }

    # 表层超阈值时，其他层联动分布
    for name, ex, exdf in [
        ("p90", surf_ex_p90, exceed_p90),
        ("p95", surf_ex_p95, exceed_p95),
        ("p99", surf_ex_p99, exceed_p99),
    ]:
        if len(ex) == 0:
            continue
        # 邻近层（1.0-3.0m，5 层）同时超阈值层数分布
        near_ex = exdf.loc[ex, near_deps[1:]].sum(axis=1)
        deep_ex = exdf.loc[ex, deep_deps].sum(axis=1)
        # 全部 19 个非表层联动
        all_ex = exdf.loc[ex, deps[1:]].sum(axis=1)
        res[f"linkage_surface_ex_{name}"] = {
            "n_surface_exceed": int(len(ex)),
            "near_mean_other_layers": fmt(near_ex.mean()),   # 邻近带平均联动层数
            "near_ge3_frac": fmt((near_ex >= 3).mean()),     # 邻近带≥3层联动的比例
            "near_ge5_frac": fmt((near_ex >= 5).mean()),     # 邻近带5层全联动的比例
            "deep_mean": fmt(deep_ex.mean()),
            "all_mean": fmt(all_ex.mean()),
            "all_ge3_frac": fmt((all_ex >= 3).mean()),
        }

    # 表层孤立超阈值的比例：表层超阈值但邻近带(1.0-3.0)无人超阈值
    for name, ex, exdf in [("p90", surf_ex_p90, exceed_p90), ("p95", surf_ex_p95, exceed_p95)]:
        if len(ex) == 0:
            continue
        near_ex = exdf.loc[ex, near_deps[1:]].sum(axis=1)
        res[f"surface_isolated_{name}"] = {
            "isolated_frac": fmt((near_ex == 0).mean()),       # 邻近层全不联动的比例（=疑似尖峰）
            "cooccur_ge1_frac": fmt((near_ex >= 1).mean()),    # 至少1层联动
            "cooccur_ge3_frac": fmt((near_ex >= 3).mean()),
        }

    # 深水层是否更稳定（作为"平台型暴发 vs 尖峰"的对照）
    res["deep_layer_stability"] = {
        "deep_cv": fmt(pv[deep_deps].std().mean() / pv[deep_deps].mean().mean()),
        "surface_cv": fmt(S.std() / S.mean()),
    }

    # ---------- 2. 时间结构分析 ----------
    # 表面超阈值连续运行长度分布（运行长度单位=采样点 3h）
    runs_p95 = _runs_of(exceed_p95[0.5])
    runs_p90 = _runs_of(exceed_p90[0.5])
    runs_p99 = _runs_of(exceed_p99[0.5])

    for name, runs in [("p90", runs_p90), ("p95", runs_p95), ("p99", runs_p99)]:
        res[f"run_length_{name}"] = {
            "n_runs": len(runs),
            "p50_pts": int(np.median(runs)),
            "p75_pts": int(np.quantile(runs, 0.75)),
            "p90_pts": int(np.quantile(runs, 0.90)),
            "max_pts": int(runs.max()),
            "frac_1pt": fmt((runs == 1).mean()),     # 单点瞬时尖峰比例
            "frac_le2pt": fmt((runs <= 2).mean()),   # ≤6h 尖峰比例
            "frac_ge16pt": fmt((runs >= 16).mean()), # ≥2 天持续比例
            "frac_ge32pt": fmt((runs >= 32).mean()), # ≥4 天持续比例
        }

    # ---------- 3. 阈值敏感性 ----------
    # 用"事件"定义探测：不同阈值下"连续超阈值段"事件数
    sens = {}
    for thr_name, thr in [("p90", 0.90), ("p95", 0.95), ("p99", 0.99)]:
        th = pv[0.5].quantile(thr)
        mask = (pv[0.5] > th)
        runs = _runs_of(mask)
        sens[thr_name] = {
            "threshold_value": fmt(th),
            "n_runs_any": int(len(runs)),
            "n_runs_ge1d": int((runs >= 8).sum()),     # ≥1天
            "n_runs_ge2d": int((runs >= 16).sum()),    # ≥2天
            "n_runs_ge3d": int((runs >= 24).sum()),    # ≥3天
            "n_runs_ge5d": int((runs >= 40).sum()),    # ≥5天
        }
    # 绝对浓度阈值
    for abs_th in [15.0, 20.0, 25.0, 30.0, 40.0, 50.0]:
        mask = (pv[0.5] > abs_th)
        runs = _runs_of(mask)
        sens[f"abs_{abs_th:.0f}"] = {
            "threshold_value": abs_th,
            "n_runs_any": int(len(runs)),
            "n_runs_ge1d": int((runs >= 8).sum()),
            "n_runs_ge2d": int((runs >= 16).sum()),
            "n_runs_ge3d": int((runs >= 24).sum()),
            "n_runs_ge5d": int((runs >= 40).sum()),
        }
    res["threshold_sensitivity"] = sens

    # 绝对浓度阈值 × 持续时长的"藻华日/事件"敏感性（对 2021 相对阈值主导更稳健）
    band = pv[[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]].median(axis=1)
    abs_sens = {}
    for abs_th in [10.0, 15.0, 20.0, 25.0, 30.0]:
        mask = (band > abs_th)
        runs = _runs_of(mask)
        abs_sens[f"abs_{abs_th:.0f}"] = {
            "threshold_value": abs_th,
            "bloom_days_total": fmt(mask.sum() / 8),
            "n_runs_any": int(len(runs)),
            "n_runs_ge2d": int((runs >= 16).sum()),
            "by_year_days": {int(y): fmt(c / 8) for y, c in mask.groupby(mask.index.year).sum().items()},
        }
    res["abs_threshold_sensitivity"] = abs_sens

    # ---------- 4. 事件检测（候选定义） ----------
    # 候选判定规则（可计算）：
    #   rule_geo: 表面超阈值 且 邻近带(1.0-3.0m)≥k 层同时超阈值（垂直联动）
    #   rule_dur: 满足 geo 的连续时刻 ≥ D 天（时间持续）
    # 输出事件：起止时间、峰值、峰值时刻、持续天数、涉及的层数
    events = {}
    for k in [1, 2, 3]:
        for dmin_pt in [8, 16, 24]:  # 1天 / 2天 / 3天
            key = f"k{k}_d{dmin_pt//8}d"
            events[key] = detect_events(pv, exceed_p95, near_deps[1:], k, dmin_pt)

    res["event_counts"] = {k: len(v) for k, v in events.items()}
    res["event_samples"] = {
        k: [e | {"start": str(e["start"]), "end": str(e["end"]), "peak_ts": str(e["peak_ts"])}
            for e in v[:12]]
        for k, v in events.items()
    }

    # ---------- 4b. 单层 dropout 特征（表头读数异常） ----------
    # 关键发现：2021-04 的"尖峰"实为表层传感器 dropout（表层 ~0 而深层持续高位）。
    # 计算 dropout 信号：表层 < 0.5×顶层带中位数 且 顶层带 > 15（高层浓度背景）
    band = pv[[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]].median(axis=1)
    drop = (pv[0.5] < 0.5 * band) & (band > 15)
    drop_runs = _runs_of(drop)
    res["surface_dropout"] = {
        "n_timestamps": int(drop.sum()),
        "frac": fmt(drop.mean()),
        "n_runs": int(len(drop_runs)),
        "max_run_pts": int(drop_runs.max()) if len(drop_runs) else 0,
        "p90_run_pts": int(np.quantile(drop_runs, 0.90)) if len(drop_runs) else 0,
        "by_year": {int(y): int(c) for y, c in drop.groupby(drop.index.year).sum().items()},
    }

    # ---------- 5. 选定定义：藻华事件最终判定规则 ----------
    # 论证（详见 results.md）：
    #   (i)  垂直联动：表面 p95 超阈值时刻 86% 都有邻近层联动；但"孤立表层尖峰
    #         (S>p99 而 1-3m<p50)"数量为 0 → 高读数基本都是真的，非传感器虚高。
    #   (ii) 时间结构：p99 极端值全部瞬时（无 ≥1 天持续）→ 持续时长是尖峰/事件的
    #         关键区分度；真藻华是"持续 + 整列联动"。
    #   (iii)表层 dropout：2021-04 的"尖峰"实为表层传感器间歇掉零（表层~0 深层高位，
    #         127/128 次发生在 2021）→ 单用表层会漏报，需用顶层带中位数（对单层稳健）。
    #   (iv) 阈值敏感性：事件数对阈值极敏感（q0.95: 2-3 个 → q0.99: 1 个），
    #         2021 是"单一连续藻华季"（4-6 月连续高位，低于 p50 的间隙最长 7 天）
    #         而非多个离散事件。
    #
    # 藻华状态（3h 时间步）：
    #   顶层带(0.5-3.0m) 中位数 > 带 p90（对单层 dropout 稳健）
    #   AND 0.5-5.0m 带 ≥3 层 > 各自 p90（垂直联动）
    # 藻华事件（时段）：
    #   连续藻华状态 ≥2 天（≥16 点），相邻段间隔 ≤24h（8 点）合并（容忍昼夜回落）
    #
    # 主指标 = bloom_days（每年处于藻华状态的总天数，季节长度），事件列表作为次级输出。
    band_p90 = band.quantile(0.90)
    band_signal = (band > band_p90) & (exceed_p90[list(pv.columns[:11])].sum(axis=1) >= 3)
    band_runs = _runs_of(band_signal)
    bloom_days_total = len(band_runs)  # 采样点计数见下
    bloom_pts = int(band_signal.sum())
    res["bloom_days"] = {
        "total_pts": bloom_pts,
        "total_days": fmt(bloom_pts / 8),
        "by_year": {int(y): fmt(c / 8) for y, c in band_signal.groupby(band_signal.index.year).sum().items()},
        "frac_by_year": {int(y): fmt(c / len(band_signal[str(y)])) for y, c in band_signal.groupby(band_signal.index.year).sum().items()},
    }

    band_events = _column_events(pv, band, band_signal, pv.columns, 3, 16, exceed_p90, gap_pts=8)

    final_events = band_events
    res["final_event_list"] = [
        {**{k: (str(v) if k in ("start", "end", "peak_ts") else v) for k, v in e.items()}}
        for e in final_events
    ]
    res["final_definition"] = {
        "rule": "藻华状态 = 顶层带(0.5-3.0m)中位数 > 带 p90 且 0.5-5.0m 带 ≥3 层 > 各自 p90；"
                "藻华事件 = 连续藻华状态 ≥2 天（≥16 个 3h 点），相邻段间隔 ≤24h 合并。"
                "对表层单层 dropout 稳健（顶层带中位数不受单层尖峰/掉零影响）",
        "n_events": int(len(final_events)),
        "bloom_days_total": fmt(bloom_pts / 8),
    }
    res["band_signal"] = {
        "band_p90": fmt(band_p90),
        "band_p95": fmt(band.quantile(0.95)),
        "n_band_runs_any": int(len(band_runs)),
        "n_band_runs_ge1d": int((band_runs >= 8).sum()),
        "n_band_runs_ge2d": int((band_runs >= 16).sum()),
        "n_band_runs_ge3d": int((band_runs >= 24).sum()),
        "by_year_ge2d": {int(y): int(c) for y, c in _count_runs_by_year(band_signal, pv, 16).items()},
    }
    # 时间分布（按年）
    if final_events:
        years = pd.Series([e["start"].year for e in final_events]).value_counts().sort_index()
        res["final_event_by_year"] = {int(y): int(c) for y, c in years.items()}
        # 峰值统计
        peaks = [e["peak"] for e in final_events]
        res["final_event_peak"] = {
            "p50": fmt(float(np.median(peaks))),
            "min": fmt(float(np.min(peaks))),
            "max": fmt(float(np.max(peaks))),
        }
        # 持续天数
        durs = [e["duration_pt"] / 8 for e in final_events]
        res["final_event_duration_days"] = {
            "p50": fmt(float(np.median(durs))),
            "min": fmt(float(np.min(durs))),
            "max": fmt(float(np.max(durs))),
        }

        # ---------- 6. 前置窗口分析 ----------
        # 前置窗口 = 从"信号第一次持续（≥18h）超过预警阈值（带 p75 / 带 p50）"到事件正式触发的时间。
        # 回看最多 30 天（240 点），反向找最后一个"连续 ≥6 点超阈值"段的起始时刻。
        band_p75 = band.quantile(0.75)
        band_p50 = band.quantile(0.50)
        lead_p75 = []
        lead_p50 = []
        ramp_days = []  # 事件内从首次超带 p90 到峰值的爬升时间
        for ev in final_events:
            i0 = pv.index.get_loc(ev["start"])
            i1 = pv.index.get_loc(ev["end"])
            look = band.iloc[max(0, i0 - 240): i0]
            if len(look) > 0:
                bl = look.values
                for thr, lead_list in [(band_p75, lead_p75), (band_p50, lead_p50)]:
                    above = (bl > thr).astype(int)
                    run = 0
                    first_sustained = None
                    for ii in range(len(bl) - 1, -1, -1):
                        if above[ii]:
                            run += 1
                        else:
                            if run >= 6:
                                first_sustained = look.index[ii + 1]
                                break
                            run = 0
                    if run >= 6:
                        first_sustained = look.index[0]
                    if first_sustained is not None:
                        lead_list.append((ev["start"] - first_sustained).total_seconds() / 86400.0)
            # 爬升时间：事件内 band 首次超带 p90 → 峰值
            if i1 > i0:
                seg_band = band.iloc[i0:i1 + 1]
                first90 = seg_band.index[seg_band > band.quantile(0.90)]
                if len(first90) > 0:
                    ramp_days.append((ev["peak_ts"] - first90[0]).total_seconds() / 86400.0)
        if lead_p75:
            res["leading_window_p75_days"] = {
                "p25": fmt(float(np.percentile(lead_p75, 25))),
                "p50": fmt(float(np.median(lead_p75))),
                "p75": fmt(float(np.percentile(lead_p75, 75))),
                "min": fmt(float(np.min(lead_p75))),
                "max": fmt(float(np.max(lead_p75))),
            }
        if lead_p50:
            res["leading_window_p50_days"] = {
                "p25": fmt(float(np.percentile(lead_p50, 25))),
                "p50": fmt(float(np.median(lead_p50))),
                "p75": fmt(float(np.percentile(lead_p50, 75))),
                "min": fmt(float(np.min(lead_p50))),
                "max": fmt(float(np.max(lead_p50))),
            }
        if ramp_days:
            res["ramp_days_to_peak"] = {
                "p25": fmt(float(np.percentile(ramp_days, 25))),
                "p50": fmt(float(np.median(ramp_days))),
                "p75": fmt(float(np.percentile(ramp_days, 75))),
                "max": fmt(float(np.max(ramp_days))),
            }

        # 前置窗口（基线口径）：事件 start 前 21 天内的顶层带最小值时刻 → start 的天数。
        # 这是"从低谷爬升到触发"的信号提前量（对 2021 连续藻华季，p75/p50 口径饱和不可用）。
        lead_min = []
        per_event_detail = []
        for ev in final_events:
            i0 = pv.index.get_loc(ev["start"])
            i1 = pv.index.get_loc(ev["end"])
            lo = max(0, i0 - 168)
            pre = band.iloc[lo:i0]
            if len(pre) == 0:
                continue
            pre_min = pre.min()
            t_min = pre.idxmin()
            lead = (ev["start"] - t_min).total_seconds() / 86400.0
            lead_min.append(lead)
            seg_band = band.iloc[i0:i1 + 1]
            ramp_to_peak = (ev["peak_ts"] - seg_band.index[0]).total_seconds() / 86400.0 if i1 > i0 else 0.0
            per_event_detail.append({
                "start": str(ev["start"]), "end": str(ev["end"]),
                "duration_days": ev["duration_days"], "peak": ev["peak"],
                "baseline_min": fmt(pre_min), "ratio_start_over_min": fmt(band.iloc[i0] / max(pre_min, 0.01)),
                "lead_from_min_days": fmt(lead), "ramp_to_peak_days": fmt(ramp_to_peak),
            })
        if lead_min:
            res["leading_window_from_min_days"] = {
                "p25": fmt(float(np.percentile(lead_min, 25))),
                "p50": fmt(float(np.median(lead_min))),
                "p75": fmt(float(np.percentile(lead_min, 75))),
                "min": fmt(float(np.min(lead_min))),
                "max": fmt(float(np.max(lead_min))),
            }
            res["event_detail_lead"] = per_event_detail

        # 前置相对：事件开始前 7 天 vs 事件期间的顶层带浓度均值（上升斜率指标）
        pre_means = []
        ev_means = []
        for ev in final_events:
            i0 = pv.index.get_loc(ev["start"])
            i1 = pv.index.get_loc(ev["end"])
            pre = band.iloc[max(0, i0 - 56): i0]
            if len(pre) > 0:
                pre_means.append(pre.mean())
                ev_means.append(band.iloc[i0:i1].mean())
        if pre_means:
            res["event_pre_ev_conc"] = {
                "pre7d_mean": fmt(float(np.mean(pre_means))),
                "event_mean": fmt(float(np.mean(ev_means))),
                "ratio": fmt(float(np.mean(ev_means) / np.mean(pre_means))),
            }

    # ---------- 7. 气象/水温背景 ----------
    # 事件期间 vs 非事件期间的天气/水温/分层条件
    if final_events:
        df = pd.read_parquet(DATA)
        # 气象在时间戳级 join（同一时刻多深度行相同），取每时刻首行
        meteo_cols = ["water_temp", "wind_speed", "air_temp", "rainfall", "pressure"]
        meta = df.groupby("timestamp")[meteo_cols].first()
        # 分层度：表层-深层水温差（从逐层水温 pivot 计算）
        wt = df.pivot_table(index="timestamp", columns="depth", values="water_temp")
        wt = wt.loc[wt.index.isin(pv.index)]
        strat = (wt[0.5] - wt[8.0]).rename("strat_0_8")
        meta = meta.join(strat, how="inner")
        in_event = pd.Series(False, index=pv.index)
        for ev in final_events:
            in_event.loc[ev["start"]: ev["end"]] = True
        meta = meta.loc[meta.index.isin(pv.index)]
        ie = meta.loc[in_event]
        no = meta.loc[~in_event]
        res["meteo_context"] = {}
        for col in list(meteo_cols) + ["strat_0_8"]:
            if ie[col].notna().sum() == 0 or no[col].notna().sum() == 0:
                continue
            res["meteo_context"][col] = {
                "event_mean": fmt(ie[col].mean()),
                "nonevent_mean": fmt(no[col].mean()),
                "event_p25": fmt(ie[col].quantile(0.25)),
                "event_p75": fmt(ie[col].quantile(0.75)),
                "nonevent_p25": fmt(no[col].quantile(0.25)),
                "nonevent_p75": fmt(no[col].quantile(0.75)),
            }

    return res


def _runs_of(mask: pd.Series) -> np.ndarray:
    """连续 True 运行长度（采样点个数）。"""
    m = mask.fillna(False).astype(bool).values
    runs = []
    cur = 0
    for v in m:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    return np.array(runs, dtype=float)


def detect_events(pv: pd.DataFrame, exceed: pd.DataFrame, near_deps, k: int, dmin_pt: int) -> list:
    """事件检测：表层超 p95 且 邻近带 ≥k 层超 p95，且连续 ≥ dmin_pt 点。

    合并相邻事件（间隔 ≤ 3 点 = 9h 视为同一事件延续）。
    """
    mask = (exceed[0.5] == 1) & (exceed[near_deps].sum(axis=1) >= k)
    m = mask.values
    events = []
    i = 0
    n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            events.append([i, j])
            i = j
        else:
            i += 1
    # 合并间隔 ≤ 3 点的相邻事件
    if not events:
        return []
    merged = [events[0]]
    for seg in events[1:]:
        if seg[0] - merged[-1][1] <= 3:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    out = []
    for a, b in merged:
        if b - a < dmin_pt:
            continue
        seg = pv.iloc[a:b]
        peak_idx = seg[0.5].idxmax()
        peak_val = float(seg[0.5].max())
        # 事件内超过阈值的平均层数
        nlayer = exceed.loc[seg.index, near_deps].sum(axis=1).mean()
        out.append({
            "start": pv.index[a],
            "end": pv.index[b - 1],
            "duration_pt": int(b - a),
            "duration_days": fmt((b - a) / 8),
            "peak": peak_val,
            "peak_ts": peak_idx,
            "mean_near_layers": fmt(nlayer),
        })
    return out


def _count_runs_by_year(mask: pd.Series, pv: pd.DataFrame, min_pts: int) -> dict:
    """按年统计满足 min_pts 持续时长的连续段数。"""
    m = mask.values
    idx = pv.index
    runs = []
    i = 0
    n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            if j - i >= min_pts:
                runs.append(idx[i].year)
            i = j
        else:
            i += 1
    return {int(y): int(c) for y, c in pd.Series(runs).value_counts().sort_index().items()}


def _column_events(pv: pd.DataFrame, band: pd.Series, signal: pd.Series,
                   columns, k: int, dmin_pt: int, exceed: pd.DataFrame, gap_pts: int = 3) -> list:
    """事件检测（dropout 稳健版）：顶层带中位数超阈值 且 垂直联动层数 ≥k。

    事件信号本身来自 band（对单层 dropout 稳健），另附加垂直联动条件：
    在事件窗口内，0.5-5.0m 带 ≥k 层同时 > 各自 p90。
    合并相邻事件（间隔 ≤ gap_pts 点视为同一事件延续，容忍昼夜回落）。
    """
    m = signal.values
    events = []
    i = 0
    n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            events.append([i, j])
            i = j
        else:
            i += 1
    if not events:
        return []
    merged = [events[0]]
    for seg in events[1:]:
        if seg[0] - merged[-1][1] <= gap_pts:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    top_band = columns[:11]  # 0.5-5.0m
    out = []
    for a, b in merged:
        if b - a < dmin_pt:
            continue
        seg = pv.iloc[a:b]
        # 垂直联动层数：事件窗口内 > 各自 p90 的平均层数（对 0.5-5.0m 带）
        nlayer = exceed.loc[seg.index, top_band].sum(axis=1).mean()
        if nlayer < k:
            continue
        peak_idx = seg[0.5].idxmax()
        peak_val = float(seg[0.5].max())
        out.append({
            "start": pv.index[a],
            "end": pv.index[b - 1],
            "duration_pt": int(b - a),
            "duration_days": fmt((b - a) / 8),
            "peak": peak_val,
            "peak_ts": peak_idx,
            "mean_col_layers": fmt(nlayer),
        })
    return out


if __name__ == "__main__":
    result = run()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    # 终端摘要
    print(json.dumps({
        "surface_exceed_counts": result.get("surface_exceed_counts"),
        "linkage_p95": result.get("linkage_surface_ex_p95"),
        "isolated_p95": result.get("surface_isolated_p95"),
        "run_length_p95": result.get("run_length_p95"),
        "surface_dropout": result.get("surface_dropout"),
        "band_signal": result.get("band_signal"),
        "event_counts": result.get("event_counts"),
        "final_definition": result.get("final_definition"),
        "final_event_by_year": result.get("final_event_by_year"),
        "leading_window_p75_days": result.get("leading_window_p75_days"),
    }, ensure_ascii=False, indent=2))
    print("saved ->", OUT / "results.json")
