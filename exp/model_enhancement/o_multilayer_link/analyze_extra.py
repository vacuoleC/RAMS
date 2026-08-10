# -*- coding: utf-8 -*-
"""O 方向补充分析：剔除昼夜周期伪影后的滞后关系 + 用 N 的正式事件列表做事件锚定。

背景：run_multilayer.py 的 3h 滞后曲线在"整数天"处出现峰值、半天处出现谷值，
这是表层 24h 昼夜周期（表层 24h 频带功率占 16%）与深层微弱昼夜分量共享引起的
滞后混叠伪影，不是真实的"深层领先"信号。本脚本用每日均值残差重算滞后相关，
并用 N 方向 12 个正式事件的起止做深层/表层首次超 p90 的先后比较。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed/standard.parquet")
OUT = Path("exp/model_enhancement/o_multilayer_link")
N_RES = Path("exp/model_enhancement/n_bloom_identify/results.json")

DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
          5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
PT_PER_DAY = 8


def fmt(x, nd=3):
    return round(float(x), nd)


def detrend(series, win_days=30):
    win = win_days * PT_PER_DAY
    base = series.rolling(win, center=True, min_periods=PT_PER_DAY * 7).median()
    base = base.fillna(series.rolling(win, center=False, min_periods=PT_PER_DAY * 7).median())
    base = base.fillna(series.median())
    return series - base


def scan_daily(a_daily, b_daily, h_days):
    best = None
    curve = {}
    for hd in h_days:
        c = a_daily.corr(b_daily.shift(-hd))  # h>0: a 领先 b
        curve[int(hd)] = fmt(c)
        if best is None or abs(c) > abs(best[1]):
            best = (int(hd), c)
    return {"argmax_h_days": best[0], "max_corr": fmt(best[1]), "curve": curve}


def run():
    df = pd.read_parquet(DATA)
    conc = df.pivot_table(index="timestamp", columns="depth", values="total_conc").sort_index()

    # 每日均值残差（去 30 天季节基线后再日平均，昼夜周期被平均掉）
    daily = {}
    for d in DEPTHS:
        daily[d] = detrend(conc[d]).resample("1D").mean().dropna()
    S = daily[0.5]

    res = {}
    res["method"] = "daily-mean detrended residual (30d median baseline); diel cycle removed by daily averaging"
    h_days = np.arange(-30, 31)
    res["daily_detr_deep_to_surface"] = {}
    for d in [3.0, 5.0, 8.0, 10.0]:
        res["daily_detr_deep_to_surface"][f"{d}m"] = scan_daily(daily[d], S, h_days)

    # 聚焦任务书窗口 h=+1..7 天（深层领先 1-7 天），同表对照
    res["lead_1to7_days"] = {}
    for d in [3.0, 5.0, 8.0, 10.0]:
        vals = {int(hd): fmt(daily[d].corr(S.shift(-hd))) for hd in range(1, 8)}
        res["lead_1to7_days"][f"{d}m"] = vals

    # 事件锚定：用 N 的正式 12 个事件
    nres = json.load(open(N_RES, encoding="utf-8"))
    events = nres["final_event_list"]
    p90 = {d: conc[d].quantile(0.90) for d in DEPTHS}
    out = []
    for ev in events:
        start = pd.Timestamp(ev["start"])
        end = pd.Timestamp(ev["end"])
        win = conc.loc[start - pd.Timedelta(days=21): end]
        rise = {}
        for d in [0.5, 3.0, 5.0, 8.0, 10.0]:
            over = win[d].index[win[d] >= p90[d]]
            over_before = over[over < start - pd.Timedelta(hours=6)]
            over_after = over[over >= start - pd.Timedelta(hours=6)]
            rise[d] = str(over_before[-1]) if len(over_before) else (str(over_after[0]) if len(over_after) else None)
        out.append({"event": str(start.date()), "start": str(start), "end": str(end), "rise_ts": rise})

    leads5 = leads8 = leads3 = 0
    delays3, delays5, delays8 = [], [], []
    for o in out:
        r0 = pd.Timestamp(o["rise_ts"][0.5]) if o["rise_ts"][0.5] else None
        t3 = pd.Timestamp(o["rise_ts"][3.0]) if o["rise_ts"][3.0] else None
        t5 = pd.Timestamp(o["rise_ts"][5.0]) if o["rise_ts"][5.0] else None
        t8 = pd.Timestamp(o["rise_ts"][8.0]) if o["rise_ts"][8.0] else None
        if r0 is not None and t3 is not None:
            dd = (r0 - t3) / pd.Timedelta(days=1)
            delays3.append(dd); leads3 += int(dd > 0.25)
        if r0 is not None and t5 is not None:
            dd = (r0 - t5) / pd.Timedelta(days=1)
            delays5.append(dd); leads5 += int(dd > 0.25)
        if r0 is not None and t8 is not None:
            dd = (r0 - t8) / pd.Timedelta(days=1)
            delays8.append(dd); leads8 += int(dd > 0.25)

    res["event_anchored_N_events"] = {
        "n_events": len(out),
        "n_events_with_surface_rise": int(sum(1 for o in out if o["rise_ts"][0.5])),
        "deep3m_leads_surface_mean_days": fmt(np.mean(delays3), 2) if delays3 else None,
        "deep5m_leads_surface_mean_days": fmt(np.mean(delays5), 2) if delays5 else None,
        "deep8m_leads_surface_mean_days": fmt(np.mean(delays8), 2) if delays8 else None,
        "deep3m_lead_frac": fmt(leads3 / len(delays3)) if delays3 else None,
        "deep5m_lead_frac": fmt(leads5 / len(delays5)) if delays5 else None,
        "deep8m_lead_frac": fmt(leads8 / len(delays8)) if delays8 else None,
        "events": out,
    }
    return res


if __name__ == "__main__":
    r = run()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results_extra.json", "w", encoding="utf-8") as f:
        json.dump(r, f, ensure_ascii=False, indent=2, default=str)
    print("saved results_extra.json")
