# -*- coding: utf-8 -*-
"""O 方向：多层信号之间的联系（20 层浓度 + 水温的同步性 / 相位滞后 / 变率谱）

核心待解问题：20 层信号之间到底是什么联系？
  - A 同步联动（深层冗余）：同期相关矩阵 —— 强相关(>0.8)说明深层只是冗余复制，
    弱相关说明深层携带独立信息
  - B 相位/滞后（垂向迁移）：深层浓度是否"领先"表层？若 3/5/8m 的 t 时刻浓度
    与表层 t+h 浓度互相关在 h>0 最大，说明深层是前导指标（蓝藻深水积聚→上浮→
    表层暴发），则反转"深层无用"结论
  - C 变率谱（尺度性）：各层能量集中在什么时间尺度（FFT 主导周期 / 自相关衰减），
    量化"表层快、深层慢"

分析口径（沿项目已证实结论）：
  - 原始水平值被年际季节主导（2021 单一藻华季），滞后相关必须去季节化才见真信号：
    主口径 = 减 30 天滚动中位数（季节基线）后的残差；副口径 = 24h 增量（项目预测目标）
  - 事件锚定（垂向迁移最直接检验）：N 定义的表层藻华事件，比较深层与表层各自
    "首次超该层 p90" 的先后

保密：数据涉密只输出统计量，不打印原始数据行。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/processed/standard.parquet")
OUT = Path("exp/model_enhancement/o_multilayer_link")

DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
          5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
PT_PER_DAY = 8  # 3h 采样


def fmt(x, nd=3):
    return round(float(x), nd)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(DATA)
    conc = df.pivot_table(index="timestamp", columns="depth", values="total_conc").sort_index()
    temp = df.pivot_table(index="timestamp", columns="depth", values="water_temp").sort_index()
    return conc, temp


def summary_corr_mat(corr: pd.DataFrame) -> dict:
    """相关矩阵摘要：表层 vs 各层、相邻层、跨层、分组均值。"""
    surf = corr.loc[0.5]
    depths = [float(d) for d in corr.index]
    diag = np.array(depths)
    res = {
        "surface_vs_layers": {d: fmt(surf[d]) for d in depths},
        "adjacent_mean": fmt(np.mean([corr.loc[d1, d2] for d1, d2 in zip(diag[:-1], diag[1:])])),
        "surface_vs_bottom": fmt(surf[10.0]),
        "surface_vs_8m": fmt(surf[8.0]),
        "surface_vs_5m": fmt(surf[5.0]),
    }
    # 组内平均相关
    groups = {"top_0_3m": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
              "mid_3_5m": [3.5, 4.0, 4.5, 5.0],
              "deep_5_7m": [5.5, 6.0, 6.5, 7.0, 7.5],
              "bottom_8_10m": [8.0, 8.5, 9.0, 9.5, 10.0]}
    for g, ds in groups.items():
        pairs = []
        for i, a in enumerate(ds):
            for b in ds[i + 1:]:
                pairs.append(corr.loc[a, b])
        res[f"intra_{g}_mean"] = fmt(np.mean(pairs))
    # 跨组（表层带 vs 深层带/底层带）平均相关
    top = groups["top_0_3m"]
    for g2 in ["mid_3_5m", "deep_5_7m", "bottom_8_10m"]:
        ds2 = groups[g2]
        pairs = [corr.loc[a, b] for a in top for b in ds2]
        res[f"cross_top_vs_{g2}_mean"] = fmt(np.mean(pairs))
    return res


def detrend(series: pd.Series, win_days: int = 30) -> pd.Series:
    """减滚动中位数去季节基线；窗内样本少时回退到全局中位数。"""
    win = win_days * PT_PER_DAY
    base = series.rolling(win, center=True, min_periods=PT_PER_DAY * 7).median()
    base = base.fillna(series.rolling(win, center=False, min_periods=PT_PER_DAY * 7).median())
    base = base.fillna(series.median())
    return series - base


def lag_corr(a: pd.Series, b: pd.Series, h_pts: int) -> float:
    """corr(a[t], b[t+h])；h>0 => a 领先 b。NaN 按可用对处理。"""
    return a.corr(b.shift(-h_pts))


def scan_lead(a: pd.Series, b: pd.Series, h_days: np.ndarray) -> dict:
    """扫描滞后，返回 max 对应 h、max 相关、滞后曲线。"""
    best = None
    curve = {}
    for hd in h_days:
        hp = int(round(hd * PT_PER_DAY))
        c = lag_corr(a, b, hp)
        curve[float(hd)] = fmt(c)
        if best is None or abs(c) > abs(best[1]):
            best = (float(hd), c)
    return {"argmax_h_days": best[0], "max_corr": fmt(best[1]), "curve": curve}


def fft_dominant_period(series: pd.Series) -> dict:
    """FFT 主导周期 + 频带能量占比（去均值/趋势后）。"""
    x = series.dropna()
    if len(x) < 512:
        return {}
    x = x.interpolate(method="linear", limit=48).dropna()
    x = x - x.mean()
    n = len(x)
    dt = 1 / PT_PER_DAY
    freqs = np.fft.rfftfreq(n, d=dt)
    spec = np.abs(np.fft.rfft(x)) ** 2
    spec[0] = 0  # 去 DC
    mask = (freqs > 1 / 730) & (freqs <= 1 / 2)  # 周期 2~730 天
    if mask.sum() == 0:
        return {}
    f_m = freqs[mask]
    s_m = spec[mask]
    total = s_m.sum()
    if total <= 0:
        return {}
    frac = s_m / total

    def frac_band(pmin, pmax):
        fm = (f_m >= 1 / pmax) & (f_m < 1 / pmin)
        return fmt(frac[fm].sum()) if fm.sum() else 0.0

    pk = f_m[np.argmax(s_m)]
    return {
        "dominant_period_days": fmt(1 / pk, 1),
        "frac_power_2_7d": frac_band(2, 7),
        "frac_power_7_30d": frac_band(7, 30),
        "frac_power_30_180d": frac_band(30, 180),
        "frac_power_180_730d": frac_band(180, 730),
    }


def acf_efold_lag(series: pd.Series, max_lag_days: int = 60) -> float:
    """自相关衰减到 1/e 的滞后（天）。"""
    x = series.interpolate(method="linear", limit=48).dropna()
    if len(x) < 256:
        return float("nan")
    x = x - x.mean()
    x = x / x.std()
    max_lag = min(int(max_lag_days * PT_PER_DAY), len(x) // 3)
    ac = [x.autocorr(lag) for lag in range(1, max_lag)]
    target = 1 / np.e
    for i, v in enumerate(ac):
        if v <= target:
            return round((i + 1) / PT_PER_DAY, 2)
    return round(max_lag / PT_PER_DAY, 2)


def _runs_of(mask: pd.Series) -> list[int]:
    runs, cur = [], 0
    for v in mask.values:
        if v:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs


def event_anchored_lead(conc: pd.DataFrame, events: list[dict]) -> dict:
    """每个表层藻华事件：深层首次超 p90 是否早于表层首次超 p90（垂向迁移直接检验）。

    对每个事件，在 [start-21d, start+peak 前] 窗口找各层"首次 ≥ 该层 p90"时刻，
    比较深层(5/8m)与表层(0.5m)的先后。
    """
    p90 = {d: conc[d].quantile(0.90) for d in DEPTHS}
    out = []
    for ev in events:
        start = ev["start"]
        end = ev["end"]
        win = conc.loc[start - pd.Timedelta(days=21): end]
        rise = {}
        for d in [0.5, 3.0, 5.0, 8.0, 10.0]:
            col = win[d]
            over = col.index[col >= p90[d]]
            over_before = over[over < start - pd.Timedelta(hours=6)]
            over_after = over[over >= start - pd.Timedelta(hours=6)]
            rise[d] = str(over_before[-1]) if len(over_before) else (str(over_after[0]) if len(over_after) else None)
        out.append({"event": str(start.date()), "rise_ts": rise})
    leads5 = leads8 = 0
    delays5, delays8 = [], []
    for o in out:
        r0, r5, r8 = (o["rise_ts"].get(d) for d in [0.5, 5.0, 8.0])
        ts0 = pd.Timestamp(r0) if r0 else None
        t5 = pd.Timestamp(r5) if r5 else None
        t8 = pd.Timestamp(r8) if r8 else None
        if ts0 is not None and t5 is not None:
            d = (ts0 - t5) / pd.Timedelta(days=1)  # >0 表示 5m 领先
            delays5.append(d)
            leads5 += int(d > 0.25)
        if ts0 is not None and t8 is not None:
            d = (ts0 - t8) / pd.Timedelta(days=1)
            delays8.append(d)
            leads8 += int(d > 0.25)
    return {
        "n_events": len(out),
        "deep5m_leads_surface_mean_days": fmt(np.mean(delays5), 2) if delays5 else None,
        "deep8m_leads_surface_mean_days": fmt(np.mean(delays8), 2) if delays8 else None,
        "deep5m_lead_frac": fmt(leads5 / len(delays5)) if delays5 else None,
        "deep8m_lead_frac": fmt(leads8 / len(delays8)) if delays8 else None,
        "events": out,
    }


def detect_events(conc: pd.DataFrame, band: pd.Series) -> list[dict]:
    """简化 N 事件判定：顶层带>带p90 且 0-5m ≥3 层超 p90，连续≥2 天，间隔≤24h 合并。"""
    band_p90 = band.quantile(0.90)
    exc = pd.DataFrame({d: (conc[d] > conc[d].quantile(0.90)).astype(float) for d in conc.columns[:11]})
    st = (band > band_p90) & (exc.sum(axis=1) >= 3)
    runs = _runs_of(st)
    idx = st.index
    evs = []
    start = prev_end = None
    acc = 0
    for r in runs:
        s = idx[acc]
        e = idx[acc + r - 1]
        acc += r
        if r >= 2 * PT_PER_DAY and (start is None or (s - prev_end) > pd.Timedelta(hours=24)):
            if start is not None:
                evs.append({"start": start, "end": prev_end})
            start, prev_end = s, e
        elif start is not None:
            prev_end = e
    if start is not None:
        evs.append({"start": start, "end": prev_end})
    return evs


def run() -> dict:
    conc, temp = load()
    res: dict = {}
    res["n_timestamps"] = int(len(conc))
    res["time_range"] = [str(conc.index.min()), str(conc.index.max())]

    # ================= A. 同期相关结构 =================
    corr_conc = conc.corr()
    corr_temp = temp.corr()
    res["A_conc_corr_summary"] = summary_corr_mat(corr_conc)
    res["A_temp_corr_summary"] = summary_corr_mat(corr_temp)
    res["A_conc_corr_matrix"] = {float(d1): {float(d2): fmt(corr_conc.loc[d1, d2]) for d2 in DEPTHS} for d1 in DEPTHS}
    res["A_temp_corr_matrix"] = {float(d1): {float(d2): fmt(corr_temp.loc[d1, d2]) for d2 in DEPTHS} for d1 in DEPTHS}
    # 浓度表层 vs 底层同期相关，按时段细分（暴发季 2021 vs 非暴发 2022-25）
    for label, sl in [("2021", slice("2021-01-01", "2021-12-31")), ("nonbloom_22_25", slice("2022-01-01", "2025-12-31"))]:
        sub = conc.loc[sl]
        res[f"A_conc_surf_bottom_corr_{label}"] = fmt(sub.corr().loc[0.5, 10.0])

    # ================= B. 相位/滞后 =================
    # 主口径：30 天去季节残差的滞后相关（深层 → 表层，h>0 = 深层领先）
    detr = {d: detrend(conc[d]) for d in DEPTHS}
    S_detr = detr[0.5]
    h_days = np.arange(-14, 14.01, 1)
    res["B_deep_to_surface_detrended"] = {}
    for d in [3.0, 5.0, 8.0, 10.0]:
        res["B_deep_to_surface_detrended"][f"{d}m"] = scan_lead(detr[d], S_detr, h_days)
    # 原始水平值对照（季节主导，作为参照）
    res["B_deep_to_surface_raw"] = {}
    for d in [3.0, 5.0, 8.0]:
        res["B_deep_to_surface_raw"][f"{d}m"] = scan_lead(conc[d], conc[0.5], h_days)
    # 24h 增量口径（项目预测目标）
    inc = {d: conc[d].diff(PT_PER_DAY) for d in DEPTHS}
    res["B_deep_to_surface_increment24h"] = {}
    for d in [3.0, 5.0, 8.0]:
        res["B_deep_to_surface_increment24h"][f"{d}m"] = scan_lead(inc[d], inc[0.5], h_days)
    # 细尺度（3h 步长 ±2 天），迁移发生在一到数天尺度时看亚天精度
    h_fine = np.arange(-2, 2.001, 0.125)
    res["B_deep_to_surface_detrended_fine"] = {}
    for d in [5.0, 8.0]:
        res["B_deep_to_surface_detrended_fine"][f"{d}m"] = scan_lead(detr[d], S_detr, h_fine)
    # 水温对照：深层水温 → 表层水温（热传导方向通常表层领先）
    temp_detr = {d: detrend(temp[d]) for d in DEPTHS}
    res["B_temp_deep_to_surface_detrended"] = {}
    for d in [5.0, 8.0, 10.0]:
        res["B_temp_deep_to_surface_detrended"][f"{d}m"] = scan_lead(temp_detr[d], temp_detr[0.5], h_days)

    # 事件锚定（垂向迁移直接检验）
    band = conc[[0.5, 1.0, 1.5, 2.0, 2.5, 3.0]].median(axis=1)
    evs = detect_events(conc, band)
    res["B_n_events"] = int(len(evs))
    res["B_event_anchored_lead"] = event_anchored_lead(conc, evs)

    # ================= C. 变率谱 =================
    spec = {}
    for d in DEPTHS:
        spec[float(d)] = {"fft": fft_dominant_period(conc[d]),
                          "acf_efold_lag_days": acf_efold_lag(conc[d]),
                          "inc24h_std": fmt(inc[d].std())}
    res["C_spectrum_by_layer"] = spec
    temp_spec = {}
    for d in DEPTHS:
        temp_spec[float(d)] = {"fft": fft_dominant_period(temp[d]),
                               "acf_efold_lag_days": acf_efold_lag(temp[d]),
                               "inc24h_std": fmt(temp[d].diff(PT_PER_DAY).std())}
    res["C_spectrum_temp_by_layer"] = temp_spec

    return res


if __name__ == "__main__":
    res = run()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=str)
    print("saved results.json; keys:", list(res.keys()))
