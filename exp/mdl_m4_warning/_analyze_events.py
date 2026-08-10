# -*- coding: utf-8 -*-
"""mdl-m4-warning 事件分析（临时）：日级 BloomLabeler 事件 vs N 探索 12 事件对齐 + 滚动测试窗覆盖。
只输出统计量/事件区间（不含原始数据行）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rams.data.tensor_builder import (  # noqa: E402
    BloomLabeler,
    DailyConfig,
    DailyTensorBuilder,
    make_rolling_anchors,
)

DATA = Path("data/processed/standard.parquet")
D0 = pd.Timestamp("2021-03-01")
TRAIN_DAYS, TEST_DAYS, STRIDE_DAYS, N_WINDOWS = 730, 90, 45, 17

# N 探索 12 事件（results.md §5.2）
N_EVENTS = [
    ("2021-03-27", "2021-07-07"), ("2021-08-06", "2021-08-08"), ("2021-08-09", "2021-08-20"),
    ("2021-10-01", "2021-10-12"), ("2022-05-18", "2022-05-24"), ("2022-06-01", "2022-06-03"),
    ("2023-04-02", "2023-05-03"), ("2023-05-04", "2023-05-15"), ("2023-10-17", "2023-10-22"),
    ("2024-08-16", "2024-08-23"), ("2024-08-25", "2024-08-27"), ("2025-09-27", "2025-09-30"),
]

builder = DailyTensorBuilder(DailyConfig())
daily = builder.load_daily_wide(DATA)
print(f"日级表: {len(daily)} 行, {daily.index.min().date()} → {daily.index.max().date()}")

# --- 全量拟合 BloomLabeler（N 定义，整集 p90）→ 事件 ---
lab = BloomLabeler(DailyConfig())
lab.fit(daily)
sig = lab.predict(daily)
evs = lab.events(sig, daily.index)
print(f"\nBloomLabeler 全量拟合: 藻华状态日 {int(sig.sum())}/{len(sig)} ({sig.mean():.3f})")
print(f"事件数: {len(evs)}")
for i, e in enumerate(evs, 1):
    print(f"  {i:2d}. {e['start']} → {e['end']}  {e['n_days']}天")

# --- 对齐 N 探索 12 事件：找 daily 事件中与每个 N 事件 start 相距最近者 ---
print("\n=== 对齐 N 探索 12 事件 ===")
n_starts = [pd.Timestamp(s) for s, _ in N_EVENTS]
n_ends = [pd.Timestamp(e) for _, e in N_EVENTS]
ev_starts = pd.DatetimeIndex([pd.Timestamp(e["start"]) for e in evs])
ev_ends = pd.DatetimeIndex([pd.Timestamp(e["end"]) for e in evs])

for i, (s, e_) in enumerate(N_EVENTS, 1):
    s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e_)
    # daily 事件中 start 最接近 N 事件 start
    d = np.abs((ev_starts - s_ts).days)
    j = int(np.argmin(d))
    overlap = min(ev_ends[j], e_ts) - max(ev_starts[j], s_ts)
    ov_days = max(0, overlap.days + 1)
    print(
        f"  N{i:2d} {s} → {e_}  | daily #{j+1}: {ev_starts[j].date()} → {ev_ends[j].date()} "
        f"start偏移{d[j]:>3}d 重叠{ov_days}天"
    )

# --- 滚动测试窗覆盖（事件 look-back Lmax=30 天是否全有 out-of-sample 预测） ---
anchors = make_rolling_anchors(D0, TRAIN_DAYS, TEST_DAYS, STRIDE_DAYS, N_WINDOWS)
test_periods = [(tr, end) for _, tr, end in anchors]  # 每窗口测试段 [tr, end)
print(f"\n=== 滚动测试窗（{len(test_periods)} 个）===")
for wi, (tr, end) in enumerate(test_periods, 1):
    print(f"  w{wi:2d}: {tr.date()} → {end.date()}")

# 每天是否被 ≥1 窗口的测试段覆盖
days = pd.date_range(daily.index.min(), daily.index.max(), freq="1D")
cover = np.zeros(len(days), dtype=bool)
for tr, end in test_periods:
    mask = (days >= tr) & (days < end)
    cover |= mask
print(f"\n测试段覆盖天数: {int(cover.sum())}/{len(days)} ({cover.mean():.2f})")
print(f"覆盖范围: {days[cover].min().date()} → {days[cover].max().date()}")

print("\n=== 每事件 look-back [s-30, s) 覆盖度 ===")
for i, (s, e_) in enumerate(N_EVENTS, 1):
    s_ts = pd.Timestamp(s)
    lb = (days >= s_ts - pd.Timedelta(days=30)) & (days < s_ts)
    frac = float(cover[lb].mean())
    flag = "可评估" if frac == 1.0 else (f"部分({frac:.2f})" if frac > 0 else "不可评估(全样本内)")
    print(f"  N{i:2d} {s}: look-back 覆盖 {frac:.2f} → {flag}")

# daily 事件里的不可评估事件（在 2023-03 之前的）
print("\n=== daily 事件是否可评估（按测试段覆盖）===")
for i, e in enumerate(evs, 1):
    s_ts = pd.Timestamp(e["start"])
    lb = (days >= s_ts - pd.Timedelta(days=30)) & (days < s_ts)
    frac = float(cover[lb].mean())
    print(f"  daily#{i:2d} {e['start']} → {e['end']}: 覆盖 {frac:.2f}")
