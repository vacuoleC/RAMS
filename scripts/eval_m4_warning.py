# -*- coding: utf-8 -*-
"""mdl-m4-warning 正式评估：M4 藻华预警 召回率 + 提前量 + 误报（H100 / CPU 冒烟）

目标（冻结设计 `modules/mdl-m4-warning/module_design.yaml`）：
  用 N 定义藻华事件评估 M4 预警头：
    - 召回率：模型预警正确覆盖了多少个藻华事件
    - 提前量：预警发出到事件发生的提前天数
    - 误报率：预警但未发生事件的次数
    - 与 N 探索 12 事件对齐

协议（对齐 train_v020 / T4 冻结口径）：
  - 日级 `DailyTensorBuilder`（T=30 回看 / H=7 视界 / 1D 均值聚合）
  - 滚动窗口：730d 训练 / 90d 测试 / 45d 步长 / 17 窗口（d0=2021-03-01）
    → 出样本评估期 [2023-03-01, 2025-05-18]（N 事件 #7-#11 落在期内）
  - 模型：`RamsNet`（GRU hidden=64 + M1 q9 + M2 分层 + M4 bloom 二分类）
  - 训练：`Trainer.fit_two_stage`（Stage1 单任务 M1 20ep → Stage2 冻结 backbone 多任务 10ep）
  - 3-seed × 17 窗口；测试段逐日 M4 P(bloom) 跨窗口/seed 均值 → 每日预警概率序列
  - 预警 = P(bloom) > θ（阈值扫描）；事件 = `BloomLabeler`（N 定义，日级）+ N 探索 12 事件
  - 指标：召回率 / 提前量（预警首日 → 事件 start 天数）/ 误报（未引致事件的预警段）

数据保密红线：只输出统计量 / 事件区间（日期），不打印任何原始数据行。

用法：
  python3 scripts/eval_m4_warning.py --smoke                      # 本地 CPU 冒烟（1窗口×1seed，fast_dev_run）
  python3 scripts/eval_m4_warning.py --windows 17 --seeds 3       # 算力机全量（默认）
  python3 scripts/eval_m4_warning.py --out exp/mdl_m4_warning/results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# UTF-8 控制台输出（Windows）：仅在作为脚本运行时生效；
# 被 pytest 等导入时不劫持 sys.stdout（否则会包住/关闭 capture 缓冲）。
if __name__ == "__main__" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import torch  # noqa: E402

from rams.data.tensor_builder import (  # noqa: E402
    BLOOM_MIN_DAYS,
    BloomLabeler,
    DailyConfig,
    DailyTensorBuilder,
    make_rolling_anchors,
)
from rams.models.rams_net import RamsNet  # noqa: E402
from rams.training.trainer import Trainer, make_m4_labels  # noqa: E402

T, H = 30, 7
EP1, EP2 = 20, 10
D0 = pd.Timestamp("2021-03-01")
TRAIN_DAYS, TEST_DAYS, STRIDE_DAYS, N_WINDOWS = 730, 90, 45, 17
BATCH, HIDDEN = 128, 64
M4_MODE, M4_LEVELS = "bloom", 2

# 预警提前量窗口（回看天数）：N 探索事件爬升提前量 12-27 天（中位 19.5）→ 用 30 天覆盖
LMAX = 30
# 阈值扫描（预警 = P(bloom) > θ）
THRESHOLDS = [0.3, 0.5, 0.6, 0.7, 0.8]

# N 探索 12 个藻华事件（`exp/model_enhancement/n_bloom_identify/results.md` §5.2，冻结）
N_EVENTS = [
    ("2021-03-27", "2021-07-07"), ("2021-08-06", "2021-08-08"), ("2021-08-09", "2021-08-20"),
    ("2021-10-01", "2021-10-12"), ("2022-05-18", "2022-05-24"), ("2022-06-01", "2022-06-03"),
    ("2023-04-02", "2023-05-03"), ("2023-05-04", "2023-05-15"), ("2023-10-17", "2023-10-22"),
    ("2024-08-16", "2024-08-23"), ("2024-08-25", "2024-08-27"), ("2025-09-27", "2025-09-30"),
]


# =====================================================================
# 事件对齐 / 评估
# =====================================================================
def events_span(events: list[tuple[str, str]], d: pd.Timestamp) -> bool:
    """判断日期 d 是否落在任一事件区间 [start, end] 内。"""
    for s, e in events:
        if pd.Timestamp(s) <= d <= pd.Timestamp(e):
            return True
    return False


def warning_episodes(mask: np.ndarray) -> list[tuple[int, int]]:
    """从逐日布尔序列提取连续预警段（[start_idx, end_idx) 半开）。"""
    m = np.asarray(mask, dtype=bool)
    eps: list[tuple[int, int]] = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]:
                j += 1
            eps.append((i, j))
            i = j
        else:
            i += 1
    return eps


def evaluate_threshold(
    prob: np.ndarray,
    dates: pd.DatetimeIndex,
    ev_span: list[tuple[pd.Timestamp, pd.Timestamp]],
    theta: float,
    Lmax: int = LMAX,
) -> dict:
    """单阈值评估：召回 / 提前量 / 误报。

    命中：事件 start 前 Lmax 天窗口 [s-Lmax, s) 内存在预警日。
    提前量：命中的最后一次预警段（含最靠近 start 的预警日）的首日 → 事件 start 天数。
    误报：评估期内不与任何事件关联的预警段（段末落在事件期内视为事件内，不计误报）。
    """
    warn = prob > theta
    n = len(dates)
    # 评估期事件
    events_in_period = [
        (s, e) for s, e in ev_span if dates[0] <= s <= dates[-1]
    ]
    ev_starts = sorted(s for s, _ in events_in_period)

    # 逐事件命中 + 提前量
    per_event = []
    n_hit = 0
    for s, e in events_in_period:
        s_ts = s
        i_s = int(np.searchsorted(dates.values, np.datetime64(s_ts), side="left"))
        lo = max(0, i_s - Lmax)
        win = warn[lo:i_s]
        if not win.any():
            per_event.append({"event_start": str(s.date()), "event_end": str(e.date()),
                              "hit": False, "lead_days": None})
            continue
        # 最后一次预警日所在段的首日 → 提前量
        last_warn = int(np.nonzero(win)[0][-1]) + lo
        # 段首回退
        seg_start = last_warn
        while seg_start > 0 and warn[seg_start - 1]:
            seg_start -= 1
        lead = int((s_ts - dates[seg_start]).days)
        n_hit += 1
        per_event.append({"event_start": str(s.date()), "event_end": str(e.date()),
                          "hit": True, "lead_days": lead})

    # 误报：评估期预警段，段末不在事件期内，且其后 Lmax 天内无事件 start
    eps = warning_episodes(warn)
    fp = 0
    fp_days = 0
    fp_episodes: list[dict] = []
    n_ep = 0
    for a, b in eps:
        if b <= 0 or a >= n:
            continue
        d_end = dates[b - 1]
        # 只统计评估期内的预警段
        if not (dates[0] <= d_end <= dates[-1]):
            continue
        n_ep += 1
        if events_span(events_in_period, d_end):
            continue  # 段末落在事件期内 → 事件内预警，不计数
        # 其后 Lmax 天内是否有事件 start（严格晚于段末）
        future = [s for s in ev_starts if s > d_end and (s - d_end).days <= Lmax]
        if future:
            continue  # 引致事件
        fp += 1
        fp_days += int(b - a)
        fp_episodes.append({
            "start": str(dates[a].date()), "end": str(dates[b - 1].date()),
            "days": int(b - a),
        })

    recall = n_hit / len(events_in_period) if events_in_period else float("nan")
    leads = [pe["lead_days"] for pe in per_event if pe["hit"]]
    total_days = max(1, n)
    return {
        "theta": theta,
        "n_events": len(events_in_period),
        "n_hit": n_hit,
        "recall": float(recall),
        "lead_median": float(np.median(leads)) if leads else None,
        "lead_mean": float(np.mean(leads)) if leads else None,
        "lead_min": float(np.min(leads)) if leads else None,
        "lead_max": float(np.max(leads)) if leads else None,
        "per_event": per_event,
        "n_warning_episodes": n_ep,
        "false_positive_episodes": fp,
        "false_positive_days": int(fp_days),
        "fpr_days": float(fp_days / total_days),
        "fp_episodes": fp_episodes,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="mdl-m4-warning：M4 藻华预警召回/提前量/误报评估")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-window", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="冒烟：1 窗口 × 1 seed，fast_dev_run（CPU 可跑）")
    ap.add_argument("--Lmax", type=int, default=LMAX)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="exp/mdl_m4_warning/results.json")
    args = ap.parse_args()

    cfg = DailyConfig(T=T, H=H, delta_target=True)
    anchors = make_rolling_anchors(
        D0, TRAIN_DAYS, TEST_DAYS, STRIDE_DAYS, args.start_window + args.windows
    )[args.start_window : args.start_window + args.windows]
    seeds = [0] if args.smoke else list(range(args.seeds))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    n_win = len(anchors)
    print(
        f"mdl-m4-warning: {len(seeds)} seed × {n_win} 窗口 | device={device} | "
        f"Lmax={args.Lmax} | smoke={args.smoke} | M4={M4_MODE}",
        flush=True,
    )

    # ---- 全量日级宽表（整集事件标签 + 基线 + 合并序列） ----
    builder = DailyTensorBuilder(cfg)
    daily = builder.load_daily_wide(args.parquet)
    daily = daily.sort_index()
    dates_all = daily.index
    print(f"日级表 {len(daily)} 行 {dates_all.min().date()} → {dates_all.max().date()}", flush=True)

    # 整集 BloomLabeler（N 定义）→ 日级事件 + 逐日信号
    lab = BloomLabeler(DailyConfig())
    lab.fit(daily)
    bloom_full = lab.predict(daily)
    daily_events = lab.events(bloom_full, dates_all)
    print(f"BloomLabeler 日级事件（N 定义，整集拟合）: {len(daily_events)} 个", flush=True)
    for i, e in enumerate(daily_events, 1):
        print(f"  #{i}: {e['start'][:10]} → {e['end'][:10]}  {e['n_days']}天", flush=True)

    top_band = daily[[f"conc_{d}" for d in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0) if f"conc_{d}" in daily.columns]].median(axis=1)

    # ---- 逐窗口训练 + 测试段 M4 概率 ----
    prob_sum = np.zeros(len(daily), dtype=np.float64)
    prob_cnt = np.zeros(len(daily), dtype=np.int64)
    warn_acc_all: list[float] = []
    baseline_warn = np.zeros(len(daily), dtype=bool)
    m4_probs: dict[str, np.ndarray] = {}

    t_start = time.time()
    for wi, (start, tr, end) in enumerate(anchors):
        t_w = time.time()
        ds = DailyTensorBuilder(cfg).build(args.parquet, start_ts=start, tr_ts=tr, end_ts=end)
        delta_scale = cfg.delta_scale
        X_flat, y_abs, y_delta, cur = ds.X_flat, ds.y_abs, ds.y_delta, ds.cur
        bloom, strat = ds.bloom, ds.strat
        n_tr = ds.n_train
        y_norm = (y_delta / delta_scale).astype(np.float32)
        warn = make_m4_labels(y_abs, n_tr, mode=M4_MODE, bloom=bloom)

        # 基线（防泄漏）：训练段顶层带 p75 → 测试段超阈值即预警（N 探索"持续爬升"启发式）
        tr_rows = daily[daily.index < tr]
        band_tr = top_band[top_band.index < tr]
        if len(band_tr) > 0:
            p75 = float(band_tr.quantile(0.75))
            te_mask = np.asarray((top_band.index >= tr) & (top_band.index < end))
            baseline_warn[te_mask] |= (top_band[te_mask] > p75).values

        probs_win = np.zeros(len(ds.X), dtype=np.float64)
        acc_win = []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            model = RamsNet(
                feat_dim=X_flat.shape[2], horizon=H, hidden=HIDDEN,
                use_m4=True, n_quantiles=9, n_levels=M4_LEVELS,
            )
            trn = Trainer(model, device=device)
            trn.fit_two_stage(
                X_flat[:n_tr], y_norm[:n_tr], strat[:n_tr], warn[:n_tr],
                ep1=2 if args.smoke else EP1, ep2=2 if args.smoke else EP2,
                batch_size=BATCH, freeze_backbone=True, fast_dev_run=args.smoke,
            )
            te = slice(n_tr, len(X_flat))
            model.eval()
            with torch.no_grad():
                _, _, m4 = model(torch.tensor(X_flat[te]).to(device))
            p_bloom = torch.softmax(m4, dim=1)[:, 1].cpu().numpy().astype(np.float64)
            full = np.zeros(len(ds.X), dtype=np.float64)
            full[te] = p_bloom
            probs_win += full / len(seeds)
            acc = float((m4.argmax(1).cpu().numpy() == np.asarray(bloom[te])).mean())
            acc_win.append(acc)
            if args.smoke:
                break
        warn_acc_all.append(float(np.mean(acc_win)))

        # 合并到日级（只取测试段 [tr, end) 的预测日）
        for i in range(len(ds.X)):
            d = ds.dates[i]
            if tr <= d < end:
                pos = int(np.searchsorted(dates_all.values, np.datetime64(d), side="left"))
                prob_sum[pos] += probs_win[i]
                prob_cnt[pos] += 1
        m4_probs[str(wi)] = probs_win
        print(
            f"[窗口 {wi+1}/{n_win}] {start.date()}→{tr.date()}→{end.date()}  n={len(X_flat)} "
            f"n_tr={n_tr} M4acc={float(np.mean(acc_win)):.3f} （{time.time()-t_w:.0f}s）",
            flush=True,
        )

    prob_mean = np.where(prob_cnt > 0, prob_sum / np.maximum(prob_cnt, 1), np.nan)

    # 评估期 = 测试段并集 [min_tr, max_end)
    tr_vals = [a[1] for a in anchors]
    end_vals = [a[2] for a in anchors]
    ev0, ev1 = min(tr_vals), max(end_vals)
    eval_mask = (dates_all >= ev0) & (dates_all < ev1)
    ev_dates = dates_all[eval_mask]
    ev_prob = prob_mean[eval_mask]
    ev_bloom = bloom_full[eval_mask]
    ev_baseline = baseline_warn[eval_mask]
    print(
        f"\n评估期 {ev0.date()} → {ev1.date()}：{int(eval_mask.sum())} 天，"
        f"概率覆盖 {int(np.isfinite(ev_prob).sum())} 天",
        flush=True,
    )

    # ---- 事件集 ----
    # 1) 评估期内的 N 探索 12 事件
    n_ev_in = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in N_EVENTS
               if ev0 <= pd.Timestamp(s) <= ev1]
    # 2) 评估期内的日级 BloomLabeler 事件
    de_ev_in = [(pd.Timestamp(e["start"]), pd.Timestamp(e["end"])) for e in daily_events
                if ev0 <= pd.Timestamp(e["start"]) <= ev1]

    # ---- 阈值扫描 ----
    print("\n================ 阈值扫描（N 探索 12 事件，评估期内） ================", flush=True)
    print(f"评估期内 N 事件: {len(n_ev_in)} 个 → {[(s, e) for s, e in n_ev_in]}", flush=True)
    sweep = []
    hdr = f"{'θ':>5} {'召回':>6} {'命中':>6} {'提前量中位':>9} {'提前量范围':>12} {'预警段':>6} {'误报段':>6}"
    print(hdr, flush=True)
    for th in THRESHOLDS:
        r = evaluate_threshold(ev_prob, ev_dates, n_ev_in, th, args.Lmax)
        sweep.append(r)
        lr = f"{r['lead_median']:>9.1f}" if r["lead_median"] is not None else f"{'—':>9}"
        lrng = (f"{r['lead_min']:.0f}-{r['lead_max']:.0f}" if r["lead_min"] is not None else "—")
        print(
            f"{th:>5.2f} {r['recall']:>6.3f} {r['n_hit']:>6} {lr} "
            f"{lrng:>12} {r['n_warning_episodes']:>6} {r['false_positive_episodes']:>6}",
            flush=True,
        )
        if r["fp_episodes"]:
            fps = ", ".join(f"{e['start']}~{e['end']}({e['days']}d)" for e in r["fp_episodes"])
            print(f"       误报段: {fps}", flush=True)

    # ---- 日级 BloomLabeler 事件召回（次级） ----
    print("\n================ 日级 BloomLabeler 事件召回（次级） ================", flush=True)
    de_sweep = []
    for th in THRESHOLDS:
        r = evaluate_threshold(ev_prob, ev_dates, de_ev_in, th, args.Lmax)
        de_sweep.append(r)
        print(
            f"  θ={th:.2f}: 召回 {r['recall']:.3f}（{r['n_hit']}/{r['n_events']}） "
            f"提前量中位 {r['lead_median'] if r['lead_median'] is not None else '—'} 天",
            flush=True,
        )

    # ---- 逐事件明细（12 事件全表对齐） ----
    print("\n================ 与 12 事件对齐明细（θ=0.5 参照） ================", flush=True)
    row = []
    for i, (s, e) in enumerate(N_EVENTS, 1):
        s_ts, e_ts = pd.Timestamp(s), pd.Timestamp(e)
        in_period = ev0 <= s_ts <= ev1
        # 日级信号：事件前后 3 天的藻华状态日
        near = (dates_all >= s_ts - pd.Timedelta(days=3)) & (dates_all <= e_ts + pd.Timedelta(days=3))
        bloom_days_near = int(bloom_full[near].sum())
        # 模型预警概率：lead 窗口内最大
        i_s = int(np.searchsorted(dates_all.values, np.datetime64(s_ts), side="left"))
        lo = max(0, i_s - args.Lmax)
        max_p = float(np.nanmax(prob_mean[lo:i_s])) if in_period and np.isfinite(prob_mean[lo:i_s]).any() else None
        hit05 = "—"
        lead05 = None
        if in_period:
            r05 = sweep[THRESHOLDS.index(0.5)]
            for pe in r05["per_event"]:
                if pd.Timestamp(pe["event_start"]) == s_ts:
                    hit05 = "✓" if pe["hit"] else "✗"
                    lead05 = pe["lead_days"]
        row.append({
            "i": i, "start": s, "end": e, "in_eval_period": in_period,
            "bloom_days_near": bloom_days_near, "max_p_lead": max_p,
            "hit_0.5": hit05, "lead_0.5": lead05,
        })
        max_p_str = f"{max_p:.3f}" if max_p is not None else "—"
        lead_str = f"{lead05}天" if lead05 is not None else "—"
        print(
            f"  N{i:2d} {s} → {e} | 期内={'是' if in_period else '否'} "
            f"日级藻华日={bloom_days_near:>2} | lead窗最大P={max_p_str:>8} "
            f"| θ=0.5命中={hit05} 提前={lead_str}",
            flush=True,
        )

    # ---- 基线对照：训练段顶层带 p75 持久化启发式（θ=0.5 等价，N 探索"爬升"直觉） ----
    print("\n================ 基线对照（顶层带 p75 阈值，评估期内） ================", flush=True)
    ev_bloom_bool = ev_baseline.astype(float)
    r_base = evaluate_threshold(ev_bloom_bool, ev_dates, n_ev_in, 0.5, args.Lmax)
    print(
        f"  θ=0.5: 召回 {r_base['recall']:.3f}（{r_base['n_hit']}/{r_base['n_events']}） "
        f"提前量中位 {r_base['lead_median'] if r_base['lead_median'] is not None else '—'} 天 "
        f"| 预警段 {r_base['n_warning_episodes']} 误报段 {r_base['false_positive_episodes']}",
        flush=True,
    )
    baseline_sweep = [r_base]

    # ---- 汇总 ----
    overall = {
        "meta": {
            "protocol": "rolling 730/90/45/17", "eval_period": [str(ev0.date()), str(ev1.date())],
            "T": T, "H": H, "ep1": EP1, "ep2": EP2, "batch": BATCH, "hidden": HIDDEN,
            "n_windows": n_win, "seeds": seeds, "m4_mode": M4_MODE, "m4_levels": M4_LEVELS,
            "Lmax": args.Lmax, "device": device, "smoke": args.smoke,
        },
        "daily_events": daily_events,
        "n_events_in_period": len(n_ev_in),
        "n_events_in_period_list": [{"start": str(s.date()), "end": str(e.date())} for s, e in n_ev_in],
        "m4_acc_window_mean": float(np.mean(warn_acc_all)) if warn_acc_all else None,
        "coverage_days": int(np.isfinite(ev_prob).sum()),
        "sweep_n_events": sweep,
        "sweep_daily_events": de_sweep,
        "baseline_p75": baseline_sweep,
        "event_alignment": row,
        "wall_seconds": round(time.time() - t_start, 1),
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(overall, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n结果已写: {out}", flush=True)
    print(f"总耗时 {time.time()-t_start:.0f}s（未打印任何原始数据行）", flush=True)


if __name__ == "__main__":
    main()
