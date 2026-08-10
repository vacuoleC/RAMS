# -*- coding: utf-8 -*-
"""L 补充分析：各尺度逐视界持久化技能（数据-only，不需再训练）

results.json 存了模型逐视界 CRPS（3 seed × 17 窗口均值），但没存逐视界持久化 CRPS。
持久化（Δ≡0 → conc_{t+h}=conc_t）只需数据即可算，本脚本：
  1. 各尺度（3h/12h/1D）同窗口协议（日历日对齐）算测试段逐视界持久化 CRPS
  2. 与 results.json 模型逐视界 CRPS 合并 → 逐视界技能表
  3. 同 24h 提前量的逐窗口对照（3h h8 / 12h h2 / 1D h1）——用于检验 24h 提前量优势是否
     在全部 17 窗口一致（而非少数窗口拉高均值）

保密：只输出聚合统计量，不打印原始数据行。
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from rams.data.tensor_builder import TensorBuilder, TensorConfig  # noqa: E402

TRAIN_DAYS = 730
TEST_DAYS = 90
STRIDE_DAYS = 45
N_WINDOWS = 17

SCALES = {
    "3h":   {"step_h": 3,  "T": 24, "H": 8},
    "12h":  {"step_h": 12, "T": 24, "H": 4},
    "1D":   {"step_h": 24, "T": 30, "H": 7},
}


def crps_quantiles(q10, q50, q90, y):
    q10 = np.asarray(q10, dtype=np.float64)
    q50 = np.asarray(q50, dtype=np.float64)
    q90 = np.asarray(q90, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    qs = np.sort(np.stack([q10, q50, q90], axis=-1), axis=-1)
    q10, q50, q90 = qs[..., 0], qs[..., 1], qs[..., 2]
    qk = np.stack([
        q10 - (q50 - q10) / 4.0,
        q10, q50, q90,
        q90 + (q90 - q50) / 4.0,
    ], axis=-1)
    ak = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
    deg = (q90 - q10) < 1e-9
    total = np.zeros_like(y, dtype=np.float64)
    for k in range(4):
        aL, aR = ak[k], ak[k + 1]
        qL, qR = qk[..., k], qk[..., k + 1]
        slope = (qR - qL) / (aR - aL)
        p1 = np.where(np.abs(slope) < 1e-12, 1.0, slope)
        p0 = qL - p1 * aL
        with np.errstate(all="ignore"):
            astar = (y - p0) / p1
            c = np.clip(astar, aL, aR)
        for u, v in ((aL, c), (c, aR)):
            mid = (u + v) / 2.0
            s = (y <= (p0 + p1 * mid)).astype(np.float64)
            C0 = s * (p0 - y)
            C1 = s * p1 - p0 + y
            total += 2.0 * (C0 * (v - u) + C1 * (v * v - u * u) / 2.0
                            - p1 * (v * v * v - u * u * u) / 3.0)
    out = np.where(deg, np.abs(y - q50), total)
    return np.maximum(out, 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--results", default="exp/model_enhancement/l_daily_scale/results.json")
    ap.add_argument("--out-json", default="exp/model_enhancement/l_daily_scale/skill_per_h.json")
    args = ap.parse_args()

    # 模型逐视界 CRPS（3 seed × 17 窗口均值）
    res = json.load(open(args.results, encoding="utf-8"))
    model_crps_h = {s: np.array(res["scales"][s]["crps_h"]) for s in SCALES}

    loader = TensorBuilder(TensorConfig())
    wide_raw = loader._load_wide(Path(args.parquet)).sort_index()
    d0 = wide_raw.index.min()

    # 每尺度宽表（含持久化需要的 conc 原始列）
    wides = {"3h": wide_raw}
    for s in ("12h", "1D"):
        c = SCALES[s]
        wides[s] = wide_raw.resample(f"{c['step_h']}h").mean().dropna()

    out = {s: {"persist_crps_h": np.zeros(SCALES[s]["H"]),
               "skill_h": np.zeros(SCALES[s]["H"]),
               "n_test_sum": 0} for s in SCALES}
    # 同 24h 提前量逐窗口持久化/模型：3h h8 / 12h h2 / 1D h1
    lead24 = {"3h": 8, "12h": 2, "1D": 1}
    at24 = {s: {"model_crps": [], "persist_crps": []} for s in SCALES}

    for wi in range(N_WINDOWS):
        start_ts = d0 + pd.Timedelta(days=STRIDE_DAYS * wi)
        tr_ts = start_ts + pd.Timedelta(days=TRAIN_DAYS)
        end_ts = start_ts + pd.Timedelta(days=TRAIN_DAYS + TEST_DAYS)

        for s in SCALES:
            c = SCALES[s]
            T, H = c["T"], c["H"]
            w = wides[s]
            df = w[(w.index >= start_ts) & (w.index < end_ts)]
            n_tr_rows = int((df.index < tr_ts).sum())
            y_raw = df["conc_0.5"].values.astype(np.float64)
            n_w = len(df) - T - H
            n_tr = max(0, n_tr_rows - T - H + 1)
            y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)])
            cur_raw = np.array([y_raw[i + T - 1] for i in range(n_w)])
            te = slice(n_tr, n_w)
            y_te = y_abs[te]
            cur_te = cur_raw[te]

            # 持久化 CRPS：预测 conc_{t+h}=conc_t（三点都退化为 conc_t → 退化分布 CRPS=|y-conc_t|）
            persist_h = [float(np.mean(np.abs(y_te[:, h] - cur_te)))
                         for h in range(H)]
            out[s]["persist_crps_h"] += np.array(persist_h)
            out[s]["n_test_sum"] += len(y_te)

            h24 = lead24[s] - 1
            at24[s]["persist_crps"].append(persist_h[h24])

    # 逐窗口模型 at24：从结果 json 没有逐窗口逐视界…… 只能给整体 at24 的模型值
    # （crps_at_24h 已在 results.json 存整体均值）。这里补逐窗口持久化 at24 的均值。
    print("===== 逐视界持久化 CRPS 与技能（3 seed × 17 窗口均值）=====", flush=True)
    for s in SCALES:
        c = SCALES[s]
        pers = out[s]["persist_crps_h"] / N_WINDOWS
        skill = (pers - model_crps_h[s]) / pers * 100.0
        out[s]["persist_crps_h"] = pers.tolist()
        out[s]["skill_h"] = skill.tolist()
        out[s]["skill_mean"] = float(skill.mean())
        print(f"\n[{s}] H={c['H']} 视界={c['H'] * c['step_h']}h  平均技能 {skill.mean():+.1f}%",
              flush=True)
        print(f"    h  时长  模型CRPS  持久化CRPS  技能%", flush=True)
        for h in range(c["H"]):
            print(f"    h{h + 1:<3}{c['step_h'] * (h + 1):>4}h  "
                  f"{model_crps_h[s][h]:>9.4f}  {pers[h]:>9.4f}  {skill[h]:>+6.1f}",
                  flush=True)

    # 同 24h 提前量：逐窗口持久化 at24 稳定性
    print("\n===== 同 24h 提前量：逐窗口持久化 CRPS at24h =====", flush=True)
    print("  w   3h_persist  12h_persist   1D_persist   ", flush=True)
    for wi in range(N_WINDOWS):
        row = f"{wi + 1:>4}"
        for s in SCALES:
            row += f"{at24[s]['persist_crps'][wi]:>12.4f}"
        print(row, flush=True)
    print("  (模型 at24 均值见 results.json：3h h8 = 1.0111 / 12h h2 = 0.7853 / 1D h1 = 0.6325)",
          flush=True)
    # 逐窗口 at24 持久化技能（模型用整体均值近似，只做窗口一致性检查）
    print("\n===== 同 24h 提前量：逐窗口持久化技能%（模型 at24 用整体均值）=====", flush=True)
    model_at24 = {"3h": res["scales"]["3h"]["crps_at_24h"],
                  "12h": res["scales"]["12h"]["crps_at_24h"],
                  "1D": res["scales"]["1D"]["crps_at_24h"]}
    print("  w   3h_skill   12h_skill    1D_skill   ", flush=True)
    for wi in range(N_WINDOWS):
        row = f"{wi + 1:>4}"
        for s in SCALES:
            p = at24[s]["persist_crps"][wi]
            row += f"{(p - model_at24[s]) / p * 100:>12.1f}"
        print(row, flush=True)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    save = {}
    for s in SCALES:
        save[s] = {"persist_crps_h": out[s]["persist_crps_h"],
                   "model_crps_h": model_crps_h[s].tolist(),
                   "skill_h": out[s]["skill_h"],
                   "skill_mean": out[s]["skill_mean"]}
    Path(args.out_json).write_text(json.dumps(save, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[result] 已写入 {args.out_json}")


if __name__ == "__main__":
    main()
