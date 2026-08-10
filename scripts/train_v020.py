"""st-train-v020 正式训练验收：3-seed × 17 窗口，日级两阶段 ts_freeze（H100）。

协议（冻结设计 modules/mdl-model-integrate + T4 评估）：
  - 数据：DailyTensorBuilder（T=30 回看 / H=7 视界，1D 均值聚合，目标 Δ=conc_{t+h}-conc_t）
  - 滚动窗口：训练 730d / 测试 90d / 步长 45d / 17 窗口（日历日对齐，d0=2021-03-01）
  - 模型：RamsNet（GRU hidden=64 + M1 q9 分位 + M2 分层 + M4 藻华预警）
  - 训练：Trainer.fit_two_stage —— Stage1 单任务 M1（20ep）→ Stage2 冻结 backbone 微调多头（10ep，ts_freeze）
  - M4 标签：mode="bloom"（N 定义藻华状态，冻结设计口径）→ M4 头 n_levels=2
  - 评估：逐窗口 CRPS（q9 分段线性闭合）/ 相对持久化技能 / 覆盖率[p10,p90] / p50 RMSE / M2·M4 acc

数据保密红线：只输出聚合统计量 / 形状，不打印任何原始数据行。

用法：
  python3 scripts/train_v020.py --windows 1 --seeds 1 --fast-dev      # 冒烟（链路验证）
  python3 scripts/train_v020.py --out exp/model_enhancement/st-train-v020/results.json   # 全量
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

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from rams.data.tensor_builder import (  # noqa: E402
    DailyConfig,
    DailyTensorBuilder,
    make_rolling_anchors,
)
from rams.models.rams_net import QUANTILE_LEVELS, RamsNet  # noqa: E402
from rams.training.trainer import Trainer, crps_cdf_pline, make_m4_labels  # noqa: E402

T, H = 30, 7
EP1, EP2 = 20, 10
D0 = pd.Timestamp("2021-03-01")
TRAIN_DAYS, TEST_DAYS, STRIDE_DAYS, N_WINDOWS = 730, 90, 45, 17
BATCH, HIDDEN, N_Q = 128, 64, 9
M4_MODE, M4_LEVELS = "bloom", 2


def eval_window(
    model: RamsNet,
    X_te: np.ndarray,
    y_abs_te: np.ndarray,
    cur_te: np.ndarray,
    strat_te: np.ndarray,
    warn_te: np.ndarray | None,
    delta_scale: float,
    device: str,
) -> dict:
    """测试段评估（还原 conc 单位）。返回 CRPS/技能/覆盖率/RMSE/M2·M4 acc。"""
    model.eval()
    with torch.no_grad():
        m1, m2, m4 = model(torch.tensor(X_te).to(device))
    q = model.quantile_matrix(m1).cpu().numpy().astype(np.float64)  # (N, n_q, H) 归一化 Δ
    q_conc = cur_te[:, None, None] + q * delta_scale  # (N, n_q, H) conc 单位
    obs = np.asarray(y_abs_te)
    i10 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.10))[0][0])
    i50 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.50))[0][0])
    i90 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.90))[0][0])
    cover = float(np.mean((obs >= q_conc[:, i10]) & (obs <= q_conc[:, i90])))
    rmse = float(np.sqrt(np.mean((q_conc[:, i50] - obs) ** 2)))
    crps_h = np.array(
        [np.mean(crps_cdf_pline(q_conc[:, :, h], QUANTILE_LEVELS, obs[:, h])) for h in range(H)]
    )
    # 持久化：Δ≡0 → conc_{t+h}=conc_t（全分位同值）
    q_p = np.repeat(np.asarray(cur_te)[:, None, None], H, axis=2)  # (N,1,H)
    q_p = np.repeat(q_p, N_Q, axis=1)  # (N,n_q,H)
    crps_p_h = np.array(
        [np.mean(crps_cdf_pline(q_p[:, :, h], QUANTILE_LEVELS, obs[:, h])) for h in range(H)]
    )
    m2_acc = float((m2.argmax(1).cpu().numpy() == np.asarray(strat_te)).mean())
    m4_acc = None
    if m4 is not None and warn_te is not None:
        m4_acc = float((m4.argmax(1).cpu().numpy() == np.asarray(warn_te)).mean())
    return {
        "crps": float(crps_h.mean()),
        "crps_h": crps_h.tolist(),
        "crps_p": float(crps_p_h.mean()),
        "crps_p_h": crps_p_h.tolist(),
        "skill": float((crps_p_h.mean() - crps_h.mean()) / crps_p_h.mean() * 100),
        "rmse": rmse,
        "coverage": cover,
        "m2_acc": m2_acc,
        "m4_acc": m4_acc,
        "n_test": int(len(X_te)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="st-train-v020 正式训练验收（3-seed × 17 窗口）")
    ap.add_argument("--parquet", default="data/processed/standard.parquet")
    ap.add_argument("--windows", type=int, default=N_WINDOWS)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--start-window", type=int, default=0)
    ap.add_argument("--fast-dev", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = DailyConfig(T=T, H=H, delta_target=True)
    anchors = make_rolling_anchors(
        D0, TRAIN_DAYS, TEST_DAYS, STRIDE_DAYS, args.start_window + args.windows
    )[args.start_window : args.start_window + args.windows]
    seeds = list(range(args.seeds))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"st-train-v020: {len(seeds)} seed × {len(anchors)} 窗口 | device={device} | "
        f"ts_freeze {EP1}+{EP2}ep | M4={M4_MODE}/levels={M4_LEVELS} | T={T} H={H}",
        flush=True,
    )

    agg = {s: {wi: None for wi in range(len(anchors))} for s in seeds}
    t_start = time.time()
    for wi, (start, tr_ts, end) in enumerate(anchors):
        t_w = time.time()
        ds = DailyTensorBuilder(cfg).build(
            args.parquet, start_ts=start, tr_ts=tr_ts, end_ts=end
        )
        delta_scale = cfg.delta_scale
        X_flat, y_abs, y_delta, cur = ds.X_flat, ds.y_abs, ds.y_delta, ds.cur
        bloom, strat = ds.bloom, ds.strat
        n_tr = ds.n_train
        y_norm = (y_delta / delta_scale).astype(np.float32)
        warn = make_m4_labels(y_abs, n_tr, mode=M4_MODE, bloom=bloom)
        print(
            f"\n[窗口 {wi+1}/{len(anchors)}] {start.date()}→{tr_ts.date()}→{end.date()}  "
            f"n={len(X_flat)} n_tr={n_tr} delta_scale={delta_scale:.3f} "
            f"bloom正例={int(bloom.sum())}/{len(bloom)} strat正例={int(strat.sum())}",
            flush=True,
        )
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)
            model = RamsNet(
                feat_dim=X_flat.shape[2],
                horizon=H,
                hidden=HIDDEN,
                use_m4=True,
                n_quantiles=N_Q,
                n_levels=M4_LEVELS,
            )
            trn = Trainer(model, device=device)
            trn.fit_two_stage(
                X_flat[:n_tr],
                y_norm[:n_tr],
                strat[:n_tr],
                warn[:n_tr],
                ep1=2 if args.fast_dev else EP1,
                ep2=2 if args.fast_dev else EP2,
                batch_size=BATCH,
                freeze_backbone=True,
                fast_dev_run=args.fast_dev,
            )
            te = slice(n_tr, len(X_flat))
            r = eval_window(
                model, X_flat[te], y_abs[te], cur[te], strat[te], warn[te],
                delta_scale, device,
            )
            agg[s][wi] = r
            print(
                f"  seed{s}: CRPS={r['crps']:.4f} 持久化={r['crps_p']:.4f} "
                f"技能={r['skill']:+.1f}% RMSE={r['rmse']:.3f} 覆盖={r['coverage']:.3f} "
                f"M2={r['m2_acc']:.3f} M4={r['m4_acc'] if r['m4_acc'] is not None else '—'}",
                flush=True,
            )
        print(f"  （窗口耗时 {time.time()-t_w:.0f}s，累计 {time.time()-t_start:.0f}s）", flush=True)

    # ---------------- 聚合 ----------------
    Nw = len(anchors)
    win = []
    for wi in range(Nw):
        cr = [agg[s][wi]["crps"] for s in seeds]
        sk = [agg[s][wi]["skill"] for s in seeds]
        cv = [agg[s][wi]["coverage"] for s in seeds]
        rm = [agg[s][wi]["rmse"] for s in seeds]
        m2 = [agg[s][wi]["m2_acc"] for s in seeds]
        m4 = [agg[s][wi]["m4_acc"] for s in seeds]
        win.append(
            {
                "window": wi + 1,
                "start": str(anchors[wi][0].date()),
                "tr": str(anchors[wi][1].date()),
                "end": str(anchors[wi][2].date()),
                "crps_mean": float(np.mean(cr)),
                "crps_std": float(np.std(cr)),
                "skill_mean": float(np.mean(sk)),
                "coverage_mean": float(np.mean(cv)),
                "rmse_mean": float(np.mean(rm)),
                "m2_acc": float(np.mean(m2)),
                "m4_acc": float(np.mean(m4)),
                "n_test": agg[seeds[0]][wi]["n_test"],
            }
        )

    crps_h = np.array([[agg[s][wi]["crps_h"] for wi in range(Nw)] for s in seeds])  # (S,Nw,H)
    crps_p_h = np.array([[agg[s][wi]["crps_p_h"] for wi in range(Nw)] for s in seeds])
    horizons = []
    for h in range(H):
        cp = float(crps_p_h[:, :, h].mean())
        ch = float(crps_h[:, :, h].mean())
        horizons.append(
            {
                "h": h + 1,
                "crps_mean": ch,
                "crps_std": float(crps_h[:, :, h].std()),
                "crps_p": cp,
                "skill": float((cp - ch) / cp * 100),
            }
        )

    crps_all = np.array([agg[s][wi]["crps"] for s in seeds for wi in range(Nw)])
    skill_all = np.array([agg[s][wi]["skill"] for s in seeds for wi in range(Nw)])
    cover_all = np.array([agg[s][wi]["coverage"] for s in seeds for wi in range(Nw)])
    rmse_all = np.array([agg[s][wi]["rmse"] for s in seeds for wi in range(Nw)])
    m2_all = np.array([agg[s][wi]["m2_acc"] for s in seeds for wi in range(Nw)])
    m4_all = np.array([agg[s][wi]["m4_acc"] for s in seeds for wi in range(Nw)])
    crps_p_all = np.array([agg[s][wi]["crps_p"] for s in seeds for wi in range(Nw)])
    overall = {
        "crps_mean": float(crps_all.mean()),
        "crps_std": float(crps_all.std()),
        "crps_p": float(crps_p_all.mean()),
        "skill_mean": float(skill_all.mean()),
        "skill_std": float(skill_all.std()),
        "coverage_mean": float(cover_all.mean()),
        "coverage_std": float(cover_all.std()),
        "rmse_mean": float(rmse_all.mean()),
        "rmse_std": float(rmse_all.std()),
        "m2_acc": float(m2_all.mean()),
        "m4_acc": float(m4_all.mean()),
    }
    per_seed = {
        str(s): {
            "crps": float(np.mean([agg[s][wi]["crps"] for wi in range(Nw)])),
            "skill": float(np.mean([agg[s][wi]["skill"] for wi in range(Nw)])),
            "coverage": float(np.mean([agg[s][wi]["coverage"] for wi in range(Nw)])),
            "rmse": float(np.mean([agg[s][wi]["rmse"] for wi in range(Nw)])),
        }
        for s in seeds
    }

    # ---------------- 打印 ----------------
    print("\n================ 逐窗口（3-seed 均值） ================", flush=True)
    hdr = f"{'窗口':>3} {'CRPS':>8} {'技能%':>7} {'RMSE':>7} {'覆盖':>6} {'M2':>6} {'M4':>6}"
    print(hdr, flush=True)
    for w in win:
        print(
            f"{w['window']:>3} {w['crps_mean']:>8.4f} {w['skill_mean']:>+7.1f} "
            f"{w['rmse_mean']:>7.3f} {w['coverage_mean']:>6.3f} {w['m2_acc']:>6.3f} "
            f"{w['m4_acc']:>6.3f}",
            flush=True,
        )
    print("\n================ 逐视界 CRPS（seed×窗口 展平） ================", flush=True)
    for h in horizons:
        print(
            f"  h{h['h']}: CRPS={h['crps_mean']:.4f}±{h['crps_std']:.4f} "
            f"持久化={h['crps_p']:.4f} 技能={h['skill']:+.1f}%",
            flush=True,
        )
    print("\n================ 总体（3-seed × 17 窗口展平） ================", flush=True)
    print(
        f"  CRPS {overall['crps_mean']:.4f} ± {overall['crps_std']:.4f} | "
        f"持久化 {overall['crps_p']:.4f} | 相对技能 {overall['skill_mean']:+.1f}% ± {overall['skill_std']:.1f}",
        flush=True,
    )
    print(
        f"  覆盖率 [p10,p90] {overall['coverage_mean']:.3f} ± {overall['coverage_std']:.3f} | "
        f"p50 RMSE {overall['rmse_mean']:.3f} ± {overall['rmse_std']:.3f}",
        flush=True,
    )
    print(
        f"  M2 分层 acc {overall['m2_acc']:.3f} | M4 预警 acc {overall['m4_acc']:.3f}",
        flush=True,
    )
    print("  3-seed 各 seed 总体:", flush=True)
    for s, v in per_seed.items():
        print(
            f"    seed{s}: CRPS {v['crps']:.4f} 技能 {v['skill']:+.1f}% "
            f"覆盖 {v['coverage']:.3f} RMSE {v['rmse']:.3f}",
            flush=True,
        )
    print(f"\n总耗时 {time.time()-t_start:.0f}s（未打印任何原始数据行）", flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "T": T, "H": H, "ep1": EP1, "ep2": EP2, "batch": BATCH, "hidden": HIDDEN,
                "n_quantiles": N_Q, "quantile_levels": list(QUANTILE_LEVELS),
                "seeds": seeds, "n_windows": Nw, "m4_mode": M4_MODE, "m4_levels": M4_LEVELS,
                "device": device, "protocol": "rolling 730/90/45",
            },
            "per_window": win,
            "per_horizon": horizons,
            "per_seed": per_seed,
            "overall": overall,
            "wall_seconds": round(time.time() - t_start, 1),
        }
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"结果已写: {out}", flush=True)


if __name__ == "__main__":
    main()
