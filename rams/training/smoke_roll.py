"""mdl-model-integrate 冒烟：日级滚动窗口 × 两阶段训练 × q9 形状/loss/覆盖率验证。

不打印任何原始数据行；只输出形状 / 统计量（保密红线）。
"""

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# UTF-8 控制台输出（Windows）：仅在作为脚本运行时生效；
# 被 pytest 等导入时不劫持 sys.stdout（否则会包住/关闭 capture 缓冲）。
if __name__ == "__main__" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np  # noqa: E402

from rams.data.tensor_builder import (  # noqa: E402
    DailyConfig,
    DailyTensorBuilder,
    make_rolling_anchors,
)
from rams.models.rams_net import QUANTILE_LEVELS, RamsNet, count_parameters  # noqa: E402
from rams.training.trainer import Trainer, crps_cdf_pline, make_m4_labels  # noqa: E402


def main():
    parquet = "data/processed/standard.parquet"
    cfg = DailyConfig(T=30, H=7, delta_target=True)

    # T4 协议滚动窗口锚点（训练 730d / 测试 90d / 步长 45d）
    import pandas as pd  # noqa: PLC0415

    d0 = pd.Timestamp("2021-03-01")
    anchors = make_rolling_anchors(d0, train_days=730, test_days=90, stride_days=45, n_windows=1)
    start, tr, end = anchors[0]
    print(f"窗口: {start} → {tr} (训练截止) → {end}", flush=True)

    ds = DailyTensorBuilder(cfg).build(parquet, start_ts=start, tr_ts=tr, end_ts=end)
    X_flat, y_abs, y_delta = ds.X_flat, ds.y_abs, ds.y_delta
    cur, bloom, strat = ds.cur, ds.bloom, ds.strat
    n_tr = ds.n_train
    print(f"X_flat: {X_flat.shape}  X 剖面: {ds.X.shape}  y_delta: {y_delta.shape}", flush=True)
    print(f"n_train={n_tr}  n_total={len(X_flat)}  delta_scale={cfg.delta_scale:.3f}", flush=True)
    print(
        f"bloom 正例: {int(bloom.sum())}/{len(bloom)} ({float(bloom.mean()):.3f})  "
        f"strat 正例: {int(strat.sum())}",
        flush=True,
    )

    # 增量归一化目标（训练段 scale，防泄漏）
    y_norm = (y_delta / cfg.delta_scale).astype(np.float32)
    # M4 标签：藻华状态（N 定义，预测日标签）
    warn_bloom = make_m4_labels(y_abs, n_tr, mode="bloom", bloom=bloom)
    # M4 标签对照：未来峰值分位数（探索 A/K 协议）
    warn_peak = make_m4_labels(y_abs, n_tr, mode="peak_quantile")

    print("\n===== M4 标签 =====", flush=True)
    print(f"  bloom 模式等级分布: {np.bincount(warn_bloom, minlength=2).tolist()}", flush=True)
    print(
        f"  peak_quantile 模式等级分布: {np.bincount(warn_peak, minlength=4).tolist()}", flush=True
    )

    print("\n===== 两阶段训练（fast_dev_run 冒烟）=====", flush=True)
    torch_ok = True
    try:
        import torch  # noqa: PLC0415

        del torch
    except ImportError:
        torch_ok = False
        print("  无 torch，跳过训练")
    if torch_ok:
        model = RamsNet(
            feat_dim=X_flat.shape[2], horizon=cfg.H, hidden=64, use_m4=True, n_quantiles=9
        )
        print(f"  RamsNet 参数量: {count_parameters(model):,}", flush=True)
        tr = Trainer(model)
        h1, h2 = tr.fit_two_stage(
            X_flat[:n_tr],
            y_norm[:n_tr],
            strat[:n_tr],
            warn_bloom[:n_tr],
            ep1=2,
            ep2=2,
            batch_size=64,
            freeze_backbone=True,
            fast_dev_run=True,
        )
        assert np.isfinite(h1[-1][0]), "Stage1 loss 非有限"
        assert np.isfinite(h2[-1][0]), "Stage2 loss 非有限"
        print(f"  Stage1 末 loss: {h1[-1][0]:.4f}  Stage2 末 loss: {h2[-1][0]:.4f}", flush=True)

        # 测试段评估（还原 conc 单位）
        te = slice(n_tr, len(X_flat))
        q_norm = tr.predict_m1(X_flat[te])  # (N, 9, H) 归一化 Δ
        q_conc = cur[te][:, None, None] + q_norm * cfg.delta_scale  # (N, 9, H) conc 单位
        obs = y_abs[te]
        i10 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.10))[0][0])
        i50 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.50))[0][0])
        i90 = int(np.where(np.isclose(QUANTILE_LEVELS, 0.90))[0][0])
        cover = float(np.mean((obs >= q_conc[:, i10]) & (obs <= q_conc[:, i90])))
        rmse = float(np.sqrt(np.mean((q_conc[:, i50] - obs) ** 2)))
        crps_h = np.array(
            [
                np.mean(crps_cdf_pline(q_conc[:, :, h], QUANTILE_LEVELS, obs[:, h]))
                for h in range(cfg.H)
            ]
        )
        # 持久化（Δ≡0 → conc_t，全分位同值）
        n_q = len(QUANTILE_LEVELS)
        q_p = np.repeat(cur[te][:, None, None], cfg.H, axis=2)  # (N,1,H)
        q_p = np.repeat(q_p, n_q, axis=1)  # (N,n_q,H)
        crps_p = np.array(
            [
                np.mean(crps_cdf_pline(q_p[:, :, h], QUANTILE_LEVELS, obs[:, h]))
                for h in range(cfg.H)
            ]
        )
        skill = (crps_p.mean() - crps_h.mean()) / crps_p.mean() * 100
        print("\n===== 测试段（冒烟，随机初始化）=====", flush=True)
        print(f"  q9 形状: {q_norm.shape}  还原 conc: {q_conc.shape}", flush=True)
        print(
            f"  CRPS 均值: {crps_h.mean():.4f}  持久化: {crps_p.mean():.4f}  "
            f"相对技能: {skill:+.1f}%",
            flush=True,
        )
        print(f"  p50 RMSE: {rmse:.4f}  覆盖率 [p10,p90]: {cover:.3f}", flush=True)
        assert q_norm.shape[1] == 9, "q9 输出维度错误"
        assert 0.0 <= cover <= 1.0, "覆盖率越界"

    print("\n冒烟通过（未打印任何原始数据行）", flush=True)


if __name__ == "__main__":
    main()
