"""RAMS 0.2.0 日级数据管线单元测试（mdl-data-scale，合成小数据，不依赖真实数据）。

覆盖：
  - DailyTensorBuilder：日级张量形状 (B, T=30, D, C)、目标口径（原始单位 + 增量 Δ）、
    训练段归一化防泄漏、M3 选层输入（5 层）、滚动窗口切分（T4 协议）。
  - BloomLabeler：藻华状态/事件标签（N 定义，顶层带 + 多层联动 + 连续 ≥2 天）。
  - TensorBuilder（3h 兼容管线）：形状与 0.1.0 契约。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.data.tensor_builder import (  # noqa: E402
    M3_RECOMMENDED_DEPTHS,
    BloomLabeler,
    DailyConfig,
    DailyTensorBuilder,
    TensorBuilder,
    TensorConfig,
    make_rolling_anchors,
)


def _make_parquet(tmp_path, n_days=120, depths=(0.5, 1.0, 1.5, 2.5, 3.0, 5.0, 10.0), seed=0):
    """合成 3h 网格长表（含 20 层中的子集 + 气象），写成 parquet。"""
    rng = np.random.default_rng(seed)
    times = pd.date_range("2022-01-01", periods=n_days * 8, freq="3h")
    rows = []
    for ti in times:
        for d in depths:
            # 藻华在 2022-03 构造高位（顶层带 + 联动）
            in_bloom = (ti.month == 3) and (ti.day >= 10)
            base = 60.0 if in_bloom else (5.0 if d <= 3.0 else 3.0)
            conc = max(0.0, base + rng.normal(0, 2.0))
            rows.append({
                "site_id_x": 902,
                "timestamp": ti,
                "depth": d,
                "water_temp": 18.0 - 0.5 * d + rng.normal(0, 0.3),
                "total_conc": conc,
                "wind_speed": 2.0, "wind_dir": 100.0, "pressure": 1000.0,
                "air_temp": 20.0, "humidity": 60.0, "rainfall": 0.0,
            })
    df = pd.DataFrame(rows)
    p = tmp_path / "standard.parquet"
    df.to_parquet(p, index=False)
    return p


class TestDailyTensorBuilder:
    def test_shape_full_profile(self, tmp_path):
        p = _make_parquet(tmp_path)
        cfg = DailyConfig(T=30, H=7)
        ds = DailyTensorBuilder(cfg).build(p)
        n_days = 120
        n_w = n_days - cfg.T - cfg.H  # 83
        # 合成子集只有 7 层 → D=7
        assert ds.X.shape == (n_w, cfg.T, 7, 2), ds.X.shape
        assert ds.X_flat.shape[0] == n_w
        assert ds.X_flat.shape[1] == cfg.T
        assert ds.y_abs.shape == (n_w, cfg.H)
        assert ds.y_delta.shape == (n_w, cfg.H)
        assert ds.bloom.shape == (n_w,)
        assert ds.strat.shape == (n_w,)
        assert ds.X.dtype == np.float32
        assert not np.isnan(ds.X).any()

    def test_targets_raw_units(self, tmp_path):
        """目标（y_abs/cur/y_delta）保持原始 conc 单位，而非归一化。"""
        p = _make_parquet(tmp_path)
        cfg = DailyConfig(T=30, H=7)
        ds = DailyTensorBuilder(cfg).build(p)
        # 合成数据浓度在 0-70 之间，若被 z 归一化会变成负值/小值
        assert ds.y_abs.max() > 10.0, f"目标疑似归一化: max={ds.y_abs.max():.2f}"
        np.testing.assert_allclose(ds.y_delta, ds.y_abs - ds.cur[:, None], atol=1e-8)

    def test_feature_normalized_train_fit(self, tmp_path):
        """特征按训练段归一化（训练段均值≈0），防泄漏。"""
        p = _make_parquet(tmp_path)
        cfg = DailyConfig(T=30, H=7)
        b = DailyTensorBuilder(cfg)
        ds = b.build(p)
        n_tr = ds.n_train
        # 训练段首个特征列（temp_0.5）归一化后均值应接近 0
        tr_means = ds.X_flat[:n_tr, :, 0].mean()
        assert abs(tr_means) < 0.3, f"训练段特征未归一化: {tr_means:.3f}"

    def test_m3_selected_layers(self, tmp_path):
        p = _make_parquet(tmp_path)
        cfg = DailyConfig(T=30, H=7, m3_depths=M3_RECOMMENDED_DEPTHS)
        ds = DailyTensorBuilder(cfg).build(p)
        # 合成深度 (0.5,1.0,1.5,2.5,3.0,5.0,10.0) 中属于 M3 子集 (1.5,5.0,8.5,9.5,10.0) 的 = 3
        present = [d for d in M3_RECOMMENDED_DEPTHS if d in (0.5, 1.0, 1.5, 2.5, 3.0, 5.0, 10.0)]
        assert ds.X.shape[2] == len(present) == 3
        assert ds.X.shape == (83, 30, 3, 2)

    def test_rolling_window_split(self, tmp_path):
        p = _make_parquet(tmp_path)
        cfg = DailyConfig(T=30, H=7)
        d0 = pd.Timestamp("2022-01-01")
        anchors = make_rolling_anchors(d0, train_days=60, test_days=20, stride_days=30, n_windows=1)
        start, tr, end = anchors[0]
        ds = DailyTensorBuilder(cfg).build(p, start_ts=start, tr_ts=tr, end_ts=end)
        assert ds.n_train > 0
        assert ds.n_train < len(ds.X)
        # 协议（与探索 run_l.py 一致）：训练样本 = 预测目标末端 < tr；测试样本的目标延伸过 tr。
        # 因此前 H 个测试样本的预测日（窗口末天）可略早于 tr，但不得超过 H 天。
        assert (ds.dates[: ds.n_train] < tr).all()
        assert ds.dates[ds.n_train] >= tr - pd.Timedelta(days=cfg.H)
        assert ds.dates[ds.n_train] <= tr + pd.Timedelta(days=cfg.H)

    def test_rolling_split_when_trts_absent(self, tmp_path):
        """回归（mdl-baseline-compare 发现的 w12 缺口 bug）：tr_ts 不在日级索引时，
        n_train 必须仍按行计数（daily.index < tr_ts），不得退回 fit_frac。
        数据缺口日恰好是切点 → 旧实现会把约 30% 训练段样本误当测试样本。"""
        p = _make_parquet(tmp_path)
        cfg = DailyConfig(T=30, H=7)
        d0 = pd.Timestamp("2022-01-01")
        anchors = make_rolling_anchors(d0, train_days=60, test_days=20, stride_days=30, n_windows=1)
        start, tr, end = anchors[0]
        # 无缺口基线：n_train 按行计数
        ds_ok = DailyTensorBuilder(cfg).build(p, start_ts=start, tr_ts=tr, end_ts=end)
        n_tr_correct = ds_ok.n_train
        # 有缺口：从合成数据删掉 tr 当天的所有 3h 行，另写 parquet（模拟数据缺口切点）
        df = pd.read_parquet(p)
        gap_ts = tr
        df_gap = df[~df["timestamp"].dt.floor("3h").eq(gap_ts)]
        p_gap = tmp_path / "standard_gap.parquet"
        df_gap.to_parquet(p_gap, index=False)
        ds_gap = DailyTensorBuilder(cfg).build(p_gap, start_ts=start, tr_ts=tr, end_ts=end)
        # 正确 n_train：预测目标末端 < tr（行计数），缺口只缺 tr 当天、不影响 < tr 计数
        assert ds_gap.n_train > 0
        assert ds_gap.n_train < len(ds_gap.X)
        assert ds_gap.n_train == n_tr_correct, (
            f"切点缺日不得改变 n_train：{ds_gap.n_train} vs {n_tr_correct}"
        )
        # 且必须远小于 0.7*len(X)（旧 bug 的退回值）
        assert ds_gap.n_train < 0.75 * len(ds_gap.X)


class TestBloomLabeler:
    def test_bloom_label_positive_in_synthetic_bloom(self, tmp_path):
        """2022-03-10 起构造连续高位 + 联动 → 应产生藻华正例。"""
        p = _make_parquet(tmp_path)
        b = DailyTensorBuilder(DailyConfig(T=30, H=7))
        raw = b.load_daily_wide(p)
        lab = BloomLabeler(DailyConfig())
        sig = lab.predict(raw)
        # 状态日（未合并）
        n_pos = int(sig.sum())
        assert n_pos >= 10, f"合成藻华应产生正例，实际 {n_pos}"
        # 事件：≥2 天连续
        evs = lab.events(sig, pd.DatetimeIndex(raw.index))
        assert len(evs) >= 1
        assert all(e["n_days"] >= 2 for e in evs)

    def test_negative_control(self, tmp_path):
        """无藻华的合成数据（恒定低浓度）→ 无正例。"""
        times = pd.date_range("2022-01-01", periods=120 * 8, freq="3h")
        rows = []
        for ti in times:
            for d in (0.5, 1.0, 1.5, 2.5, 3.0, 5.0, 10.0):
                rows.append({
                    "site_id_x": 902, "timestamp": ti, "depth": d,
                    "water_temp": 18.0, "total_conc": 2.0,
                    "wind_speed": 2.0, "wind_dir": 100.0, "pressure": 1000.0,
                    "air_temp": 20.0, "humidity": 60.0, "rainfall": 0.0,
                })
        df = pd.DataFrame(rows)
        p2 = tmp_path / "standard_low.parquet"
        df.to_parquet(p2, index=False)
        b = DailyTensorBuilder(DailyConfig())
        raw = b.load_daily_wide(p2)
        sig = BloomLabeler(DailyConfig()).predict(raw)
        assert int(sig.sum()) == 0, f"低浓度数据不应有藻华状态: {int(sig.sum())}"


class TestTensorBuilderCompat:
    def test_legacy_build(self, tmp_path):
        """0.1.0 兼容管线（3h 网格）形状契约。"""
        p = _make_parquet(tmp_path)
        cfg = TensorConfig(T=12, H=3)
        ds = TensorBuilder(cfg).build(p)
        x_legacy, y_legacy, s, w = ds["train"]
        # 合成子集 7 层水温 + 6 气象 = 13 特征
        assert x_legacy.shape[1] == 12
        assert x_legacy.shape[2] == 13, x_legacy.shape
        assert x_legacy.shape[0] > 0
        assert ds["feat_dim"] == 13
        assert s is not None
        assert w is None  # warn_as_task 默认 False
