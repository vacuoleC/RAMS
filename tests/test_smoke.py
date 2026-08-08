# -*- coding: utf-8 -*-
"""RAMS 冒烟测试：核心模块快速验证（不依赖全量数据）。

用合成小数据验证：
  - 模型前向/输出形状
  - 数据管线（张量构建）
  - 训练循环（fast_dev_run 式）
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rams.models.rams_net import RamsNet, count_parameters  # noqa: E402


class TestRamsNet:
    """模型结构冒烟。"""

    def test_forward_shape(self):
        torch = pytest.importorskip("torch")
        net = RamsNet(feat_dim=26, horizon=8, quantile=True)
        x = torch.randn(4, 24, 26)
        m1, m2, m4 = net(x)
        assert m1.shape == (4, 24), f"M1 形状错误: {m1.shape}"
        assert m2.shape == (4, 2), f"M2 形状错误: {m2.shape}"
        assert m4 is not None and m4.shape == (4, 4), f"M4 形状错误: {m4.shape}"

    def test_forward_small(self):
        torch = pytest.importorskip("torch")
        net = RamsNet(feat_dim=6, horizon=3, hidden=8, quantile=False)
        x = torch.randn(2, 10, 6)
        m1, m2, m4 = net(x)
        assert m1.shape == (2, 3)
        assert m2.shape == (2, 2)
        assert m4.shape == (2, 4)

    def test_no_m4(self):
        torch = pytest.importorskip("torch")
        net = RamsNet(feat_dim=6, horizon=3, hidden=8, use_m4=False)
        m1, m2, m4 = net(torch.randn(2, 10, 6))
        assert m4 is None

    def test_parameters(self):
        torch = pytest.importorskip("torch")
        net = RamsNet(feat_dim=26, horizon=8)
        n = count_parameters(net)
        # 探索版 GRU backbone，参数量应远小于 1.9M（正式版可放宽）
        assert 0 < n < 1_000_000

    def test_predict_interval(self):
        torch = pytest.importorskip("torch")
        net = RamsNet(feat_dim=6, horizon=3, hidden=8, quantile=True)
        m1, _, _ = net(torch.randn(2, 10, 6))
        pred = net.predict_mean(m1)
        p10, p90 = net.predict_interval(m1)
        assert pred.shape == (2, 3)
        assert p10.shape == (2, 3) and p90.shape == (2, 3)


class TestTensorBuilder:
    """张量构建冒烟（合成长表）。"""

    def _make_parquet(self, tmp_path):
        pd = pytest.importorskip("pandas")
        # 合成: 60 时刻 × 3 深度，含温度/浓度/气象
        n_t, n_d = 60, 3
        t = pd.date_range("2024-01-01", periods=n_t, freq="3h")
        rows = []
        for ti in t:
            for d in [0.5, 1.0, 1.5]:
                rows.append({
                    "site_id": "0902", "timestamp": ti, "depth": d,
                    "water_temp": 20 + d, "total_conc": 5 + d * 2,
                    "wind_speed": 2.0, "wind_dir": 100, "pressure": 1000,
                    "air_temp": 21, "humidity": 60, "rainfall": 0,
                })
        df = pd.DataFrame(rows)
        p = tmp_path / "standard.parquet"
        df.to_parquet(p, index=False)
        return p

    def test_build(self, tmp_path):
        np.random.seed(0)
        p = self._make_parquet(tmp_path)
        from rams.data.tensor_builder import TensorBuilder, TensorConfig

        cfg = TensorConfig(T=12, H=3)
        ds = TensorBuilder(cfg).build(p)
        X_tr, y_tr, s_tr, w_tr = ds["train"]
        # 合成表 3 深度 + 6 气象列 → feat_dim = 3 + 6 = 9
        assert X_tr.shape[2] == 9, f"feat_dim 应为 9 (3温+6气), 实际 {X_tr.shape[2]}"
        assert X_tr.shape[0] > 0
        assert s_tr is not None
        assert w_tr is None  # 默认 warn_as_task=False


class TestTrainer:
    """训练循环冒烟（合成小数据，fast_dev_run）。"""

    def test_fit_smoke(self):
        torch = pytest.importorskip("torch")
        from rams.training.trainer import Trainer

        torch.manual_seed(0)
        model = RamsNet(feat_dim=6, horizon=3, hidden=8, quantile=True)
        X = np.random.randn(64, 10, 6).astype(np.float32)
        y = np.random.randn(64, 3).astype(np.float32)
        s = np.random.randint(0, 2, 64)
        w = np.random.randint(0, 4, 64)  # 预警标签（四级）
        tr = Trainer(model, w_m2=3.0)
        tr.fit(X, y, s, X, y, s, warn_tr=w, warn_va=w, epochs=2, batch_size=16,
               fast_dev_run=True)
        res = tr.evaluate(X, y, s, w, y_sd=1.0)
        assert "rmse" in res and np.isfinite(res["rmse"])
        assert "acc" in res and np.isfinite(res["acc"])
        assert "warn_acc" in res and np.isfinite(res["warn_acc"])
