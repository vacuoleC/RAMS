"""RAMS 0.2.0 模型整合单元测试（mdl-model-integrate，合成小数据，不依赖真实数据）。

覆盖：
  - RamsNet q9 分位数前向形状 / quantile_matrix / predict_mean / predict_interval
  - 3 分位兼容（0.1.0 契约）
  - 两阶段训练（Stage1 单任务 M1 → Stage2 冻结 backbone 多任务，fast_dev_run）
  - 多任务 loss（w=1/3/2）、分位数 loss 有限
  - CRPS（q9 分段线性 / 3 分位闭合形式）
  - M4 标签（peak_quantile / bloom 两种模式）
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rams.models.rams_net import QUANTILE_LEVELS, RamsNet, count_parameters  # noqa: E402
from rams.training.trainer import (  # noqa: E402
    MultiTaskLoss,
    QuantileLoss,
    Trainer,
    crps_cdf_pline,
    crps_quantiles,
    make_m4_labels,
)


def _rand(*shape):
    """合成随机小张量（测试用，不涉数据）。"""
    import torch

    return torch.randn(*shape)


class TestRamsNetQ9:
    """q9 分位数模型结构冒烟。"""

    def test_q9_forward_shape(self):
        pytest.importorskip("torch")
        net = RamsNet(feat_dim=29, horizon=7, use_m4=True, n_quantiles=9)
        m1, m2, m4 = net(_rand(4, 30, 29))
        assert m1.shape == (4, 63), f"M1 q9 形状错误: {m1.shape}"  # 9×7
        assert m2.shape == (4, 2), f"M2 形状错误: {m2.shape}"
        assert m4 is not None, "M4 头不存在"
        assert m4.shape == (4, 4), f"M4 形状错误: {m4.shape}"
        assert net.n_quantiles == 9

    def test_quantile_matrix_and_mean_interval(self):
        pytest.importorskip("torch")
        net = RamsNet(feat_dim=29, horizon=7, use_m4=True, n_quantiles=9)
        m1, _, _ = net(_rand(4, 30, 29))
        qm = net.quantile_matrix(m1)
        assert qm.shape == (4, 9, 7), f"quantile_matrix 形状错误: {qm.shape}"
        p50 = net.predict_mean(m1)
        assert p50.shape == (4, 7), f"p50 形状错误: {p50.shape}"
        # p50 应为中间结（0.50 在索引 4）
        np.testing.assert_allclose(p50.detach().numpy(), qm[:, 4].detach().numpy(), atol=1e-6)
        p10, p90 = net.predict_interval(m1)
        assert p10.shape == (4, 7), f"p10 形状错误: {p10.shape}"
        assert p90.shape == (4, 7), f"p90 形状错误: {p90.shape}"
        # p10/p90 应为结 0.10(索引1) / 0.90(索引7)
        np.testing.assert_allclose(p10.detach().numpy(), qm[:, 1].detach().numpy(), atol=1e-6)
        np.testing.assert_allclose(p90.detach().numpy(), qm[:, 7].detach().numpy(), atol=1e-6)

    def test_legacy_3quantile_compat(self):
        pytest.importorskip("torch")
        net = RamsNet(feat_dim=26, horizon=8, quantile=True, n_quantiles=3)
        m1, _, _ = net(_rand(4, 24, 26))
        assert m1.shape == (4, 24), f"M1 形状错误: {m1.shape}"
        p50 = net.predict_mean(m1)
        p10, p90 = net.predict_interval(m1)
        assert p50.shape == (4, 8), f"p50 形状错误: {p50.shape}"
        assert p10.shape == (4, 8), f"p10 形状错误: {p10.shape}"
        assert p90.shape == (4, 8), f"p90 形状错误: {p90.shape}"

    def test_parameters_budget(self):
        pytest.importorskip("torch")
        net = RamsNet(feat_dim=29, horizon=7, hidden=64, n_quantiles=9)
        n = count_parameters(net)
        assert 0 < n < 200_000, f"参数量异常: {n}"


class TestLosses:
    """分位数损失 + 多任务损失。"""

    def test_quantile_loss_finite(self):
        torch = pytest.importorskip("torch")
        pred = torch.randn(8, 63)
        target = torch.randn(8, 7)
        ql = QuantileLoss(n_quantiles=9)
        loss = ql(pred, target)
        assert torch.isfinite(loss).item()

    def test_multi_task_loss(self):
        torch = pytest.importorskip("torch")
        m1 = torch.randn(8, 63)
        m2 = torch.randn(8, 2)
        m4 = torch.randn(8, 4)
        y = torch.randn(8, 7)
        s = torch.randint(0, 2, (8,))
        w = torch.randint(0, 4, (8,))
        crit = MultiTaskLoss(horizon=7, n_quantiles=9, use_m4=True)
        total, l1, l2, l4 = crit(m1, m2, y, s, m4, w)
        assert torch.isfinite(total).item()
        assert torch.isfinite(l1).item(), "M1 分位数 loss 非有限"
        assert torch.isfinite(l2).item(), "M2 loss 非有限"
        assert l4 is not None, "M4 loss 应为非空"
        assert torch.isfinite(l4).item(), "M4 loss 非有限"
        # w=1/3/2：total ≈ l1 + 3·l2 + 2·l4
        expected = l1 + 3.0 * l2 + 2.0 * l4
        np.testing.assert_allclose(total.item(), expected.item(), atol=1e-5)


class TestTrainerTwoStage:
    """两阶段训练（fast_dev_run）冒烟。"""

    def _data(self, n=64, t=10, f=6, h=3):
        pytest.importorskip("torch")
        rng = np.random.default_rng(0)
        x = rng.normal(size=(n, t, f)).astype(np.float32)
        y = rng.normal(size=(n, h)).astype(np.float32)
        s = rng.integers(0, 2, n)
        w = rng.integers(0, 4, n)
        return x, y, s, w

    def test_two_stage_fast_dev_run(self):
        pytest.importorskip("torch")
        x, y, s, w = self._data()
        model = RamsNet(feat_dim=6, horizon=3, hidden=8, use_m4=True, n_quantiles=9)
        tr = Trainer(model)
        h1, h2 = tr.fit_two_stage(
            x, y, s, w, ep1=2, ep2=2, batch_size=16, freeze_backbone=True, fast_dev_run=True
        )
        assert np.isfinite(h1[-1][0]), "Stage1 loss 非有限"
        assert np.isfinite(h2[-1][0]), "Stage2 loss 非有限"
        # Stage2 冻结 backbone：只训头，backbone 参数可训练状态恢复
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_fit_multi_freeze_restores_grad(self):
        pytest.importorskip("torch")
        x, y, s, w = self._data()
        model = RamsNet(feat_dim=6, horizon=3, hidden=8, use_m4=True, n_quantiles=9)
        tr = Trainer(model)
        tr.fit_multi(x, y, s, w, epochs=2, batch_size=16, freeze_backbone=True, fast_dev_run=True)
        # 训练结束后 requires_grad 恢复 True
        assert all(p.requires_grad for p in model.backbone.parameters())

    def test_evaluate_metrics(self):
        pytest.importorskip("torch")
        x, y, s, w = self._data()
        model = RamsNet(feat_dim=6, horizon=3, hidden=8, use_m4=True, n_quantiles=9)
        tr = Trainer(model)
        res = tr.evaluate(x, y, s, w, y_sd=1.0)
        assert np.isfinite(res["rmse"]), "RMSE 非有限"
        assert np.isfinite(res["acc"]), "M2 acc 非有限"
        assert np.isfinite(res["warn_acc"]), "M4 acc 非有限"
        assert 0.0 <= res["coverage"] <= 1.0, "覆盖率越界"

    def test_predict_m1_shape(self):
        pytest.importorskip("torch")
        x, y, s, w = self._data()
        model = RamsNet(feat_dim=6, horizon=3, hidden=8, use_m4=True, n_quantiles=9)
        tr = Trainer(model)
        q = tr.predict_m1(x)
        assert q.shape == (64, 9, 3), f"q9 预测形状错误: {q.shape}"


class TestCRPS:
    """CRPS 评估（T4 协议）。"""

    def test_crps_q9_finite(self):
        rng = np.random.default_rng(1)
        q = np.sort(rng.normal(size=(20, 9)), axis=-1)  # (N, 9) 升序分位数
        y = rng.normal(size=20)
        crps = crps_cdf_pline(q, QUANTILE_LEVELS, y)
        assert np.all(np.isfinite(crps))
        assert np.all(crps >= 0)

    def test_crps_3quantile_equivalent(self):
        rng = np.random.default_rng(2)
        q10 = rng.normal(size=20)
        q50 = rng.normal(size=20)
        q90 = q50 + np.abs(rng.normal(size=20))
        y = rng.normal(size=20)
        c1 = crps_quantiles(q10, q50, q90, y)
        qs = np.sort(np.stack([q10, q50, q90], axis=-1), axis=-1)
        c2 = crps_cdf_pline(qs, [0.1, 0.5, 0.9], y)
        np.testing.assert_allclose(c1, c2, atol=1e-12)

    def test_persist_skill_positive(self):
        # 持久化 CRPS = E|y - conc_t|（Δ≡0）；模型分位数若更准，skill>0
        rng = np.random.default_rng(3)
        y = rng.normal(size=(50,))
        scale = np.abs(y).max() + 1.0
        q = y[:, None] * 0.1 + np.sort(np.linspace(0.05, 0.95, 9)) * scale * 0.05
        q = np.sort(q, axis=-1)
        crps_model = crps_cdf_pline(q, QUANTILE_LEVELS, y).mean()
        cur = np.zeros(50)  # conc_t
        q_p = np.repeat(cur[:, None], 9, axis=1)
        crps_persist = crps_cdf_pline(q_p, QUANTILE_LEVELS, y).mean()
        skill = (crps_persist - crps_model) / crps_persist * 100
        assert skill > 0, f"技能应为正: {skill:.2f}%"


class TestM4Labels:
    """M4 预警标签（日级协议）。"""

    def test_peak_quantile(self):
        rng = np.random.default_rng(4)
        y = np.abs(rng.normal(size=(100, 7)))
        labels = make_m4_labels(y, n_train=60, mode="peak_quantile")
        assert labels.shape == (100,)
        assert set(np.unique(labels)).issubset({0, 1, 2, 3})

    def test_bloom_mode(self):
        bloom = np.array([0, 1, 1, 0, 1, 0, 0, 1])
        y = np.zeros((8, 7))
        labels = make_m4_labels(y, n_train=4, mode="bloom", bloom=bloom)
        np.testing.assert_array_equal(labels, bloom)
        with pytest.raises(ValueError, match="bloom"):
            make_m4_labels(y, n_train=4, mode="bloom")  # 缺 bloom 标签
