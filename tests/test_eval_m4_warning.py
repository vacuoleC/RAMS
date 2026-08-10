"""mdl-m4-warning 评估逻辑单元测试（合成事件/概率序列，不依赖真实数据）。

覆盖 scripts/eval_m4_warning.py 的纯函数：
  - events_span：日期是否落在事件区间
  - warning_episodes：逐日布尔序列 → 连续预警段
  - evaluate_threshold：召回 / 提前量 / 误报 计算

数据保密红线：全部用合成事件区间 / 合成概率序列，不读原始数据。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_MOD_PATH = _ROOT / "scripts" / "eval_m4_warning.py"


@pytest.fixture(scope="module")
def ev():
    """导入 scripts/eval_m4_warning.py（避免重复 import）。"""
    if _MOD_PATH in sys.modules:
        return sys.modules[_MOD_PATH]
    spec = importlib.util.spec_from_file_location("eval_m4_warning_mod", _MOD_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[_MOD_PATH] = mod
    return mod


class TestEventsSpan:
    def test_inside(self, ev):
        d = pd.Timestamp("2023-01-02")
        assert ev.events_span([("2023-01-01", "2023-01-03")], d) is True

    def test_boundary_inclusive(self, ev):
        ev_list = [("2023-01-01", "2023-01-03")]
        assert ev.events_span(ev_list, pd.Timestamp("2023-01-01")) is True
        assert ev.events_span(ev_list, pd.Timestamp("2023-01-03")) is True

    def test_outside(self, ev):
        assert ev.events_span([("2023-01-01", "2023-01-03")], pd.Timestamp("2023-01-05")) is False

    def test_empty_events(self, ev):
        assert ev.events_span([], pd.Timestamp("2023-01-02")) is False


class TestWarningEpisodes:
    def test_basic(self, ev):
        mask = np.array([True, True, False, True, False, True, True, True])
        eps = ev.warning_episodes(mask)
        assert eps == [(0, 2), (3, 4), (5, 8)]

    def test_no_warning(self, ev):
        assert ev.warning_episodes(np.zeros(10, dtype=bool)) == []

    def test_all_warning(self, ev):
        assert ev.warning_episodes(np.ones(5, dtype=bool)) == [(0, 5)]


class TestEvaluateThreshold:
    def _setup(self):
        """合成 60 天场景：1 个事件 + 1 个命中预警 + 1 个误报预警。"""
        dates = pd.date_range("2023-01-01", periods=60)
        prob = np.zeros(60)
        # 命中预警：事件(01-15~01-17) lead 窗口内，01-11 触发（lead=4 天）
        prob[10] = 0.9
        # 误报预警：01-31，其后 Lmax 内无事件 start
        prob[30] = 0.9
        ev_span = [(pd.Timestamp("2023-01-15"), pd.Timestamp("2023-01-17"))]
        return prob, dates, ev_span

    def test_hit_lead_and_recall(self, ev):
        prob, dates, ev_span = self._setup()
        r = ev.evaluate_threshold(prob, dates, ev_span, theta=0.5, Lmax=10)
        assert r["n_events"] == 1
        assert r["n_hit"] == 1
        assert r["recall"] == 1.0
        # 提前量 = 事件 start(01-15) − 预警段首日(01-11) = 4 天
        assert r["lead_median"] == 4.0
        assert r["lead_min"] == 4.0 and r["lead_max"] == 4.0
        assert r["per_event"][0]["hit"] is True
        assert r["per_event"][0]["lead_days"] == 4
        # 01-11 的预警段引致事件（其后 4 天内事件 start）→ 不计误报
        assert r["false_positive_episodes"] == 1  # 只有 01-31 那段
        assert r["false_positive_days"] == 1

    def test_lead_window_exceeded_not_hit(self, ev):
        prob, dates, ev_span = self._setup()
        # 预警在 01-11，事件 01-15 → lead=4；Lmax=2 时 01-11 不在 [start-2, start) 内 → 不命中
        r = ev.evaluate_threshold(prob, dates, ev_span, theta=0.5, Lmax=2)
        assert r["n_hit"] == 0
        assert r["recall"] == 0.0
        assert all(not pe["hit"] for pe in r["per_event"])

    def test_no_event_recall_nan(self, ev):
        prob, dates, _ = self._setup()
        r = ev.evaluate_threshold(prob, dates, [], theta=0.5, Lmax=10)
        assert r["n_events"] == 0
        assert np.isnan(r["recall"])
        assert r["n_hit"] == 0

    def test_threshold_controls_warning(self, ev):
        prob, dates, ev_span = self._setup()
        # θ=0.95：无预警（最高 0.9）→ 无预警段、无命中
        r = ev.evaluate_threshold(prob, dates, ev_span, theta=0.95, Lmax=10)
        assert r["n_hit"] == 0
        assert r["n_warning_episodes"] == 0

    def test_baseline_probability_functions(self, ev):
        """合成布尔预警（如顶层带 p75 基线）直接喂给 evaluate_threshold。"""
        dates = pd.date_range("2023-01-01", periods=20)
        warn = np.zeros(20)
        warn[5:8] = 1.0  # 连续预警段 (5,8)
        ev_span = [(pd.Timestamp("2023-01-10"), pd.Timestamp("2023-01-11"))]
        r = ev.evaluate_threshold(warn, dates, ev_span, theta=0.5, Lmax=10)
        # 预警段首日 01-06 → 提前 4 天，命中
        assert r["n_hit"] == 1
        assert r["lead_median"] == 4.0
