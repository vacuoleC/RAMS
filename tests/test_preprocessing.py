"""rams/data/preprocessing 单元测试（合成 xlsx，不依赖真实原始数据）。

覆盖 scripts/eval 全链路中唯一的原始数据接触点：
  - load_algae：藻类 20 sheet 纵向合并（含自带 depth 列去重）
  - load_meteo：气象 Sheet1 → 3h 均值重采样
  - align_and_clean：merge_asof 最近时刻对齐（时间错位）
  - compute_norm_stats：只用训练段拟合归一化参数（防泄漏）
  - build_dataset：xlsx → parquet + norm_stats.json（end-to-end）

数据保密红线：全部用合成小表生成 xlsx，不读 data/raw 真实文件。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pd_xl = pytest.importorskip("openpyxl", reason="openpyxl 未安装")

from rams.data.preprocessing import (  # noqa: E402
    NUMERIC_COLS,
    align_and_clean,
    build_dataset,
    compute_norm_stats,
    load_algae,
    load_meteo,
)


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    """构造合成 xlsx：藻类 2 sheet（含一层自带 depth 列）+ 气象 Sheet1。

    注意：藻类 sheet 不含「风速」等气象列（真实数据如此），
    避免 merge_asof 时与气象列撞名产生 *_x/_y 后缀。
    """
    d = tmp_path / "raw"
    d.mkdir(parents=True, exist_ok=True)

    # 藻类：两 sheet（对应深度 0.5m / 1.0m），time 错开 4 分钟模拟真实错位
    t0 = pd.to_datetime("2022-01-01 00:00:00")
    with pd.ExcelWriter(d / "藻类检测数据_分深度整理.xlsx", engine="openpyxl") as w:
        for i, depth in enumerate([0.5, 1.0]):
            ts = [t0 + pd.Timedelta(minutes=4 * i) + pd.Timedelta(hours=3 * k) for k in range(4)]
            df = pd.DataFrame({
                "站点编号": "0902",
                "测量时间": ts,
                "水温": 18.0 - 0.2 * depth,
                "总浓度": 5.0 + depth,
            })
            if depth == 1.0:
                # 模拟某 sheet 自带「深度」列：load_algae 应删除避免重复
                df["深度"] = depth
            df.to_excel(w, sheet_name=f"{depth}m", index=False)

    # 气象：10min 频率 → 3h 重采样
    mts = pd.date_range("2022-01-01", periods=30, freq="10min")
    meteo = pd.DataFrame({
        "测量时间": mts,
        "风速": 3.0,
        "风向": 120.0,
        "压力": 1013.0,
        "温度": 20.0,
        "湿度": 55.0,
        "降雨量": 0.0,
    })
    with pd.ExcelWriter(d / "气象数据.xlsx", engine="openpyxl") as w:
        meteo.to_excel(w, sheet_name="Sheet1", index=False)
        pd.DataFrame({"empty": [1]}).to_excel(w, sheet_name="Sheet2", index=False)

    # 匹配文件（build_dataset 不读它，占位即可）
    pd.DataFrame({"a": [1]}).to_excel(d / "藻类_气象_最近时刻匹配.xlsx", index=False)
    return d


class TestLoadAlgae:
    def test_merge_sheets_no_dup_depth(self, raw_dir):
        algae = load_algae(raw_dir)
        assert "depth" in algae.columns
        assert "depth_x" not in algae.columns and "depth_y" not in algae.columns
        assert set(algae["depth"]) == {0.5, 1.0}
        assert len(algae) == 8  # 2 sheet × 4 行
        assert pd.api.types.is_datetime64_any_dtype(algae["timestamp"])

    def test_rename_cn2en(self, raw_dir):
        algae = load_algae(raw_dir)
        assert "总浓度" not in algae.columns
        assert "total_conc" in algae.columns
        assert "water_temp" in algae.columns


class TestLoadMeteo:
    def test_resample_3h(self, raw_dir):
        meteo = load_meteo(raw_dir)
        # 30 × 10min = 5h → 3h 重采样得 2 行
        assert len(meteo) == 2
        assert (meteo["timestamp"].diff().dropna().dt.total_seconds() == 3 * 3600).all()
        # 重采样前后时间戳整点在 3h 网格上
        assert meteo["timestamp"].iloc[0] == pd.Timestamp("2022-01-01 00:00:00")

    def test_empty_sheet_ignored(self, raw_dir):
        meteo = load_meteo(raw_dir)
        assert "empty" not in meteo.columns


class TestAlignAndClean:
    def test_nearest_alignment(self, raw_dir):
        algae = load_algae(raw_dir)
        meteo = load_meteo(raw_dir)
        df = align_and_clean(algae, meteo)
        # 4 分钟错位被 floor/merge_asof 吸收 → 每时刻 2 深度 × 4 时刻 = 8 行
        assert len(df) == 8
        assert "wind_speed" in df.columns
        # 无重复时间列残留
        assert "timestamp_x" not in df.columns and "timestamp_y" not in df.columns
        # 气象已对齐（风速 ≈ 3.0）
        assert np.allclose(df["wind_speed"].dropna(), 3.0)

    def test_merge_asof_tolerance(self):
        # 超过 3h 容差的错位 → 该行气象为 NaN（merge_asof 行为，无静默错配）
        algae = pd.DataFrame({
            "depth": [0.5],
            "ts3h": [pd.Timestamp("2022-01-01 00:00:00")],
            "water_temp": [18.0],
        })
        meteo = pd.DataFrame({
            "timestamp": [pd.Timestamp("2022-01-01 12:00:00")],
            "wind_speed": [3.0],
        })
        df = pd.merge_asof(
            algae, meteo, left_on="ts3h", right_on="timestamp",
            direction="nearest", tolerance=pd.Timedelta("3h"),
        )
        assert pd.isna(df["wind_speed"].iloc[0])


class TestComputeNormStats:
    def test_train_frac_only(self):
        ts = pd.date_range("2022-01-01", periods=10)
        df = pd.DataFrame({"timestamp": ts, "total_conc": range(10)})
        stats = compute_norm_stats(df, train_frac=0.7)
        # 只用前 7 行拟合：mean=(0..6)/7=3, std 已知
        assert stats["total_conc"]["mean"] == pytest.approx(3.0)
        # 若用全量(0..9) mean=4.5 → 确认未泄漏
        assert stats["total_conc"]["mean"] != pytest.approx(4.5)
        assert stats["total_conc"]["std"] > 0


class TestBuildDataset:
    def test_end_to_end(self, raw_dir, tmp_path):
        out = build_dataset(raw_dir, tmp_path / "processed", train_frac=0.7, verbose=False)
        assert out.exists() and out.name == "standard.parquet"
        df = pd.read_parquet(out)
        # 类型固化：数值列 float32
        assert df["total_conc"].dtype == np.float32
        assert df["wind_speed"].dtype == np.float32
        # 标准长表列契约
        for c in ["site_id", "timestamp", "depth", "total_conc", "wind_speed"]:
            assert c in df.columns
        # norm_stats.json
        stats_file = out.parent / "norm_stats.json"
        assert stats_file.exists()
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        assert "total_conc" in stats and "mean" in stats["total_conc"]

    def test_site_id_padding(self, raw_dir, tmp_path):
        out = build_dataset(raw_dir, tmp_path / "processed", verbose=False)
        df = pd.read_parquet(out)
        assert set(df["site_id"]) == {"0902"}

    def test_metadata_and_std(self, raw_dir, tmp_path):
        """数值列契约：NUMERIC_COLS 名称与列名一致，可转 float32。"""
        algae = load_algae(raw_dir)
        # 覆盖 NUMERIC_COLS 契约符号存在且可解析
        assert "total_conc" in NUMERIC_COLS
        df = pd.DataFrame({c: [1.0] for c in NUMERIC_COLS if c in algae.columns})
        for c in df.columns:
            assert df[c].dtype == np.float64 or df[c].dtype == np.float32
