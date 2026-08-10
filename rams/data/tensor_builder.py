"""RAMS 0.2.0 数据管线 —— 日级张量构建 + 藻华状态标签（mdl-data-scale）

模块目标（frozen design `modules/mdl-data-scale/module_design.yaml`）：
  - 日级张量（T=30, H=7）构建通过
  - 藻华状态标签（N 定义）生成
  - M3 选层输入（5 层）集成
  - 冒烟通过

设计来源（探索报告，`docs/` 与 `exp/model_enhancement/`）：
  - 日级尺度（`l_daily_scale/results.md`）：`resample('1D').mean()` 均值聚合；
    T=30 天回看；H=7 天视界；目标 = 表层 conc_0.5 日平均浓度；M1 增量口径
    Δ = conc_{t+h} - conc_t（abs_delta，B1）。
  - 藻华状态（`n_bloom_identify/results.md`）：
    顶层带(0.5-3.0m)中位数 > 带 p90，且 0.5-5.0m 带 ≥3 层 > 各自 p90，
    连续 ≥2 天；对表层单层 dropout 稳健。
  - M3 选层（`docs/m3_sensor_placement.md`）：M3 只做部署建议，输入保持 20 层；
    `M3_RECOMMENDED_DEPTHS` 供部署对照/瘦身实验使用。

数据保密红线：本模块**只读** `data/`（绝不写/改），所有对外输出为
形状与聚合统计量，不打印任何原始数据行。

兼容性：`TensorConfig` / `TensorBuilder`（3h 网格）为 0.1.0 原管线，
供既有探索脚本（run_l / run_ml_baselines / run_f / run_a 等）与旧冒烟测试使用；
日级新管线为 `DailyConfig` / `DailyTensorBuilder` / `BloomLabeler`。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# ---- 常量 ----
DEPTHS = [0.5 + 0.5 * i for i in range(20)]  # 0.5 ~ 10.0m，20 层
GRID = "3h"  # 3h 原网格（时间对齐基准）
DAILY_GRID = "1D"  # 日级聚合网格（pandas 3.0：'1D'，非 'D'）

METEO_COLS = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
CONC_COLS = [f"conc_{d}" for d in DEPTHS]  # 20 层浓度（预测目标/输入）
TEMP_COLS = [f"temp_{d}" for d in DEPTHS]  # 20 层水温（输入特征）
STRAT_COLS = ["delta_T", "thermo_grad"]  # 分层指标（M2 输入/标签来源）

# 藻华定义（N 探索，见 n_bloom_identify）：
TOP_BAND_DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]  # 顶层带 0.5-3.0m（6 层）
LINK_BAND_DEPTHS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # 0.5-5.0m（11 层）
BLOOM_MIN_DAYS = 2  # 连续藻华状态 ≥ 2 天
BLOOM_GAP_DAYS = 1  # 相邻段间隔 ≤ 1 天合并（日级"容忍昼夜回落"）

# M3 选层（跨 seed 共识子集，docs/m3_sensor_placement.md §4）：
#   近表层 1 层（1.5m）+ 中层 1 层（5.0m）+ 深层 3 层（8.5/9.5/10.0m）
M3_RECOMMENDED_DEPTHS = (1.5, 5.0, 8.5, 9.5, 10.0)
TARGET_DEPTH = 0.5  # 目标 = 表层浓度（M1/M4 预警口径）


# =====================================================================
# 0.1.0 兼容管线（3h 网格）
# =====================================================================
@dataclass
class TensorConfig:
    """张量构建配置（3h 原网格，0.1.0 兼容）。"""

    T: int = 24  # 回看窗口（3h 步，24 = 3 天）
    H: int = 8  # 预测未来步数（8 = 24h）
    use_meteo: bool = True
    use_strat_feat: bool = False  # 是否把分层指标当输入特征
    strat_as_task: bool = True  # 是否输出 M2 分层标签（多任务）
    warn_as_task: bool = False  # 是否输出 M4 预警等级标签（多任务）
    warn_levels: int = 4  # 预警等级数（安全/注意/警告/危险）
    train_frac: float = 0.7
    val_frac: float = 0.15
    seed: int = 0

    # 归一化参数（训练段拟合，防泄漏）
    x_stats: dict = field(default_factory=dict)
    y_stats: dict = field(default_factory=dict)


class TensorBuilder:
    """把 standard.parquet 长表构建为 3h 时序张量（0.1.0 兼容，只读数据）。"""

    def __init__(self, config: TensorConfig | None = None):
        self.cfg = config or TensorConfig()

    def _load_wide(self, parquet_path: Path) -> pd.DataFrame:
        """读长表 → 按 3h 网格透视成宽表（每时刻一行）。

        Returns:
            pd.DataFrame: 宽表，列 = temp_* (20) + conc_* (20) + delta_T + thermo_grad + meteo (6)
        """
        df = pd.read_parquet(parquet_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # 关键：floor 到 3h 网格，统一各深度层错位的时间戳
        df["ts3h"] = df["timestamp"].dt.floor(GRID)
        # 同一 (ts3h, depth) 聚合均值
        agg = df.groupby(["ts3h", "depth"]).agg(
            water_temp=("water_temp", "mean"),
            total_conc=("total_conc", "mean"),
        ).reset_index()

        pivot_temp = agg.pivot_table(index="ts3h", columns="depth", values="water_temp")
        pivot_conc = agg.pivot_table(index="ts3h", columns="depth", values="total_conc")
        pivot_temp.columns = [float(c) for c in pivot_temp.columns]
        pivot_conc.columns = [float(c) for c in pivot_conc.columns]

        wide = pivot_temp.add_prefix("temp_").join(pivot_conc.add_prefix("conc_"))

        # 气象（3h 网格）
        meteo = df.drop_duplicates("ts3h")[["ts3h"] + METEO_COLS].set_index("ts3h")
        wide = wide.join(meteo)

        # 分层指标（从温度剖面）
        ds_sorted = sorted(pivot_temp.columns)
        surface, bottom = pivot_temp[ds_sorted[0]].values, pivot_temp[ds_sorted[-1]].values
        delta_T = surface - bottom
        grads = np.zeros(len(pivot_temp))
        for i in range(len(ds_sorted) - 1):
            g = (
                pivot_temp[ds_sorted[i + 1]].values - pivot_temp[ds_sorted[i]].values
            ) / (ds_sorted[i + 1] - ds_sorted[i])
            grads = np.maximum(grads, g)
        wide["delta_T"] = delta_T
        wide["thermo_grad"] = grads

        return wide.dropna()

    def _fit_stats(self, wide: pd.DataFrame) -> None:
        """用训练段拟合归一化参数。"""
        wide_sorted = wide.sort_index()
        n_tr = int(len(wide_sorted) * self.cfg.train_frac)
        tr = wide_sorted.iloc[:n_tr]
        conc_cols = [c for c in wide.columns if c.startswith("conc_")]
        x_cols = [c for c in wide.columns if c not in conc_cols]
        self.cfg.x_stats = {c: (float(tr[c].mean()), float(tr[c].std()) + 1e-8) for c in x_cols}
        self.cfg.y_stats = {c: (float(tr[c].mean()), float(tr[c].std()) + 1e-8) for c in conc_cols}

    def _normalize(self, wide: pd.DataFrame) -> pd.DataFrame:
        """归一化（用训练段参数）。"""
        out = wide.copy()
        for c, (mu, sd) in self.cfg.x_stats.items():
            if c in out.columns:
                out[c] = (out[c] - mu) / sd
        for c, (mu, sd) in self.cfg.y_stats.items():
            if c in out.columns:
                out[c] = (out[c] - mu) / sd
        return out

    def _make_windows(self, wide: pd.DataFrame):
        """滑动窗口构建 (X_seq, y_seq, strat_label)。"""
        cfg = self.cfg
        T, H = cfg.T, cfg.H
        temp_cols = [c for c in wide.columns if c.startswith("temp_")]
        conc_cols = [c for c in wide.columns if c.startswith("conc_")]
        if not temp_cols or not conc_cols:
            raise ValueError("宽表缺少 temp_*/conc_* 列，请检查数据")
        feat_cols = temp_cols + (METEO_COLS if cfg.use_meteo else [])
        if cfg.use_strat_feat:
            feat_cols += STRAT_COLS
        feat_cols = [c for c in feat_cols if c in wide.columns]
        X_all = wide[feat_cols].values.astype(np.float32)
        y_all = wide[conc_cols[0]].values.astype(np.float32)  # 简化：预测表层浓度

        n = len(wide)
        Xw, yw = [], []
        for i in range(n - T - H):
            Xw.append(X_all[i:i + T])
            yw.append(y_all[i + T:i + T + H])
        X = np.stack(Xw).astype(np.float32)
        y = np.stack(yw).astype(np.float32)

        # 分层标签（多任务）：窗口末时刻的分层状态
        if cfg.strat_as_task:
            delta = wide["delta_T"].values
            n_tr = int(n * cfg.train_frac)
            threshold = np.median(delta[:n_tr])
            strat = (delta > threshold).astype(np.int64)
            strat_w = np.array([strat[i + T - 1] for i in range(n - T - H)])
        else:
            strat_w = None

        # 预警等级标签（M4）：基于未来 H 步浓度峰值（窗口内 y 最大值），
        # 用训练段 y 的分位数定等级阈值（防泄漏）
        if cfg.warn_as_task:
            n_tr = int(len(y) * cfg.train_frac)
            train_y = y[:n_tr]
            warn_val = y.max(axis=1)
            train_warn = train_y.max(axis=1)
            qs = np.quantile(train_warn, [0.75, 0.90, 0.97])
            warn = np.zeros(len(y), dtype=np.int64)
            for i, v in enumerate(warn_val):
                warn[i] = np.searchsorted(qs, v)
            warn_w = warn
        else:
            warn_w = None
        return X, y, strat_w, warn_w

    def build(self, parquet_path: str | Path) -> dict:
        """构建并切分数据集（3h 网格）。

        Returns:
            dict: {"train"/"val"/"test": (X, y, strat, warn), "feat_dim", "y_sd"}
        """
        wide = self._load_wide(Path(parquet_path))
        self._fit_stats(wide)
        wide = self._normalize(wide)
        X, y, strat, warn = self._make_windows(wide)

        n = len(X)
        n_tr, n_va = int(n * self.cfg.train_frac), int(n * self.cfg.val_frac)
        idx_tr, idx_va, idx_te = range(n_tr), range(n_tr, n_tr + n_va), range(n_tr + n_va, n)
        first_conc = next(iter(self.cfg.y_stats))
        y_sd = self.cfg.y_stats[first_conc][1]

        return {
            "train": (X[idx_tr], y[idx_tr],
                      strat[idx_tr] if strat is not None else None,
                      warn[idx_tr] if warn is not None else None),
            "val": (X[idx_va], y[idx_va],
                    strat[idx_va] if strat is not None else None,
                    warn[idx_va] if warn is not None else None),
            "test": (X[idx_te], y[idx_te],
                     strat[idx_te] if strat is not None else None,
                     warn[idx_te] if warn is not None else None),
            "feat_dim": X.shape[2],
            "y_sd": y_sd,
        }


# =====================================================================
# 0.2.0 日级管线（mdl-data-scale）
# =====================================================================
@dataclass
class DailyConfig:
    """日级张量构建配置（RAMS 0.2.0）。

    Attributes:
        T: 回看天数（30 天）。
        H: 预测天数（7 天视界）。
        grid: 日级聚合网格，恒为 "1D"。
        use_meteo: 是否把 6 项气象纳入特征通道。
        use_strat_feat: 是否把 delta_T / thermo_grad 纳入特征通道。
        delta_target: M1 增量口径——目标是 Δ=conc_{t+h}-conc_t（可选）。
        target_depth: 预测目标所在深度（表层 0.5m）。
        m3_depths: M3 选层输入子集；空元组 = 20 层全输入（正式口径）。
        fit_frac: 无显式训练切点时用于拟合归一化/阈值的前段比例。
    """

    T: int = 30
    H: int = 7
    grid: str = DAILY_GRID
    use_meteo: bool = True
    use_strat_feat: bool = True
    delta_target: bool = False
    target_depth: float = TARGET_DEPTH
    m3_depths: tuple = ()  # () = 全 20 层
    fit_frac: float = 0.7

    # 归一化/阈值参数（训练段拟合，防泄漏）
    x_stats: dict = field(default_factory=dict)
    y_stats: dict = field(default_factory=dict)
    delta_scale: float = 1.0
    bloom_band_p90: float = float("nan")
    bloom_layer_p90: dict = field(default_factory=dict)

    @property
    def layer_depths(self) -> list:
        """输入特征所用的深度层（M3 子集或全 20 层）。"""
        return list(self.m3_depths) if self.m3_depths else list(DEPTHS)


@dataclass
class DailyDataset:
    """日级窗口数据集（单窗口）。

    Attributes:
        X: (B, T, D, C) 剖面张量（D=层，C=每层通道：0=浓度, 1=水温），已归一化。
        X_flat: (B, T, F) 展平特征（RamsNet/GRU 兼容），已归一化。
        y_abs: (B, H) 未来 H 天表层浓度（conc 单位，原始口径）。
        y_delta: (B, H) 未来 H 天增量 Δ=conc_{t+h}-conc_t（conc 单位）。
        cur: (B,) 预测日表层浓度（窗口末天）。
        dates: (B,) 预测日日期（窗口末天）。
        bloom: (B,) 预测日的藻华状态标签（N 定义，0/1）。
        strat: (B,) M2 分层标签（窗口末天 delta_T > 训练段中位数）。
        feature_names: (F,) 展平特征列名。
        channel_names: (C,) 剖面张量通道名。
        n_train: 预测末端落在训练段内的样本数（滚动窗口切分用）。
    """

    X: np.ndarray
    X_flat: np.ndarray
    y_abs: np.ndarray
    y_delta: np.ndarray
    cur: np.ndarray
    dates: np.ndarray
    bloom: np.ndarray
    strat: np.ndarray
    feature_names: list
    channel_names: list
    n_train: int


class DailyTensorBuilder:
    """standard.parquet → 日级张量（T=30, H=7）+ 藻华标签（只读数据）。"""

    def __init__(self, config: DailyConfig | None = None):
        self.cfg = config or DailyConfig()

    # ---------- 读取 / 聚合 ----------
    def load_daily_wide(self, parquet_path: str | Path) -> pd.DataFrame:
        """读长表 → 3h 宽表 → 日级宽表（`resample('1D').mean().dropna()`）。

        Returns:
            pd.DataFrame: 索引 = 日历日；列 = temp_* + conc_* + delta_T + thermo_grad + meteo(6)。
        """
        wide3h = TensorBuilder()._load_wide(Path(parquet_path)).sort_index()
        return wide3h.resample(DAILY_GRID).mean().dropna()

    # ---------- 归一化（训练段拟合） ----------
    def fit_stats(self, daily: pd.DataFrame, tr_ts: pd.Timestamp | None = None) -> None:
        """用训练段（tr_ts 之前 / fit_frac 前段）拟合归一化参数。

        Args:
            daily: 日级宽表。
            tr_ts: 训练段截止（不含）；None 时用前 fit_frac 比例。
        """
        daily = daily.sort_index()
        tr = daily[daily.index < tr_ts] if tr_ts is not None else daily.iloc[: int(len(daily) * self.cfg.fit_frac)]
        feat_cols = self._feature_cols(daily)
        conc_cols = self._conc_cols(daily)
        self.cfg.x_stats = {c: (float(tr[c].mean()), float(tr[c].std()) + 1e-8) for c in feat_cols}
        self.cfg.y_stats = {c: (float(tr[c].mean()), float(tr[c].std()) + 1e-8) for c in conc_cols}
        # 剖面张量 (B,T,D,C) 的 conc 通道需逐层归一化 → 全 conc_* 层也纳入 x_stats
        for c in conc_cols:
            if c not in self.cfg.x_stats:
                self.cfg.x_stats[c] = (float(tr[c].mean()), float(tr[c].std()) + 1e-8)
        # delta_scale 由 build() 在窗口构建后用训练段 Δ=conc_{t+h}-conc_t 的 std 填充
        # （与探索 run_l.py 的 scale=std(raw[:n_tr]) 一致）；此处保留全量近似兜底。
        target = self._target_col(daily)
        idx = daily.index.values
        lo = 0
        if tr_ts is not None:
            pos = int(np.searchsorted(idx, np.datetime64(tr_ts), side="left"))
            lo = max(pos - 1, 0)
        else:
            lo = max(int(len(daily) * self.cfg.fit_frac) - 1, 0)
        delta_tr = np.diff(daily[target].values[:lo + 1])
        self.cfg.delta_scale = float(np.std(delta_tr)) + 1e-8

    # ---------- 列 / 特征选择 ----------
    def _feature_cols(self, daily: pd.DataFrame) -> list:
        """输入特征列：M3 选层 temp_* + (可选 meteo/strat) + 目标层浓度。"""
        depths = self.cfg.layer_depths
        cols = [f"temp_{d}" for d in depths if f"temp_{d}" in daily.columns]
        if self.cfg.use_meteo:
            cols += [c for c in METEO_COLS if c in daily.columns]
        if self.cfg.use_strat_feat:
            cols += [c for c in STRAT_COLS if c in daily.columns]
        # 目标层浓度入特征（自回归主导，M5 实证），保持与探索协议一致
        tgt = self._target_col(daily)
        if tgt not in cols:
            cols += [tgt]
        return cols

    def _conc_cols(self, daily: pd.DataFrame) -> list:
        return [c for c in daily.columns if c.startswith("conc_")]

    def _target_col(self, daily: pd.DataFrame) -> str:
        tgt = f"conc_{self.cfg.target_depth}"
        if tgt not in daily.columns:
            raise ValueError(f"目标深度 {self.cfg.target_depth}m 列 {tgt} 不存在")
        return tgt

    # ---------- 窗口构建 ----------
    def make_windows(self, daily: pd.DataFrame, bloom_signal: np.ndarray | pd.Series | None = None) -> DailyDataset:
        """从（原始 conc 单位）日级宽表滑动窗口构建张量。

        Args:
            daily: 日级宽表（**原始 conc 单位**；特征在内部按训练段参数归一化，
                目标保持原始单位——与探索协议 run_l.py 一致）。
            bloom_signal: 全长的逐日藻华状态（与 daily 行数对齐，原始 conc 单位
                下生成）；None 时退化为对 daily 就地拟合生成。

        Returns:
            DailyDataset: 见类 docstring。B = n - T - H。
        """
        cfg = self.cfg
        T, H = cfg.T, cfg.H
        daily = daily.sort_index()
        feat_cols = self._feature_cols(daily)

        n = len(daily)
        n_w = n - T - H
        if n_w <= 0:
            raise ValueError(f"样本不足：n={n} ≤ T+H={T + H}")

        # 有训练段归一化参数时归一化特征（目标保持原始单位）
        has_stats = bool(self.cfg.x_stats)

        def _z(col, key):
            if has_stats and key in self.cfg.x_stats:
                mu, sd = self.cfg.x_stats[key]
                return (col - mu) / sd
            return col

        # --- 剖面张量 (B, T, D, C)：C=[conc, temp] 逐层通道，特征归一化 ---
        layer_depths = [d for d in cfg.layer_depths if f"conc_{d}" in daily.columns]
        if not layer_depths:
            raise ValueError("日级宽表缺少 conc_* 深度列")
        conc_ch = np.stack(
            [
                _z(daily[f"conc_{d}"].values, f"conc_{d}") for d in layer_depths
            ],
            axis=1,
        ).astype(np.float32)
        temp_ch = np.stack(
            [_z(daily[f"temp_{d}"].values, f"temp_{d}") for d in layer_depths],
            axis=1,
        ).astype(np.float32)
        d_arr = np.stack([conc_ch, temp_ch], axis=2)  # (n, D, C)
        X = np.stack([d_arr[i:i + T] for i in range(n_w)]).astype(np.float32)  # (B,T,D,C)

        # --- 展平特征 (B, T, F)，归一化 ---
        f_arr = np.stack([_z(daily[c].values, c) for c in feat_cols], axis=1).astype(np.float32)
        X_flat = np.stack([f_arr[i:i + T] for i in range(n_w)]).astype(np.float32)  # (B,T,F)

        # --- 目标（原始 conc 单位） ---
        target = self._target_col(daily)
        y_raw = daily[target].values.astype(np.float64)
        y_abs = np.stack([y_raw[i + T:i + T + H] for i in range(n_w)]).astype(np.float64)  # (B,H)
        cur = np.array([y_raw[i + T - 1] for i in range(n_w)]).astype(np.float64)  # (B,)
        y_delta = y_abs - cur[:, None]  # Δ = conc_{t+h} - conc_t

        # --- 预测日日期 ---
        dates = np.array([daily.index[i + T - 1] for i in range(n_w)])  # (B,)

        # --- 藻华标签：预测日是否处于藻华状态（N 定义，日级） ---
        if bloom_signal is not None:
            full = np.asarray(bloom_signal).astype(np.int64)
            if len(full) != n:
                raise ValueError(f"bloom_signal 长度 {len(full)} ≠ 日级表行数 {n}")
            bloom = np.array([full[i + T - 1] for i in range(n_w)], dtype=np.int64)
        else:
            full = BloomLabeler(config=cfg).predict(daily).astype(np.int64)
            bloom = np.array([full[i + T - 1] for i in range(n_w)], dtype=np.int64)

        # --- M2 分层标签（窗口末天） ---
        if "delta_T" in daily.columns:
            delta = daily["delta_T"].values
            n_tr_rows = int(len(daily) * cfg.fit_frac)
            thr = float(np.median(delta[:n_tr_rows]))
            strat = np.array([int(delta[i + T - 1] > thr) for i in range(n_w)], dtype=np.int64)
        else:
            strat = np.zeros(n_w, dtype=np.int64)

        channel_names = ["conc", "temp"]
        return DailyDataset(
            X=X,
            X_flat=X_flat,
            y_abs=y_abs,
            y_delta=y_delta,
            cur=cur,
            dates=dates,
            bloom=bloom,
            strat=strat,
            feature_names=feat_cols,
            channel_names=channel_names,
            n_train=0,  # 调用方按窗口切点回填
        )

    def build(self, parquet_path: str | Path,
              start_ts: pd.Timestamp | None = None,
              tr_ts: pd.Timestamp | None = None,
              end_ts: pd.Timestamp | None = None) -> DailyDataset:
        """完整管线：读 parquet → 日级聚合 → 训练段拟合归一化 → 滚动窗口构建。

        Args:
            parquet_path: standard.parquet 路径。
            start_ts: 窗口起始（含）；None = 数据最早日。
            tr_ts: 训练段截止（不含）；None = 前 fit_frac 比例。
            end_ts: 窗口结束（不含）；None = 数据最晚日。

        Returns:
            DailyDataset: 张量 + 标签；n_train 按"预测末端 < tr_ts"回填。
        """
        daily = self.load_daily_wide(parquet_path)
        if start_ts is not None:
            daily = daily[daily.index >= start_ts]
        if end_ts is not None:
            daily = daily[daily.index < end_ts]
        self.fit_stats(daily, tr_ts)
        # 藻华信号在原始 conc 单位上按训练段阈值生成（N 定义），
        # 再在 make_windows 内取窗口末天标签（特征归一化、目标保持原始单位）。
        lab = BloomLabeler(config=self.cfg)
        lab.fit(daily, tr_ts=tr_ts)
        bloom_full = lab.predict(daily)
        ds = self.make_windows(daily, bloom_signal=bloom_full)
        # 训练样本数：预测末端索引 i+T+H-1 < tr 行位置（与探索协议一致）。
        # 注意：只要 tr_ts 给定，就按行计数（daily.index < tr_ts），
        # 不要求 tr_ts 恰好出现在日级索引里（数据缺口日缺失时按行计数仍正确）。
        if tr_ts is not None:
            n_tr_rows = int((daily.index < tr_ts).sum())
            ds.n_train = max(0, n_tr_rows - self.cfg.T - self.cfg.H + 1)
        else:
            ds.n_train = int(len(ds.X) * self.cfg.fit_frac)
        # delta_scale：训练段窗口 Δ=conc_{t+h}-conc_t 的 std（与探索 run_l.py 口径一致）
        if ds.n_train > 0:
            self.cfg.delta_scale = float(np.std(ds.y_delta[: ds.n_train])) + 1e-8
        return ds


# =====================================================================
# 藻华状态标签（N 定义）
# =====================================================================
class BloomLabeler:
    """藻华状态/事件标签（N 探索定义，日级网格）。

    规则（`n_bloom_identify/results.md` §4）：
      top_band = 顶层带(0.5-3.0m) 逐日中位数
      藻华状态(日) = (top_band > 带 p90) 且 (0.5-5.0m 带 ≥3 层 > 各自 p90)
      藻华事件(时段) = 连续藻华状态 ≥ 2 天，相邻段间隔 ≤ 1 天合并
    对表层单层 dropout 稳健（顶层带中位数不受单层尖峰/掉零影响）。
    """

    def __init__(self, config: DailyConfig | None = None):
        self.cfg = config or DailyConfig()

    def _band_cols(self, daily: pd.DataFrame) -> list:
        return [f"conc_{d}" for d in TOP_BAND_DEPTHS if f"conc_{d}" in daily.columns]

    def _link_cols(self, daily: pd.DataFrame) -> list:
        return [f"conc_{d}" for d in LINK_BAND_DEPTHS if f"conc_{d}" in daily.columns]

    def fit(self, daily: pd.DataFrame, fit_frac: float | None = None, tr_ts: pd.Timestamp | None = None) -> None:
        """用训练段拟合带 p90 与逐层 p90（防泄漏）。

        Args:
            daily: 日级宽表（原始 conc 单位）。
            fit_frac: 训练段比例；None 用 config.fit_frac。
            tr_ts: 训练段截止（不含）；优先于 fit_frac 使用。
        """
        daily = daily.sort_index()
        if tr_ts is not None:
            tr = daily[daily.index < tr_ts]
        else:
            f = fit_frac if fit_frac is not None else self.cfg.fit_frac
            tr = daily.iloc[: int(len(daily) * f)]
        band = tr[self._band_cols(daily)].median(axis=1)
        self.cfg.bloom_band_p90 = float(band.quantile(0.90))
        self.cfg.bloom_layer_p90 = {
            c: float(tr[c].quantile(0.90)) for c in self._link_cols(daily)
        }

    def predict(self, daily: pd.DataFrame) -> np.ndarray:
        """逐日藻华状态（0/1），与 daily 行数一致。

        Args:
            daily: 日级宽表（原始 conc 单位）。

        Returns:
            np.ndarray[int64]: 每行藻华状态。
        """
        daily = daily.sort_index()
        # 未拟合 → 用全量拟合（冒烟/整集标签）
        if not self.cfg.bloom_layer_p90:
            self.fit(daily)
        band = daily[self._band_cols(daily)].median(axis=1)
        band_signal = band > self.cfg.bloom_band_p90
        layer_p90s = self.cfg.bloom_layer_p90
        link_signal = (
            pd.concat([(daily[c] > th).astype(int) for c, th in layer_p90s.items() if c in daily.columns], axis=1)
            .sum(axis=1)
            >= 3
        )
        signal = (band_signal & link_signal).astype(int)
        return signal.values

    def events(self, signal: np.ndarray | pd.Series, dates: pd.DatetimeIndex | None = None) -> list:
        """从逐日信号提取藻华事件（连续 ≥2 天，间隔 ≤1 天合并）。

        Args:
            signal: 逐日藻华状态（0/1）。
            dates: 对应日期；None 时用行号。

        Returns:
            list[dict]: 每事件 {"start", "end", "duration_days", "n_days"}。
        """
        s = np.asarray(signal).astype(int)
        segs: list[list[int]] = []
        i, n = 0, len(s)
        while i < n:
            if s[i]:
                j = i
                while j < n and s[j]:
                    j += 1
                segs.append([i, j])
                i = j
            else:
                i += 1
        if not segs:
            return []
        merged = [segs[0]]
        for seg in segs[1:]:
            if seg[0] - merged[-1][1] <= BLOOM_GAP_DAYS:
                merged[-1][1] = seg[1]
            else:
                merged.append(seg)
        out = []
        for a, b in merged:
            if b - a < BLOOM_MIN_DAYS:
                continue
            out.append({
                "start": str(dates[a]) if dates is not None else int(a),
                "end": str(dates[b - 1]) if dates is not None else int(b - 1),
                "duration_days": float(b - a),
                "n_days": int(b - a),
            })
        return out


# =====================================================================
# 滚动窗口锚点（T4 协议）
# =====================================================================
def make_rolling_anchors(d0: pd.Timestamp, train_days: int = 730,
                         test_days: int = 90, stride_days: int = 45,
                         n_windows: int = 17) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """按日历日生成滚动窗口锚点（T4 评价协议：训练 730d / 测试 90d / 步长 45d）。

    Args:
        d0: 数据起始日。
        train_days: 训练段天数（730）。
        test_days: 测试段天数（90）。
        stride_days: 窗口步长天数（45）。
        n_windows: 窗口数（17）。

    Returns:
        list[(start_ts, tr_ts, end_ts)]：每窗口起始 / 训练截止(不含) / 测试结束(不含)。
    """
    return [
        (
            d0 + pd.Timedelta(days=stride_days * wi),
            d0 + pd.Timedelta(days=stride_days * wi + train_days),
            d0 + pd.Timedelta(days=stride_days * wi + train_days + test_days),
        )
        for wi in range(n_windows)
    ]


def build_daily_dataset(parquet_path: str | Path, config: DailyConfig | None = None) -> DailyDataset:
    """便捷入口：全量日级数据集（无滚动切分，单次 70% 拟合归一化）。

    用于冒烟/整集标签生成；正式训练用 `DailyTensorBuilder.build(parquet, start, tr, end)`
    逐窗口构建，配合 `make_rolling_anchors`。
    """
    cfg = config or DailyConfig()
    return DailyTensorBuilder(cfg).build(parquet_path)


if __name__ == "__main__":
    import argparse
    import io
    import sys as _sys

    _sys.stdout = io.TextIOWrapper(_sys.stdout.buffer, encoding="utf-8")

    parser = argparse.ArgumentParser(description="RAMS 日级张量构建（冒烟，只输出形状/统计量）")
    parser.add_argument("--parquet", default="data/processed/standard.parquet")
    parser.add_argument("--T", type=int, default=30)
    parser.add_argument("--H", type=int, default=7)
    parser.add_argument("--m3", action="store_true", help="使用 M3 选层子集（5 层）")
    args = parser.parse_args()

    cfg = DailyConfig(T=args.T, H=args.H)
    if args.m3:
        cfg.m3_depths = M3_RECOMMENDED_DEPTHS
    builder = DailyTensorBuilder(cfg)
    ds = builder.build(args.parquet)
    print(f"X: {ds.X.shape}   (B={ds.X.shape[0]}, T={ds.X.shape[1]}, D={ds.X.shape[2]}, C={ds.X.shape[3]})")
    print(f"X_flat: {ds.X_flat.shape}   y_abs: {ds.y_abs.shape}   y_delta: {ds.y_delta.shape}   cur: {ds.cur.shape}")
    print(f"bloom 标签数(日): {int(ds.bloom.sum())}/{len(ds.bloom)}  "
          f"正例比例: {float(ds.bloom.mean()):.3f}   strat 标签正例: {int(ds.strat.sum())}")
    print(f"feature_dim: {ds.X_flat.shape[2]}   delta_scale: {cfg.delta_scale:.4f}   "
          f"bloom_band_p90: {cfg.bloom_band_p90:.4f}")
    print("冒烟通过（未打印任何原始数据行）")
