# -*- coding: utf-8 -*-
"""T2 站位：Tick Tick Bloom 冠军方案复现（分级预测，LightGBM）

原竞赛：DrivenData × NASA "Tick Tick Bloom: HAB Detection Challenge"
  - 任务：小水体蓝藻严重度 5 级序数分类（1~5）
  - 冠军方案要点（arXiv 2505.03808 + DrivenData 基准博客）：
      * 特征 = 卫星影像颜色统计（Sentinel-2 优先）+ 气温/湿度/降水 + 高程 + 日期 + 地理位置
      * 模型 = 梯度提升树（LightGBM/XGBoost/CatBoost），类别不平衡用 class_weight
      * 评估 = RMSE by region（序数误差），不是 plain accuracy
  - 冠军分数（private）：1st=0.7608, 2nd=0.7616, 5th=0.811（Ouranos）
  - 公开数据要自己从 API 取（Planetary Computer / GEE / NOAA）

本文件：把"LightGBM 分级预测"方法移植到 RAMS 自家数据，做 M4 预警分级复现。
  - 目标：M4 四级预警（安全/注意/警告/危险），与 RamsNet M4 同标签协议
  - 特征：滞后浓度（回看窗口浓度统计）+ 气象（同 RAMS 输入特征）
  - 评估：整体准确率 + 序数 RMSE（挑战口径）+ 加权 recall
  - 输出只含统计量，不打印任何原始数据行

用法：python scripts/explore/t2_ticktick_repro.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rams.data.tensor_builder import TensorBuilder, TensorConfig  # noqa: E402

T, H = 24, 8
PARQUET = Path("data/processed/standard.parquet")
N_LEVELS = 4          # M4 预警等级数（安全/注意/警告/危险）
SEED = 0


def ordinal_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """序数 RMSE（挑战评估口径：等级差平方根均值）。"""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def main() -> None:
    np.random.seed(SEED)
    print(f"[1] 读标准数据集构建宽表: {PARQUET}", flush=True)
    builder = TensorBuilder(TensorConfig(T=T, H=H))
    wide = builder._load_wide(PARQUET)
    target = "conc_0.5"
    conc = wide[target].values.astype(np.float64)

    # 特征：滞后浓度统计（回看窗口内 min/max/mean/std/末值）+ 6 气象
    meteo_cols = ["wind_speed", "wind_dir", "pressure", "air_temp", "humidity", "rainfall"]
    # wind_dir 做圆形编码 sin/cos
    meteo = wide[meteo_cols].copy()
    meteo["wind_sin"] = np.sin(np.deg2rad(meteo["wind_dir"]))
    meteo["wind_cos"] = np.cos(np.deg2rad(meteo["wind_dir"]))
    meteo_feat = meteo.drop(columns=["wind_dir"]).values.astype(np.float64)

    n = len(conc)
    X_win, y_win, feats = [], [], []
    for i in range(n - T - H):
        seg = conc[i:i + T]                                   # 回看 24 点
        fut = conc[i + T:i + T + H]                           # 未来 24h
        # 回看窗口统计特征
        feat = [seg[-1], seg.mean(), seg.std(), seg.min(), seg.max(),
                seg[-1] - seg[-8] if T >= 8 else seg[-1] - seg[0]]
        # 未来 24h 峰值 → M4 预警值（与 tensor_builder 相同定义）
        warn_val = fut.max()
        # 当前时刻气象
        feat += list(meteo_feat[i + T - 1])
        X_win.append(feat)
        y_win.append(warn_val)
    Xw = np.array(X_win, dtype=np.float64)
    yw = np.array(y_win, dtype=np.float64)
    N = len(Xw)
    n_tr, n_va = int(N * 0.7), int(N * 0.15)
    print(f"  样本: {N} 个，切分 train {n_tr} / val {n_va} / test {N - n_tr - n_va}", flush=True)

    # M4 等级标签：用训练段未来峰值的分位数定阈值（与 tensor_builder 协议一致）
    qs = np.quantile(yw[:n_tr], [0.75, 0.90, 0.97])
    y_cls = np.zeros(N, dtype=int)
    for i, v in enumerate(yw):
        y_cls[i] = np.searchsorted(qs, v)
    print(f"  等级阈值(训练段峰值 p75/p90/p97): {np.round(qs, 2)}", flush=True)
    print(f"  等级分布: {np.bincount(y_cls[:n_tr], minlength=N_LEVELS).tolist()} (train)", flush=True)

    # ---- LightGBM 分级（冠军方法核心：GBDT + 类别权重）----
    import lightgbm as lgb
    X_tr, y_tr = Xw[:n_tr], y_cls[:n_tr]
    X_va, y_va = Xw[n_tr:n_tr + n_va], y_cls[n_tr:n_tr + n_va]
    X_te, y_te = Xw[n_tr + n_va:], y_cls[n_tr + n_va:]

    counts = np.bincount(y_tr, minlength=N_LEVELS)
    class_weight = (len(y_tr) / (N_LEVELS * (counts + 1))).astype(np.float64)

    lgbm = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31, max_depth=-1,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        class_weight={i: float(w) for i, w in enumerate(class_weight)},
        random_state=SEED, verbosity=-1,
    )
    lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
             callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])

    p_tr = lgbm.predict(X_tr)
    p_te = lgbm.predict(X_te)
    acc_tr = float((p_tr == y_tr).mean())
    acc_te = float((p_te == y_te).mean())
    rmse_te = ordinal_rmse(y_te, p_te)
    # 加权 recall（少数类权重，类数量越多权重越大）——预警场景更关注不漏报
    recall_per = [float((p_te[y_te == c] == c).mean()) if (y_te == c).any() else float("nan")
                  for c in range(N_LEVELS)]
    print("\n===== Tick Tick Bloom 式分级复现（自家数据，M4 预警四级）=====", flush=True)
    print(f"  LightGBM 分级   : 训练 acc={acc_tr:.3f}  测试 acc={acc_te:.3f}", flush=True)
    print(f"  序数 RMSE (挑战口径): {rmse_te:.3f}", flush=True)
    print(f"  各级 recall(安全/注意/警告/危险): {np.round(recall_per, 3).tolist()}", flush=True)

    # ---- 与 RamsNet M4 对比 ----
    print("\n  对照（算力机实测，同标签协议）:", flush=True)
    print("    RamsNet M4 (GRU 多任务, 类别加权): test acc=0.898, 覆盖率 87.4%", flush=True)
    print("    RamsNet M4 (不加权)             : test acc=0.939, 覆盖率 80.1%", flush=True)
    print("    LightGBM 分级（本复现）          : test acc={:.3f}, 序数 RMSE={:.3f}".format(acc_te, rmse_te), flush=True)
    print("\n冒烟通过", flush=True)


if __name__ == "__main__":
    main()
