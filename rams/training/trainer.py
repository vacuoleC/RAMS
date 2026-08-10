"""RAMS 0.2.0 训练编排 —— 分位数损失 + 多任务加权 + 两阶段训练（ts_freeze）

模块目标（frozen design `modules/mdl-model-integrate/module_design.yaml`）：
  - 分位数损失 + 多任务加权（w=1/3/2）
  - 两阶段训练（ts_freeze）：Stage1 单任务 M1 → Stage2 冻结 backbone 微调多头
  - fast_dev_run 冒烟支持

设计来源（探索实证）：
  - 多任务（A）：M1+M2+M4，w=1/3/2；M4 用训练段阈值标签 + 逆频率类别权重（防泄漏）。
  - 两阶段（K）：Stage1 单任务 M1（保精度）→ Stage2 冻结 backbone 只训头（保校准），
    是 4 arm 最优（CRPS 0.808 / 覆盖 0.704，`k_two_stage/results.md`）。
  - 评估（T4）：CRPS（分位数分段线性闭合形式）+ 覆盖率 + p50 RMSE + 相对持久化技能。

数据保密红线：只输出聚合统计量 / 形状，不打印任何原始数据行。

接口：
  - `QuantileLoss(pred, target)`：pred (B, n_q*H) → 标量分位数损失
  - `MultiTaskLoss(m1, m2, y, strat, m4, warn)` → (total, l1, l2, l4)
  - `Trainer.fit(...)` 单阶段多任务 / `fit_m1_only` + `fit_multi` 两阶段
  - `Trainer.evaluate(...)` → dict(RMSE / acc / warn_acc / coverage)
  - `crps_cdf_pline(q, levels, y)` / `crps_quantiles(q10, q50, q90, y)`：CRPS 评估
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from ..models.rams_net import QUANTILE_LEVELS, RamsNet  # 包内相对导入
except ImportError:  # 直接运行本文件（python rams/training/trainer.py）
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rams.models.rams_net import QUANTILE_LEVELS, RamsNet  # noqa: F401

# 多任务权重（探索 A/K 冻结口径 w=1/3/2）
W_M1, W_M2, W_M4 = 1.0, 3.0, 2.0


# =====================================================================
# 损失
# =====================================================================
class QuantileLoss(nn.Module):
    """分位数（pinball）损失，支持任意固定分位数集合（默认 q9）。

    I/O:
        forward(pred, target): pred (B, n_q*H), target (B, H) → 标量
    """

    def __init__(
        self, n_quantiles: int | None = None, levels: tuple[float, ...] | list[float] | None = None
    ):
        super().__init__()
        if levels is not None:
            self.levels = tuple(float(level) for level in levels)
            self.n_quantiles = len(self.levels)
        else:
            self.n_quantiles = n_quantiles or len(QUANTILE_LEVELS)
            if self.n_quantiles == len(QUANTILE_LEVELS):
                self.levels = tuple(QUANTILE_LEVELS)
            else:  # 均匀内插（与 rams_net.M1Head 的兜底一致）
                self.levels = tuple(
                    round((i + 1) / (self.n_quantiles + 1), 4) for i in range(self.n_quantiles)
                )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        H = target.shape[1]
        n_q = self.n_quantiles
        e = target.unsqueeze(1) - pred.reshape(-1, n_q, H)  # (B, n_q, H)
        lv = torch.tensor(self.levels, device=pred.device)
        losses = [
            torch.mean(torch.maximum(lv[i] * e[:, i], (lv[i] - 1.0) * e[:, i])) for i in range(n_q)
        ]
        return torch.stack(losses).mean()


class MultiTaskLoss(nn.Module):
    """多任务损失：M1 分位数 + M2 交叉熵 + M4 交叉熵（可配权 + 类别权重）。

    I/O:
        forward(m1_out, m2_out, y, strat_label, m4_out=None, warn_label=None)
          m1_out (B, n_q*H); y (B, H); strat (B,); m4 (B, n_levels); warn (B,)
          → (total, l1, l2, l4)
    """

    def __init__(
        self,
        horizon: int,
        n_quantiles: int | None = None,
        levels: tuple[float, ...] | list[float] | None = None,
        w_m1: float = W_M1,
        w_m2: float = W_M2,
        w_m4: float = W_M4,
        use_m4: bool = True,
        warn_class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.horizon = horizon
        self.w_m1 = w_m1
        self.w_m2 = w_m2
        self.w_m4 = w_m4
        self.use_m4 = use_m4
        self.q_loss = QuantileLoss(n_quantiles, levels)
        self.ce = nn.CrossEntropyLoss()
        self.ce_warn = (
            nn.CrossEntropyLoss(weight=warn_class_weights)
            if warn_class_weights is not None
            else nn.CrossEntropyLoss()
        )

    def forward(self, m1_out, m2_out, y, strat_label, m4_out=None, warn_label=None):
        l1 = self.q_loss(m1_out, y)
        l2 = self.ce(m2_out, strat_label)
        l4 = None
        if self.use_m4 and m4_out is not None and warn_label is not None:
            l4 = self.ce_warn(m4_out, warn_label)
        total = self.w_m1 * l1 + self.w_m2 * l2
        if l4 is not None:
            total = total + self.w_m4 * l4
        return total, l1, l2, l4


# =====================================================================
# 评估指标（CRPS / 覆盖率，T4 协议）
# =====================================================================
def crps_cdf_pline(q, p_levels, y):
    """分段线性 CDF 的 CRPS 闭合形式（run_g.crps_cdf_pline 的正式版，任意分位数结）。

    尾部：最低/最高结之外用最外侧段斜率线性外推到 p=0 / p=1（与 B7 一致）。

    Args:
        q: (..., n_q) 分位数预测，结序升序。
        p_levels: (n_q,) 对应分位数水平，升序。
        y: (...) 观测。

    Returns:
        CRPS，与 y 同形状。
    """
    q = np.asarray(q, dtype=np.float64)
    p = np.asarray(p_levels, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if q.shape[-1] != p.shape[0]:
        raise ValueError("q 最后一维须与 p_levels 等长")
    q = np.sort(q, axis=-1)
    p = np.sort(p)
    sL = (q[..., 1] - q[..., 0]) / (p[1] - p[0])
    sR = (q[..., -1] - q[..., -2]) / (p[-1] - p[-2])
    qk = np.concatenate(
        [(q[..., 0] - sL * p[0])[..., None], q, (q[..., -1] + sR * (1.0 - p[-1]))[..., None]],
        axis=-1,
    )
    ak = np.concatenate([[0.0], p, [1.0]])
    deg = (qk[..., -1] - qk[..., 0]) < 1e-9
    total = np.zeros_like(y, dtype=np.float64)
    for k in range(len(ak) - 1):
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
            total += 2.0 * (
                C0 * (v - u) + C1 * (v * v - u * u) / 2.0 - p1 * (v * v * v - u * u * u) / 3.0
            )
    out = np.where(deg, np.abs(y - np.median(q, axis=-1)), total)
    return np.maximum(out, 0.0)


def crps_quantiles(q10, q50, q90, y):
    """3 分位（p10/p50/p90）的 CRPS 闭合形式（0.1.0/探索兼容，5 结含外推）。

    Args:
        q10/q50/q90: 分位数预测（conc 单位，任意广播形状）。
        y: 观测。

    Returns:
        CRPS，与 y 同形状。
    """
    qs = np.sort(np.stack([np.asarray(q10), np.asarray(q50), np.asarray(q90)], axis=-1), axis=-1)
    return crps_cdf_pline(qs, [0.1, 0.5, 0.9], np.asarray(y))


# =====================================================================
# M4 预警标签（日级协议）
# =====================================================================
def make_m4_labels(
    y_abs: np.ndarray,
    n_train: int,
    n_levels: int = 4,
    mode: str = "peak_quantile",
    bloom: np.ndarray | None = None,
) -> np.ndarray:
    """从未来 H 天目标构建 M4 预警等级标签（训练段阈值，防泄漏）。

    Args:
        y_abs: (B, H) 未来浓度（原始 conc 单位，日级协议用 y_abs）。
        n_train: 训练样本数（仅用其拟合阈值）。
        n_levels: 等级数。mode="peak_quantile" 时固定 4（0.75/0.90/0.97 分位切）；
            mode="bloom" 时强制 2（0/1）。
        mode: "peak_quantile" = 未来峰值浓度分位阈值（探索 A/K 协议，4 级）；
            "bloom" = 藻华状态（N 定义，预测日是否藻华，二分类）。
        bloom: (B,) 藻华状态标签（N 定义，mdl-data-scale BloomLabeler 产出）；
            mode="bloom" 时必须提供。

    Returns:
        np.ndarray[int64]: (B,) 预警等级标签。
    """
    if mode == "peak_quantile":
        warn_val = np.asarray(y_abs).max(axis=1)
        qs = np.quantile(warn_val[:n_train], [0.75, 0.90, 0.97])
        return np.searchsorted(qs, warn_val).astype(np.int64)
    if mode == "bloom":
        if bloom is None:
            raise ValueError("mode='bloom' 需要提供 bloom 标签")
        return np.asarray(bloom).astype(np.int64)
    raise ValueError(f"未知 M4 标签模式: {mode}")


# =====================================================================
# 训练器
# =====================================================================
class Trainer:
    """轻量训练器（不依赖 Lightning）：多任务加权 + 两阶段 + fast_dev_run 冒烟。

    Args:
        model: RamsNet。
        lr / w_m1 / w_m2 / w_m4: 学习率与多任务权重（默认 1/3/2）。
        device: "cuda" / "cpu"。
        warn_class_weights: M4 类别权重；None 时由训练段自动逆频率（见 fit_multi）。
    """

    def __init__(
        self,
        model: RamsNet,
        lr: float = 1e-3,
        w_m1: float = W_M1,
        w_m2: float = W_M2,
        w_m4: float = W_M4,
        device: str | None = None,
        warn_class_weights: torch.Tensor | None = None,
    ):
        """Args:
        model: RamsNet。
        lr / w_m1 / w_m2 / w_m4: 学习率与多任务权重（默认 1/3/2）。
        device: "cuda" / "cpu"；None = 自动（有 CUDA 用 cuda，否则 cpu）。
        warn_class_weights: M4 类别权重；None 时由训练段自动逆频率（见 fit_multi）。
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.w_m1, self.w_m2, self.w_m4 = w_m1, w_m2, w_m4
        self.use_m4 = model.use_m4
        self.n_levels = model.m4.mlp[-1].out_features if model.use_m4 else 0
        self.warn_class_weights = (
            warn_class_weights.to(self.device) if warn_class_weights is not None else None
        )
        self._opt = None  # 每个 stage 独立 optimizer

    # ---------- 数据准备 ----------
    def _m4_weights(self, warn_tr: np.ndarray) -> torch.Tensor:
        """M4 类别权重（训练段逆频率，处理等级不平衡，防泄漏）。"""
        counts = np.bincount(warn_tr, minlength=self.n_levels)
        inv = 1.0 / (counts.astype(np.float64) + 1.0)
        w = torch.tensor(inv / inv.sum() * self.n_levels, dtype=torch.float32)
        return w.to(self.device)

    def _loader(self, *arrays: np.ndarray, batch_size: int, shuffle: bool = True):
        tensors = [torch.tensor(a) for a in arrays]
        ds = TensorDataset(*tensors)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    def _to(self, *tensors: torch.Tensor) -> list[torch.Tensor]:
        return [t.to(self.device) for t in tensors]

    # ---------- 训练循环 ----------
    def _run_epochs(
        self, dl, Xv, yv, sv, wv, epochs: int, fast_dev_run: bool
    ) -> list[tuple[float, float, float]]:
        """共享训练循环骨架。Xv/yv/sv/wv 为验证张量（可为 None）。"""
        epochs = 2 if fast_dev_run else epochs
        history: list[tuple[float, float, float]] = []
        for ep in range(epochs):
            self.model.train()
            last_loss = float("nan")
            for bi, batch in enumerate(dl):
                if fast_dev_run and bi >= 2:
                    break
                xb, yb, sb, *rest = self._to(*batch)
                sb = sb.long()  # 分类标签强制 long（numpy Windows 默认 int32）
                wb = rest[0].long() if rest else None
                self._opt.zero_grad()
                m1, m2, m4 = self.model(xb)
                loss, l1, l2, l4 = self.criterion(m1, m2, yb, sb, m4, wb)
                loss.backward()
                self._opt.step()
                last_loss = float(loss.item())
            # 验证
            val_rmse, val_acc, val_wacc = float("nan"), float("nan"), float("nan")
            if Xv is not None:
                self.model.eval()
                with torch.no_grad():
                    m1v, m2v, m4v = self.model(Xv)
                    pred = self.model.predict_mean(m1v)
                    val_rmse = torch.sqrt(torch.mean((pred - yv) ** 2)).item()
                    val_acc = (m2v.argmax(1) == sv).float().mean().item()
                    if m4v is not None and wv is not None:
                        val_wacc = (m4v.argmax(1) == wv).float().mean().item()
            history.append((last_loss, val_rmse, val_acc))
            if ep % 10 == 0 or ep == epochs - 1:
                extra = f" val_wacc={val_wacc:.4f}" if wv is not None else ""
                print(
                    f"  ep{ep} loss={last_loss:.4f} val_rmse={val_rmse:.4f} "
                    f"val_acc={val_acc:.4f}{extra}",
                    flush=True,
                )
        return history

    # ---------- 单阶段多任务（A 口径） ----------
    def fit(
        self,
        X_tr,
        y_tr,
        strat_tr,
        X_va=None,
        y_va=None,
        strat_va=None,
        warn_tr=None,
        warn_va=None,
        epochs: int = 30,
        batch_size: int = 128,
        fast_dev_run: bool = False,
    ) -> list[tuple[float, float, float]]:
        """单阶段多任务训练（M1+M2+M4，w=1/3/2）。

        Args:
            X_tr (n, T, F); y_tr (n, H); strat_tr (n,); warn_tr (n,) 可选。
            X_va / y_va / strat_va / warn_va: 验证集（可选）。
            epochs: 训练轮数（fast_dev_run 时强制 2）。
            fast_dev_run: 冒烟——2 epoch × 前 2 batch。

        Returns:
            history: [(loss, val_rmse, val_acc), ...]
        """
        self.criterion = MultiTaskLoss(
            self.model.horizon,
            n_quantiles=self.model.n_quantiles,
            levels=getattr(self.model.m1, "quantile_levels", None),
            w_m1=self.w_m1,
            w_m2=self.w_m2,
            w_m4=self.w_m4,
            use_m4=self.use_m4,
            warn_class_weights=self.warn_class_weights,
        )
        if self.use_m4 and warn_tr is not None and self.warn_class_weights is None:
            self.criterion.ce_warn = nn.CrossEntropyLoss(
                weight=self._m4_weights(np.asarray(warn_tr))
            )
        self._opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        arrays = [X_tr, y_tr, strat_tr]
        if self.use_m4 and warn_tr is not None:
            arrays.append(warn_tr)
        dl = self._loader(*arrays, batch_size=batch_size)
        Xv = yv = sv = wv = None
        if X_va is not None:
            Xv, yv, sv = self._to(torch.tensor(X_va), torch.tensor(y_va), torch.tensor(strat_va))
            if self.use_m4 and warn_va is not None:
                wv = torch.tensor(warn_va).to(self.device)
        return self._run_epochs(dl, Xv, yv, sv, wv, epochs, fast_dev_run)

    # ---------- 两阶段（K 口径） ----------
    def fit_two_stage(
        self,
        X_tr,
        y_tr,
        strat_tr,
        warn_tr=None,
        ep1: int = 20,
        ep2: int = 10,
        batch_size: int = 128,
        freeze_backbone: bool = True,
        fast_dev_run: bool = False,
    ) -> tuple[list, list]:
        """两阶段训练（K 口径 ts_freeze）：Stage1 单任务 M1 → Stage2 冻结 backbone 微调多头。

        Args:
            X_tr (n, T, F); y_tr (n, H) 增量归一化目标; strat_tr (n,); warn_tr (n,) 可选。
            ep1: Stage1 轮数（默认 20）。
            ep2: Stage2 轮数（默认 10）。
            freeze_backbone: True=ts_freeze（推荐）；False=ts_full。
            fast_dev_run: 冒烟（2 ep × 前 2 batch）。

        Returns:
            (history_stage1, history_stage2)。
        """
        h1 = self.fit_m1_only(
            X_tr, y_tr, epochs=ep1, batch_size=batch_size, fast_dev_run=fast_dev_run
        )
        h2 = self.fit_multi(
            X_tr,
            y_tr,
            strat_tr,
            warn_tr,
            epochs=ep2,
            batch_size=batch_size,
            freeze_backbone=freeze_backbone,
            fast_dev_run=fast_dev_run,
        )
        return h1, h2

    def fit_m1_only(
        self, X_tr, y_tr, epochs: int = 20, batch_size: int = 128, fast_dev_run: bool = False
    ) -> list[tuple[float, float, float]]:
        """Stage1：单任务只训 M1 分位数（保精度；M2/M4 头零梯度不参与）。

        Args:
            X_tr (n, T, F); y_tr (n, H)（增量归一化目标）。
            epochs: Stage1 轮数（默认 20，K 口径）。

        Returns:
            history: [(loss, val_rmse, val_acc), ...]（val 均为 NaN，无验证集）。
        """
        self.criterion = MultiTaskLoss(
            self.model.horizon,
            n_quantiles=self.model.n_quantiles,
            levels=getattr(self.model.m1, "quantile_levels", None),
            w_m1=1.0,
            w_m2=0.0,
            w_m4=0.0,
            use_m4=False,
        )
        self._opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        dl = self._loader(X_tr, y_tr, np.zeros(len(X_tr), dtype=np.int64), batch_size=batch_size)
        return self._run_epochs(dl, None, None, None, None, epochs, fast_dev_run)

    def fit_multi(
        self,
        X_tr,
        y_tr,
        strat_tr,
        warn_tr=None,
        epochs: int = 10,
        batch_size: int = 128,
        freeze_backbone: bool = True,
        fast_dev_run: bool = False,
    ) -> list[tuple[float, float, float]]:
        """Stage2：多任务微调（M1+M2+M4，w=1/3/2），默认冻结 backbone。

        Args:
            X_tr (n, T, F); y_tr (n, H); strat_tr (n,); warn_tr (n,) 可选。
            epochs: Stage2 轮数（默认 10，K 口径）。
            freeze_backbone: True=冻结 GRU 只训头（ts_freeze，推荐）；
                False=解冻全部（ts_full）。
            fast_dev_run: 冒烟。

        Returns:
            history: [(loss, val_rmse, val_acc), ...]（无验证集时 val 为 NaN）。
        """
        self.criterion = MultiTaskLoss(
            self.model.horizon,
            n_quantiles=self.model.n_quantiles,
            levels=getattr(self.model.m1, "quantile_levels", None),
            w_m1=self.w_m1,
            w_m2=self.w_m2,
            w_m4=self.w_m4,
            use_m4=self.use_m4,
            warn_class_weights=self.warn_class_weights,
        )
        if self.use_m4 and warn_tr is not None and self.warn_class_weights is None:
            self.criterion.ce_warn = nn.CrossEntropyLoss(
                weight=self._m4_weights(np.asarray(warn_tr))
            )
        if freeze_backbone:
            for p in self.model.backbone.parameters():
                p.requires_grad = False
        try:
            params = [p for p in self.model.parameters() if p.requires_grad]
            self._opt = torch.optim.Adam(params, lr=self.lr)
            arrays = [X_tr, y_tr, strat_tr]
            if self.use_m4 and warn_tr is not None:
                arrays.append(warn_tr)
            dl = self._loader(*arrays, batch_size=batch_size)
            return self._run_epochs(dl, None, None, None, None, epochs, fast_dev_run)
        finally:
            if freeze_backbone:
                for p in self.model.backbone.parameters():
                    p.requires_grad = True

    # ---------- 评估 ----------
    def evaluate(self, X_te, y_te, strat_te, warn_te=None, y_sd: float = 1.0) -> dict:
        """测试集评估：RMSE（还原尺度）+ M2 acc + M4 acc + p10-p90 覆盖率。

        Args:
            X_te (n, T, F); y_te (n, H); strat_te (n,); warn_te (n,) 可选。
            y_sd: 目标标准差（归一化目标 → 原始单位换算）。

        Returns:
            dict: rmse / rmse_norm / acc / warn_acc / coverage / interval_width。
        """
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X_te).to(self.device)
            m1, m2, m4 = self.model(Xt)
            pred = self.model.predict_mean(m1).cpu().numpy()
            acc = float((m2.argmax(1).cpu().numpy() == np.asarray(strat_te)).mean())
            rmse_norm = float(np.sqrt(np.mean((pred - np.asarray(y_te)) ** 2)))
            result = {
                "rmse": float(rmse_norm * y_sd),
                "rmse_norm": rmse_norm,
                "acc": acc,
            }
            if m4 is not None and warn_te is not None:
                result["warn_acc"] = float(
                    (m4.argmax(1).cpu().numpy() == np.asarray(warn_te)).mean()
                )
            if self.model.quantile:
                p10, p90 = self.model.predict_interval(m1)
                p10 = p10.cpu().numpy()
                p90 = p90.cpu().numpy()
                cover = float(np.mean((np.asarray(y_te) >= p10) & (np.asarray(y_te) <= p90)))
                result["coverage"] = cover
                result["interval_width"] = float(np.mean(p90 - p10) * y_sd)
        return result

    def predict_m1(self, X_te) -> np.ndarray:
        """M1 分位数预测，还原为 (N, n_q, H)（供 CRPS 评估）。"""
        self.model.eval()
        with torch.no_grad():
            m1, _, _ = self.model(torch.tensor(X_te).to(self.device))
            q = self.model.quantile_matrix(m1).cpu().numpy().astype(np.float64)
        return q


if __name__ == "__main__":
    # 冒烟：小张量两阶段训练 + 形状/loss/覆盖率验证（不涉真实数据）
    import sys

    sys.path.insert(0, ".")
    torch.manual_seed(0)
    np.random.seed(0)

    from rams.models.rams_net import RamsNet

    B, T, F, H = 64, 30, 29, 7
    model = RamsNet(feat_dim=F, horizon=H, hidden=16, use_m4=True, n_quantiles=9)
    X = np.random.randn(B, T, F).astype(np.float32)
    y = np.random.randn(B, H).astype(np.float32)
    s = np.random.randint(0, 2, B)
    w = np.random.randint(0, 4, B)

    tr = Trainer(model, device="cpu")
    print("Stage1 (单任务 M1):")
    tr.fit_m1_only(X, y, epochs=2, batch_size=16, fast_dev_run=True)
    print("Stage2 (冻结 backbone 多任务):")
    tr.fit_multi(X, y, s, w, epochs=2, batch_size=16, freeze_backbone=True, fast_dev_run=True)
    res = tr.evaluate(X, y, s, w, y_sd=1.0)
    print(f"evaluate: {res}")
    assert np.isfinite(res["rmse"]), "RMSE 非有限"
    assert 0.0 <= res["coverage"] <= 1.0, "覆盖率越界"
    q = tr.predict_m1(X)
    assert q.shape == (B, 9, H), f"q9 输出形状错误: {q.shape}"
    # CRPS 冒烟：q9 分位数 CRPS 有限（逐视界调用，q: (B, n_q)）
    levels = np.array([0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95])
    y_arr = np.asarray(y)
    crps = np.mean([crps_cdf_pline(q[:, :, h], levels, y_arr[:, h]) for h in range(H)])
    assert np.isfinite(crps), "CRPS 含非有限值"
    print(f"CRPS 均值: {float(crps):.4f}")
    print("冒烟通过")
