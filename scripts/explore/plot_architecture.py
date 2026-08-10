# -*- coding: utf-8 -*-
"""RAMS 模型架构图：仿照 "Attention Is All You Need" 的框图风格。

绘制共享 GRU backbone + M1/M2/M4 三头的完整架构：
  - 每个组件一个方框（虚线=维度可变/可选，实线=数据流）
  - 标注每个张量的形状 (B, T, D, C)
  - 标注多任务 loss 汇合
  - 标注 M5 因果结论回流（虚线反馈）

用 matplotlib + matplotlib.patches，无第三方依赖。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis("off")

# ===== 样式 =====
C_INPUT = "#E8F0FE"    # 输入浅蓝
C_EMBED = "#FFF3E0"    # 编码浅橙
C_BACKBONE = "#E8F5E9"  # 主干浅绿
C_HEAD = "#FCE4EC"     # 头浅粉
C_OUT = "#F3E5F5"      # 输出浅紫
C_FEED = "#90CAF9"     # 反馈蓝
EDGE = "#455A64"

def box(x, y, w, h, text, fc, fs=9, lw=1.5, style="round,pad=0.02", ec=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec or EDGE, lw=lw)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, fontfamily="DejaVu Sans", linespacing=1.4)
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, color=EDGE, lw=1.5, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                        color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)

# ===== 1. 输入层 =====
box(0.4, 9.4, 3.0, 1.2, "Input\n(B, T=24, D, C)\n20层水温 + 6气象", C_INPUT, fs=9)

# 输入分解：三个来源
box(0.4, 7.9, 1.4, 1.1, "温度剖面\ntemp_0.5~10\n(D=20)", C_INPUT, fs=8)
box(2.0, 7.9, 1.4, 1.1, "气象\n6通道\n(短滞)", C_INPUT, fs=8)
box(3.6, 7.9, 1.6, 1.1, "M5 因果先验\nwind_u 短滞\n(可选)", C_FEED, fs=7.5, lw=1, ec=C_FEED)

arrow(1.9, 9.4, 1.1, 9.0, ls="--")   # 输入→分解（示意）
arrow(1.1, 7.9, 1.1, 9.4)           # 温度→输入
arrow(2.7, 7.9, 2.7, 9.4)           # 气象→输入
arrow(4.4, 7.9, 4.0, 9.4, color=C_FEED, ls="--")  # 因果先验→输入

# ===== 2. 时间/深度嵌入 =====
box(4.6, 9.0, 2.2, 1.2, "时间-深度嵌入\n(线性投影)\n(B,T,D,C)→(B,T,F)", C_EMBED, fs=8)
arrow(3.4, 10.0, 4.6, 9.6)

# ===== 3. 共享 GRU backbone =====
box(7.4, 9.0, 2.6, 1.2, "共享 GRU Backbone\n(T 维时序建模)\nhidden=64, 1层\n输出末时刻隐状态 (B,64)", C_BACKBONE, fs=8)
arrow(6.8, 9.6, 7.4, 9.6)

# ===== 4. 三个任务头 =====
# M1
box(4.8, 5.6, 2.6, 1.4, "M1 预测头 (MLP)\n分位数输出\n(B, 3×H=24)\np10/p50/p90", C_HEAD, fs=8)
# M2
box(7.9, 5.6, 2.4, 1.4, "M2 分层头 (MLP)\n二分类\n(B, 2)\n分层/不分层", C_HEAD, fs=8)
# M4
box(10.9, 5.6, 2.4, 1.4, "M4 预警头 (MLP)\n四级分类\n(B, 4)\n安全/注意/警告/危险", C_HEAD, fs=8)

arrow(8.7, 9.0, 8.7, 7.0)  # backbone → 分叉
# 分叉到三个头
arrow(8.7, 7.0, 6.1, 7.0)
arrow(8.7, 7.0, 9.1, 7.0)
arrow(8.7, 7.0, 12.1, 7.0)
arrow(6.1, 7.0, 6.1, 7.0)  # 已在原地

# ===== 5. 输出与 loss =====
box(4.8, 3.4, 2.6, 1.2, "M1 输出\n预测值+区间\n(未来24h)", C_OUT, fs=8)
box(7.9, 3.4, 2.4, 1.2, "M2 输出\n分层状态", C_OUT, fs=8)
box(10.9, 3.4, 2.4, 1.2, "M4 输出\n预警等级", C_OUT, fs=8)

arrow(6.1, 5.6, 6.1, 4.6)
arrow(9.1, 5.6, 9.1, 4.6)
arrow(12.1, 5.6, 12.1, 4.6)

# ===== 6. 多任务 loss 汇合 =====
box(6.0, 1.6, 6.8, 1.2, "多任务 Loss\nL = w₁·L_quantile(M1) + w₂·CE(M2) + w₄·CE(M4)\nw₁=1.0, w₂=3.0, w₄=2.0（M2/M4 高权重）", C_EMBED, fs=8)
arrow(6.1, 3.4, 7.0, 2.8)
arrow(9.1, 3.4, 9.0, 2.8)
arrow(12.1, 3.4, 11.0, 2.8)

# ===== 7. 反馈（M5 因果回流 / M2 分层反馈） =====
# M5 因果结论回流特征选择（虚线）
arrow(13.3, 8.5, 13.3, 5.2, color=C_FEED, ls="--", lw=1)
ax.text(13.45, 6.8, "M5 因果", fontsize=7, color=C_FEED, rotation=90, ha="center")
# M2 分层状态反馈 M1（旁路，可选）
arrow(9.1, 5.6, 6.1, 4.9, color=C_FEED, ls="--", lw=1)
ax.text(7.6, 5.15, "M2 分层→M1（可选旁路）", fontsize=6.5, color=C_FEED, ha="center")

# ===== 图例 =====
legend_items = [
    (C_INPUT, "输入 / 数据"),
    (C_EMBED, "嵌入 / 汇合"),
    (C_BACKBONE, "共享主干"),
    (C_HEAD, "任务头"),
    (C_OUT, "输出"),
]
for i, (c, lbl) in enumerate(legend_items):
    ax.add_patch(mpatches.Rectangle((0.4, 0.3 + i*0.35), 0.3, 0.25, fc=c, ec=EDGE, lw=1))
    ax.text(0.85, 0.43 + i*0.35, lbl, fontsize=7.5, va="center")

# 虚线说明
ax.text(6.0, 0.35, "实线=数据流   虚线=可选/反馈(M5因果, M2旁路)    标注=张量形状 (B,T,D,C)", fontsize=7, color="#78909C")

plt.tight_layout()
out = "/data/RAMS/proj/docs/rams_architecture.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"架构图已保存: {out}")
