# -*- coding: utf-8 -*-
"""RAMS architecture diagram (English labels only - no CJK font on server).

Style follows "Attention Is All You Need" block diagrams:
one box per component, annotated tensor shapes, solid=mandatory / dashed=optional.
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

C_INPUT = "#E8F0FE"
C_EMBED = "#FFF3E0"
C_BACKBONE = "#E8F5E9"
C_HEAD = "#FCE4EC"
C_OUT = "#F3E5F5"
C_FEED = "#90CAF9"
EDGE = "#455A64"

def box(x, y, w, h, text, fc, fs=9, lw=1.5, style="round,pad=0.02", ec=None):
    b = FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec or EDGE, lw=lw)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs,
            fontfamily="DejaVu Sans", linespacing=1.4)
    return (x, y, w, h)

def arrow(x1, y1, x2, y2, color=EDGE, lw=1.5, style="-|>", ls="-"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=12,
                        color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)

# ===== 1. Input =====
box(0.4, 9.4, 3.0, 1.2, "Input\n(B, T=24, D, C)\n20 depth temps + 6 meteo", C_INPUT, fs=9)
box(0.4, 7.9, 1.5, 1.1, "Temp profile\ntemp_0.5~10\n(D=20)", C_INPUT, fs=8)
box(2.1, 7.9, 1.3, 1.1, "Meteo\n6 ch", C_INPUT, fs=8)
box(3.6, 7.9, 1.7, 1.1, "M5 causal\nwind_u\n(optional)", C_FEED, fs=7.5, lw=1, ec=C_FEED)
arrow(1.1, 7.9, 1.5, 9.4)
arrow(2.7, 7.9, 2.7, 9.4)
arrow(4.4, 7.9, 3.9, 9.4, color=C_FEED, ls="--")

# ===== 2. Embedding =====
box(4.6, 9.0, 2.2, 1.2, "Time-Depth\nEmbedding\n(linear proj)", C_EMBED, fs=8)
arrow(3.4, 10.0, 4.6, 9.6)

# ===== 3. Shared backbone =====
box(7.4, 9.0, 2.6, 1.2, "Shared GRU Backbone\ntemporal modeling\nhidden=64, 1 layer\nlast hidden (B,64)", C_BACKBONE, fs=8)
arrow(6.8, 9.6, 7.4, 9.6)

# ===== 4. Heads =====
box(4.8, 5.6, 2.6, 1.4, "M1 Forecast (MLP)\nquantile out\n(B, 3H=24)\np10/p50/p90", C_HEAD, fs=8)
box(7.9, 5.6, 2.4, 1.4, "M2 Stratify (MLP)\nbinary\n(B, 2)", C_HEAD, fs=8)
box(10.9, 5.6, 2.4, 1.4, "M4 Warning (MLP)\n4-class\n(B, 4)", C_HEAD, fs=8)
arrow(8.7, 9.0, 8.7, 7.0)
arrow(8.7, 7.0, 6.1, 7.0)
arrow(8.7, 7.0, 9.1, 7.0)
arrow(8.7, 7.0, 12.1, 7.0)

# ===== 5. Output =====
box(4.8, 3.4, 2.6, 1.2, "M1 Output\npred + interval\n(next 24h)", C_OUT, fs=8)
box(7.9, 3.4, 2.4, 1.2, "M2 Output\nstratification", C_OUT, fs=8)
box(10.9, 3.4, 2.4, 1.2, "M4 Output\nwarning level", C_OUT, fs=8)
arrow(6.1, 5.6, 6.1, 4.6)
arrow(9.1, 5.6, 9.1, 4.6)
arrow(12.1, 5.6, 12.1, 4.6)

# ===== 6. Loss =====
box(6.0, 1.6, 6.8, 1.2, "Multi-task Loss\nL = w1*Lq(M1) + w2*CE(M2) + w4*CE(M4)\nw=(1.0, 3.0, 2.0)", C_EMBED, fs=8)
arrow(6.1, 3.4, 7.0, 2.8)
arrow(9.1, 3.4, 9.0, 2.8)
arrow(12.1, 3.4, 11.0, 2.8)

# ===== 7. Feedback =====
arrow(13.3, 8.5, 13.3, 5.2, color=C_FEED, ls="--", lw=1)
ax.text(13.45, 6.8, "M5 causal\nfeedback", fontsize=7, color=C_FEED, rotation=90, ha="center")
arrow(9.1, 5.6, 6.1, 4.9, color=C_FEED, ls="--", lw=1)
ax.text(7.6, 5.15, "M2->M1 (optional bypass)", fontsize=6.5, color=C_FEED, ha="center")

# ===== Legend =====
legend_items = [
    (C_INPUT, "Input / Data"),
    (C_EMBED, "Embed / Merge"),
    (C_BACKBONE, "Shared backbone"),
    (C_HEAD, "Task head"),
    (C_OUT, "Output"),
]
for i, (c, lbl) in enumerate(legend_items):
    ax.add_patch(mpatches.Rectangle((0.4, 0.3 + i*0.35), 0.3, 0.25, fc=c, ec=EDGE, lw=1))
    ax.text(0.85, 0.43 + i*0.35, lbl, fontsize=7.5, va="center")
ax.text(6.0, 0.35, "solid=data flow  dashed=optional/feedback  annot=tensor shape (B,T,D,C)",
        fontsize=7, color="#78909C")

plt.tight_layout()
out = "/data/RAMS/proj/docs/rams_architecture.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved: {out}")
