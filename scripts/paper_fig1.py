"""Figure 1: the SatSplit-Bench pipeline (split inference, payload, observer, attackers, controls).

Usage: python scripts/paper_fig1.py   (writes paper/figures/fig1_pipeline.png)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

INK, MUTED, LINE = "#2b2b29", "#6b6b66", "#9a9a94"
BLUE, BLUE_BG = "#2a78d6", "#e3eefb"      # split-inference path
ORANGE, ORANGE_BG = "#eb6834", "#fdebe3"  # observer / attackers
GREY_BG = "#f1f1ee"                       # controls

fig, ax = plt.subplots(figsize=(7.4, 3.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 54); ax.axis("off")


def box(x, y, w, h, title, body="", fc=GREY_BG, ec=LINE, tc=INK, lw=1.0, bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw))
    ty = y + h - 2.2 if body else y + h / 2
    ax.text(x + w / 2, ty, title, ha="center", va="center", fontsize=7.6, color=tc,
            fontweight="bold" if bold else "normal")
    if body:
        ax.text(x + w / 2, y + (h - 3.6) / 2, body, ha="center", va="center", fontsize=6.4, color=MUTED,
                linespacing=1.35)


def arrow(x1, y1, x2, y2, col=LINE, ls="-", lw=1.2, text=None, tx=0, ty=1.4):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9, color=col, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0))
    if text:
        ax.text((x1 + x2) / 2 + tx, (y1 + y2) / 2 + ty, text, ha="center", va="bottom", fontsize=6.3, color=col)


# Group labels
ax.text(1, 52.5, "Satellite A (on board)", fontsize=7.2, color=BLUE, fontweight="bold")
ax.text(66, 52.5, "Satellite B / relay / ground", fontsize=7.2, color=BLUE, fontweight="bold")
ax.text(1, 23.5, "Observer of the payload (audit)", fontsize=7.2, color=ORANGE, fontweight="bold")

# Top row: split inference
box(1, 38, 13, 12, "Tile", "64 x 64 x 13\nSentinel-2, 10 m\n106,496 B", fc="white")
box(18, 38, 16, 12, "Encoder head", "split after stage k\nResNet layer1-4\nViT block 3/6/9/12", fc=BLUE_BG, ec=BLUE)
box(38, 38, 18, 12, "Payload", "pooled vector (gap)\nor spatial map (map4)\n+ PCA / int8 / 1-8 bit", fc=BLUE_BG, ec=BLUE,
    lw=1.8)
box(66, 38, 15, 12, "Encoder tail", "remaining stages", fc=BLUE_BG, ec=BLUE)
box(85, 38, 14, 12, "Task", "land cover\nEuroSAT,\nBigEarthNet", fc="white")
arrow(14.8, 44, 17.4, 44, BLUE)
arrow(34.8, 44, 37.4, 44, BLUE)
arrow(56.8, 44, 65.4, 44, BLUE, text="ISL", ty=1.0)
ax.text(61, 42.6, "S / Ka /\noptical", ha="center", va="top", fontsize=5.8, color=MUTED)
arrow(81.8, 44, 84.4, 44, BLUE)
ax.text(92, 36.2, "= utility", ha="center", va="top", fontsize=6.3, color=BLUE)

# Observer tap
arrow(47, 37.4, 47, 22.6, ORANGE, ls=(0, (3, 2)), lw=1.3, text="sees payload only", tx=8.5, ty=-1.5)

# Bottom row: attackers
box(1, 9, 23, 12, "Classifiers", "MLP, LightGBM, SVM\n-> presence AUC\n-> 4 x 4 cell location", fc=ORANGE_BG, ec=ORANGE)
box(27, 9, 23, 12, "Inverter + segmenter", "kNN / GAN rebuilds tile;\nfrozen U-Net finds panels\n-> detection AUC, mask IoU",
    fc=ORANGE_BG, ec=ORANGE)
box(53, 9, 22, 12, "Localiser", "query-based transformer\ndecoder -> 64 x 64 mask\n-> mask IoU", fc=ORANGE_BG, ec=ORANGE)
for x in (12.5, 38.5, 64):
    arrow(47, 22.6, x, 21.6, ORANGE, lw=1.0)

# Controls
box(79, 9, 20, 12, "Compared with", "pixel statistics (65)\nlocation prior\nraw-image CNN bound", fc=GREY_BG)
arrow(75.8, 15, 78.4, 15, LINE)

ax.text(50, 3.2, "= leakage: solar-farm presence and location, recovered by models trained after the encoder is frozen",
        ha="center", fontsize=6.6, color=ORANGE)

fig.savefig(f"{OUT}/fig1_pipeline.png", dpi=300, bbox_inches="tight")
print("written", f"{OUT}/fig1_pipeline.png")
