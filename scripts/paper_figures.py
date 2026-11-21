"""Paper figures (Phase 9), drawn only from saved result files.

  fig2_tradeoff.png        utility vs presence leakage along the compression curve (Phase 4b)
  fig3_reconstructions.png real tiles vs GAN reconstructions, true panel mask outlined (Phase 5)
  fig4_split_curves.png    presence AUC, spatial macro AP and BigEarthNet AP by split point (Phase 8b)
  fig5_isl.png             payload bytes vs live capacity of each link class, coloured by leakage (Phase 8c)

Usage: python scripts/paper_figures.py   (writes to paper/figures/)
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "paper/figures"
os.makedirs(OUT, exist_ok=True)

# Categorical slots (fixed order, never cycled); controls in neutral greys.
ENC = [("resnet18_moco", "R18-MoCo", "#2a78d6"), ("vits16_moco", "ViT-MoCo", "#eb6834"),
       ("resnet50_dino", "R50-DINO", "#1baf7a"), ("vits16_dino", "ViT-DINO", "#eda100")]
GREY, INK, MUTED = "#8a8a85", "#2b2b29", "#6b6b66"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b5b0",
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": "#e6e6e2", "grid.linewidth": 0.6, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 2, "lines.markersize": 6,
})


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------- Figure 2: trade-off ----------
def fig_tradeoff():
    best = read("results/phase4b/analysis/best.csv")
    g = {(r["encoder"], r["variant"], r["task"]): float(r["test"]) for r in best if r["split_seed"] == "0"}
    levels = ["full", "pca256", "pca64", "pca16"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), sharey=True)
    for ax, task, xlab in ((axes[0], "utility_ben", "BigEarthNet macro AP (utility)"),
                           (axes[1], "utility_eurosat", "EuroSAT accuracy (utility)")):
        for key, name, col in ENC + [("resnet18_random", "Random R18 (control)", GREY)]:
            xs = [g[(key, v, task)] for v in levels if (key, v, task) in g]
            ys = [g[(key, v, "leakage")] for v in levels if (key, v, task) in g]
            ls = "--" if key == "resnet18_random" else "-"
            ax.plot(xs, ys, ls, color=col, marker="o", label=name, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.2)
            ax.plot([xs[0]], [ys[0]], marker="o", ms=9, color=col, markeredgecolor=INK, markeredgewidth=1.2,
                    ls="none", zorder=4)
        px, py = g[("pixelstats", "full", task)], g[("pixelstats", "full", "leakage")]
        ax.axhline(py, color=GREY, lw=1, ls=":", zorder=1)
        ax.text(0.02, py + 0.002, "pixel statistics", transform=ax.get_yaxis_transform(),
                fontsize=7, color=MUTED, va="bottom")
        ax.plot([px], [py], marker="D", color=INK, ls="none", label="Pixel statistics (control)", zorder=4)
        ax.set_xlabel(xlab)
    axes[0].set_ylabel("Presence leakage (test ROC-AUC)")
    axes[0].set_title("(a) BigEarthNet utility")
    axes[1].set_title("(b) EuroSAT utility")
    axes[1].legend(loc="lower right", frameon=False)
    fig.savefig(f"{OUT}/fig2_tradeoff.png")
    plt.close(fig)


# ---------- Figure 3: reconstructions ----------
def fig_recon():
    cols = [("real", "Real tile"), ("vits16_moco", "GAN: ViT-MoCo"), ("resnet50_dino", "GAN: R50-DINO"),
            ("pixelstats", "GAN: pixel stats")]
    ex = {k: np.load(f"results/phase5/gan/{k}/full/examples.npz") for k, _ in cols[1:]}
    rows = ex["vits16_moco"]["rows"]
    y = np.load("data/embeddings/splits_glosofarid.npz")["y"][rows]
    masks = np.load("data/glosofarid/npy/tiles_mask.npy", mmap_mode="r")
    pick = [i for i in range(len(rows)) if y[i] == 1][:4] + [i for i in range(len(rows)) if y[i] == 0][:1]

    def rgb(t):  # bands B4, B3, B2 (indices 3, 2, 1), stretched per tile from the real tile's range
        return t[[3, 2, 1]].transpose(1, 2, 0).astype(np.float32)

    fig, axes = plt.subplots(len(pick), len(cols), figsize=(5.6, 1.45 * len(pick)))
    for r, i in enumerate(pick):
        real = rgb(ex["vits16_moco"]["real"][i])
        lo, hi = np.percentile(real, 2), np.percentile(real, 98)
        m = np.asarray(masks[rows[i]])
        for c, (k, title) in enumerate(cols):
            img = real if k == "real" else rgb(ex[k]["recon"][i])
            ax = axes[r, c]
            ax.imshow(np.clip((img - lo) / (hi - lo + 1e-6), 0, 1), interpolation="nearest")
            if m.any():
                ax.contour(m, levels=[0.5], colors="#e8f0fe", linewidths=0.9)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(title, fontsize=8)
        axes[r, 0].set_ylabel("panels" if m.any() else "no panels", fontsize=7.5, color=MUTED)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(f"{OUT}/fig3_reconstructions.png")
    plt.close(fig)


# ---------- Figure 4: split curves ----------
def fig_splits():
    cells = read("results/phase8/analysis/cells.csv")
    panels = [("gap", "leakage", "(a) Presence leakage, pooled vector", "ROC-AUC"),
              ("map4", "spatial", "(b) Location, spatial map", "Spatial macro AP"),
              ("gap", "utility_ben", "(c) Utility, pooled vector", "BigEarthNet macro AP")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, (form, task, title, ylab) in zip(axes, panels):
        for key, name, col in ENC + [("resnet18_random", "Random R18", GREY)]:
            rs = sorted([r for r in cells if r["encoder"] == key and r["form"] == form and r["task"] == task],
                        key=lambda r: int(r["split_index"]))
            x = [int(r["split_index"]) for r in rs]
            yv = np.array([float(r["test"]) for r in rs])
            lo = yv - np.array([float(r["ci_lo"]) for r in rs])
            hi = np.array([float(r["ci_hi"]) for r in rs]) - yv
            ax.errorbar(x, yv, yerr=[lo, hi], color=col, marker="o", capsize=2, elinewidth=0.8,
                        ls="--" if key == "resnet18_random" else "-", label=name,
                        markeredgecolor="white", markeredgewidth=1)
        if task == "leakage":
            ax.axhline(0.910, color=INK, lw=1, ls=":", label="Pixel statistics")
        ax.set_xticks([1, 2, 3, 4], ["1st", "2nd", "3rd", "final"])
        ax.set_xlabel("Split point")
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=8.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(f"{OUT}/fig4_split_curves.png")
    plt.close(fig)


# ---------- Figure 5: ISL feasibility ----------
def fig_isl():
    pay = read("results/phase8/isl_payloads.csv")
    links = [("S-band 2 Mbit/s", 2e6), ("Ka-band 500 Mbit/s", 5e8), ("optical 1.8 Gbit/s", 1.8e9),
             ("optical 10 Gbit/s", 1e10)]
    tiles_s = 4693.5
    kinds = [("quantised", "Quantised PCA codes (q8-q1)", "#2a78d6"),
             ("compressed", "PCA / int8 embeddings", "#eb6834"),
             ("split", "Split-point payloads (pooled and map)", "#1baf7a")]
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    for kind, name, col in kinds:
        rs = [r for r in pay if r["kind"] == kind and r["leakage"]]
        ax.scatter([float(r["bytes"]) for r in rs], [float(r["leakage"]) for r in rs], s=22, color=col,
                   edgecolor="white", linewidth=0.6, label=name, zorder=3)
    caps = []
    for name, rate in links:
        cap = rate / (8 * tiles_s)
        caps.append(cap)
        ax.axvline(cap, color=MUTED, lw=1, ls="--", zorder=1)
    from matplotlib.ticker import NullLocator
    top = ax.twiny()
    top.set_xscale("log"); top.set_xlim(1, 5e5)
    top.set_xticks(caps)
    top.set_xticklabels(["S-band\n2 Mbit/s", "Ka-band\n500 Mbit/s", "optical\n1.8 Gbit/s", "optical\n10 Gbit/s"])
    top.xaxis.set_minor_locator(NullLocator())
    top.tick_params(length=0, colors=MUTED, labelsize=6.5)
    top.grid(False)
    top.spines["top"].set_visible(False)
    top.set_xlabel("largest payload each link class carries live (dashed)", fontsize=7, color=MUTED)
    for label, val, ha, dx in (("zlib raw tile", 36013, "right", 0.9), ("raw tile", 106496, "left", 1.1)):
        ax.axvline(val, color=INK, lw=1.2, zorder=1)
        ax.text(val * dx, 0.793, label, fontsize=6.5, color=INK, rotation=90, ha=ha, va="bottom")
    ax.axhline(0.910, color=GREY, lw=1, ls=":")
    ax.text(1.2, 0.9115, "pixel statistics", fontsize=6.5, color=MUTED, va="bottom")
    ax.set_xscale("log"); ax.set_xlim(1, 5e5); ax.set_ylim(0.79, 0.955)
    ax.set_xlabel("Payload size per tile (bytes, log scale)")
    ax.set_ylabel("Presence leakage (ROC-AUC)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, frameon=False, fontsize=7)
    fig.savefig(f"{OUT}/fig5_isl.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_tradeoff(); fig_recon(); fig_splits(); fig_isl()
    print("figures written to", OUT)
