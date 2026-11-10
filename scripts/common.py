"""Shared paths, data loading and statistics for Phase 4b and later."""
import os

import numpy as np

EMB = "data/embeddings"
COMP = "data/compressed"
AUDITED = ["resnet18_moco", "vits16_moco", "resnet50_dino", "vits16_dino"]
CONTROLS = ["resnet18_random", "pixelstats"]
FLOAT_VARIANTS = ["full", "pca256", "pca64", "pca16"]
EUROSAT_CLASSES = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
TILE = 64


# ---------------------------------------------------------------- splits and labels

def splits_path(seed=0):
    return os.path.join(EMB, "splits_glosofarid.npz" if seed == 0 else f"splits_glosofarid_seed{seed}.npz")


def load_splits(seed=0):
    """Per GloSoFarID tile: y, patch_id, tile_idx, split (0 train, 1 val, 2 test)."""
    p = splits_path(seed)
    if not os.path.exists(p):
        raise SystemExit(f"Missing {p}: run scripts/splits.py --seeds {seed}")
    return dict(np.load(p))


def eurosat_labels():
    return {s: np.load(f"data/eurosat/npy/{s}_y.npy") for s in ("train", "val", "test")}


def patch_groups(patch_ids):
    """Index arrays, one per patch, for cluster bootstrap over rows with these patch ids."""
    order = np.argsort(patch_ids, kind="stable")
    _, starts = np.unique(patch_ids[order], return_index=True)
    return np.split(order, starts[1:])


# ---------------------------------------------------------------- compressed embeddings

def dequantise(d, a):
    if os.path.exists(os.path.join(d, "codebook.npy")):  # per-dimension quantile codes
        cb = np.load(os.path.join(d, "codebook.npy"))
        return cb[np.arange(cb.shape[0])[None, :], a.astype(np.int64)].astype(np.float32)
    if os.path.exists(os.path.join(d, "scale.npy")):  # symmetric int8
        return a.astype(np.float32) * np.load(os.path.join(d, "scale.npy"))
    return a.astype(np.float32)


QUANT = "data/quantised"


def variant_dir(enc, var):
    """Phase 3 variants live in data/compressed, Phase 4b quantile variants (<level>_q<b>) in data/quantised."""
    for root in (COMP, QUANT):
        d = os.path.join(root, enc, var)
        if os.path.isdir(d):
            return d
    raise FileNotFoundError(f"no variant {enc}/{var} in {COMP} or {QUANT}")


def load_variant(enc, var, names=("eurosat_train", "eurosat_val", "eurosat_test", "glosofarid")):
    d = variant_dir(enc, var)
    out = {}
    for n in names:
        f = os.path.join(d, f"{n}.npy")
        if os.path.exists(f):
            out[n] = dequantise(d, np.load(f))
    return out


def variant_bytes(enc, var, dims):
    if var.startswith("q") or "_q" in var:
        bits = int(var.split("q")[-1])
        return dims * bits / 8
    return dims * (1 if var.endswith("_int8") else 4)


# ---------------------------------------------------------------- GloSoFarID tile images (Phase 4b upper bound, Phase 5)

TILES_X = "data/glosofarid/npy/tiles_x.npy"        # (55408, 13, 64, 64) uint16, row = tile id
TILES_MASK = "data/glosofarid/npy/tiles_mask.npy"  # (55408, 64, 64) uint8
BAND_STATS = "data/glosofarid/npy/band_stats_train.npz"


def build_tile_cache():
    """Contiguous per-tile copies of images and masks (same tile order as embeddings and splits)."""
    if os.path.exists(TILES_X) and os.path.exists(TILES_MASK):
        return
    x = np.load("data/glosofarid/npy/solar_2023_global_x.npy", mmap_mode="r")
    m = np.load("data/glosofarid/npy/solar_2023_global_mask.npy", mmap_mode="r")
    n = len(x) * 16
    tx = np.lib.format.open_memmap(TILES_X + ".tmp", mode="w+", dtype=np.uint16, shape=(n, 13, TILE, TILE))
    tm = np.lib.format.open_memmap(TILES_MASK + ".tmp", mode="w+", dtype=np.uint8, shape=(n, TILE, TILE))
    for i in range(0, len(x), 128):
        a, b = np.asarray(x[i:i + 128]), np.asarray(m[i:i + 128])
        k = len(a)
        tx[i * 16:(i + k) * 16] = a.reshape(k, 13, 4, 64, 4, 64).transpose(0, 2, 4, 1, 3, 5).reshape(k * 16, 13, 64, 64)
        tm[i * 16:(i + k) * 16] = b.reshape(k, 4, 64, 4, 64).transpose(0, 1, 3, 2, 4).reshape(k * 16, 64, 64)
    tx.flush()
    tm.flush()
    del tx, tm
    os.replace(TILES_X + ".tmp", TILES_X)
    os.replace(TILES_MASK + ".tmp", TILES_MASK)


def band_stats():
    """Per-band mean/std of reflectance (/10,000) over GloSoFarID TRAIN tiles of split seed 0."""
    if os.path.exists(BAND_STATS):
        d = np.load(BAND_STATS)
        return d["mean"], d["std"]
    build_tile_cache()
    x = np.load(TILES_X, mmap_mode="r")
    tr = np.where(load_splits(0)["split"] == 0)[0]
    s = np.zeros(13)
    s2 = np.zeros(13)
    n = 0
    for i in range(0, len(tr), 2048):
        a = np.asarray(x[tr[i:i + 2048]], dtype=np.float64) / 10000.0
        s += a.sum(axis=(0, 2, 3))
        s2 += (a ** 2).sum(axis=(0, 2, 3))
        n += a.shape[0] * a.shape[2] * a.shape[3]
    mean = s / n
    std = np.sqrt(s2 / n - mean ** 2)
    np.savez(BAND_STATS, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def position_prior(sp, cells=None):
    """Location prior from TRAIN tiles: mean mask (64x64) and mean 4x4 cell occupancy per tile position,
    over positive tiles only (the spatial tasks ask 'where', given panels are present)."""
    m = np.load(TILES_MASK, mmap_mode="r")
    mask_prior = np.zeros((16, TILE, TILE), np.float32)
    cell_prior = np.zeros((16, 16), np.float32)
    for k in range(16):
        rows = np.where((sp["split"] == 0) & (sp["y"] == 1) & (sp["tile_idx"] == k))[0]
        mask_prior[k] = np.asarray(m[rows], dtype=np.float32).mean(0)
        if cells is not None:
            cell_prior[k] = cells[rows].mean(0)
    return mask_prior, cell_prior


# ---------------------------------------------------------------- statistics

def boot_stat(stat, groups, rng, n=1000):
    """Bootstrap distribution of stat(idx), resampling groups (cluster bootstrap)."""
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        try:
            vals.append(stat(idx))
        except ValueError:  # resample with a single class
            continue
    return np.asarray(vals)


def paired_test(metric, pred_a, pred_b, y, groups, rng, n=1000, alternative="two-sided"):
    """Difference metric(a) - metric(b) on the same rows, cluster bootstrap CI and p-value.

    p-value from the bootstrap distribution of the difference (percentile method):
      greater:   P(diff <= 0)
      two-sided: 2 * min(P(diff <= 0), P(diff >= 0))
    """
    full = np.arange(len(y))
    diff = metric(y, pred_a, full) - metric(y, pred_b, full)
    d = boot_stat(lambda idx: metric(y, pred_a, idx) - metric(y, pred_b, idx), groups, rng, n)
    lo, hi = np.percentile(d, [2.5, 97.5])
    le, ge = (d <= 0).mean(), (d >= 0).mean()
    p = le if alternative == "greater" else min(1.0, 2 * min(le, ge))
    p = max(p, 1.0 / (len(d) + 1))  # bootstrap resolution
    return float(diff), float(lo), float(hi), float(p)


def holm(pvals, alpha=0.05):
    """Holm step-down: returns adjusted p-values and reject flags, in input order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj, adj <= alpha


def auc_metric(y, s, idx):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y[idx], s[idx])


def acc_metric(y, pred, idx):
    return float((pred[idx] == y[idx]).mean())


def macro_ap_metric(y, s, idx):
    """Macro average precision over labels present in idx (multi-label)."""
    from sklearn.metrics import average_precision_score
    yy, ss = y[idx], s[idx]
    keep = (yy.sum(0) > 0) & (yy.sum(0) < len(yy))
    return float(average_precision_score(yy[:, keep], ss[:, keep], average="macro"))
