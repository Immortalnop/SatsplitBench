"""Phase 5 analysis: the pre-registered inversion tests (results/preregistration.md, Phase 5 section).

Inputs: results/phase5/<method>/<encoder>/<variant>/pertile.npz from scripts/invert.py, the segmenter
(results/phase5/segmenter.json) and, for RQ3, results/phase4b/analysis/best.csv.

Outputs (results/phase5/analysis/)
  summary.csv     every run: pixel metrics, detection AUC, mask IoU, each with a 95% CI
  primary.csv     8 primary tests (learned and spatial leakage through GAN inversion, full size), Holm
  secondary.csv   the same at 16 dims, kNN, random control; GAN vs kNN
  rq3.csv         Kendall tau between pixel metrics and object measures, exact permutation p

Usage
  python scripts/analysis5.py
"""
import csv
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "results/phase5"
OUT = os.path.join(ROOT, "analysis")
RNG_SEED = 0


def write_csv(path, rows):
    if not rows:
        print(f"  (nothing to write for {path})")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def load_runs():
    runs = {}
    for method in ("mean", "knn", "gan"):
        d = os.path.join(ROOT, method)
        if not os.path.isdir(d):
            continue
        for enc in sorted(os.listdir(d)):
            for var in sorted(os.listdir(os.path.join(d, enc))):
                f = os.path.join(d, enc, var, "pertile.npz")
                if os.path.exists(f):
                    runs[(method, enc, var)] = dict(np.load(f))
    return runs


def prior_iou_per_tile(sp, rows):
    """Location prior IoU on positive tiles among `rows` (threshold chosen on val, as in segmenter.py)."""
    import segmenter as S
    info = json.load(open(S.INFO))
    mask_prior, _ = C.position_prior(sp)
    m = np.load(C.TILES_MASK, mmap_mode="r")
    pos = rows[sp["y"][rows] == 1]
    order = np.argsort(pos)
    iou = np.empty(len(pos), np.float32)
    iou[order] = S.tile_iou(mask_prior[sp["tile_idx"][pos[order]]], np.asarray(m[pos[order]]),
                            info["test_location_prior"]["threshold"])
    return iou


def boot_ci(stat, groups):
    d = C.boot_stat(stat, groups, np.random.default_rng(RNG_SEED))
    return np.percentile(d, [2.5, 97.5])


def paired(a, b, metric, sp):
    """a, b: per-tile dicts on the same test rows; metric 'detection' (AUC, all tiles) or 'iou' (positives)."""
    if metric == "detection":
        assert np.array_equal(a["rows"], b["rows"])
        g = C.patch_groups(sp["patch_id"][a["rows"]])
        return C.paired_test(C.auc_metric, a["detection_score"], b["detection_score"], a["y"], g,
                             np.random.default_rng(RNG_SEED), alternative="greater")
    pos_rows = a["rows"][a["y"] == 1]
    g = C.patch_groups(sp["patch_id"][pos_rows])
    ia, ib = a["iou_positive"], b["iou_positive"]
    diff = float(ia.mean() - ib.mean())
    d = C.boot_stat(lambda idx: ia[idx].mean() - ib[idx].mean(), g, np.random.default_rng(RNG_SEED))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return diff, float(lo), float(hi), float(max((d <= 0).mean(), 1 / (len(d) + 1)))


def summary(runs, sp):
    rows = []
    for (method, enc, var), r in sorted(runs.items()):
        g_all = C.patch_groups(sp["patch_id"][r["rows"]])
        g_pos = C.patch_groups(sp["patch_id"][r["rows"][r["y"] == 1]])
        row = {"method": method, "encoder": enc, "variant": var, "n_test": len(r["rows"])}
        for k in ("l1", "ssim", "lpips", "sam"):
            lo, hi = boot_ci(lambda idx, v=r[k]: v[idx].mean(), g_all)
            row.update({k: round(float(r[k].mean()), 4), f"{k}_ci": f"[{lo:.4f}, {hi:.4f}]"})
        auc = C.auc_metric(r["y"], r["detection_score"], np.arange(len(r["y"])))
        lo, hi = boot_ci(lambda idx: C.auc_metric(r["y"], r["detection_score"], idx), g_all)
        row.update({"detection_auc": round(auc, 4), "detection_auc_ci": f"[{lo:.4f}, {hi:.4f}]"})
        lo, hi = boot_ci(lambda idx: r["iou_positive"][idx].mean(), g_pos)
        row.update({"mean_iou_positive": round(float(r["iou_positive"].mean()), 4),
                    "mean_iou_positive_ci": f"[{lo:.4f}, {hi:.4f}]"})
        rows.append(row)
    write_csv(os.path.join(OUT, "summary.csv"), rows)


def comparisons(runs, sp):
    prior_cache = {}
    prim, sec = [], []
    for method in ("gan", "knn"):
        ctrl = runs.get((method, "pixelstats", "full"))
        for enc in C.AUDITED + ["resnet18_random"]:
            for var in ("full", "pca16"):
                r = runs.get((method, enc, var))
                if r is None:
                    continue
                key = r["rows"].tobytes()
                if key not in prior_cache:
                    prior_cache[key] = prior_iou_per_tile(sp, r["rows"])
                prior = prior_cache[key]
                primary = method == "gan" and var == "full" and enc in C.AUDITED
                tests = []
                if ctrl is not None:
                    tests.append(("learned_detection", "pixelstats", paired(r, ctrl, "detection", sp)))
                tests.append(("spatial_iou_vs_prior", "location_prior",
                              paired(r, {"iou_positive": prior, "rows": r["rows"], "y": r["y"]}, "iou", sp)))
                for name, vs, (diff, lo, hi, p) in tests:
                    row = {"method": method, "encoder": enc, "variant": var, "test": name, "vs": vs,
                           "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 4)}
                    (prim if primary else sec).append(row)
    if prim:
        adj, rej = C.holm([r["p"] for r in prim])
        for r, a, j in zip(prim, adj, rej):
            r.update({"p_holm": round(float(a), 4), "significant_holm": bool(j)})
        if len(prim) != 8:
            print(f"  WARNING: {len(prim)} primary tests found, 8 pre-registered (some runs missing)")
    # strong vs weak inverter
    for enc in C.AUDITED + ["resnet18_random", "pixelstats"]:
        for var in ("full", "pca16"):
            a, b = runs.get(("gan", enc, var)), runs.get(("knn", enc, var))
            if a is None or b is None:
                continue
            for metric in ("detection", "iou"):
                diff, lo, hi, p = paired(a, b, metric, sp)
                sec.append({"method": "gan_vs_knn", "encoder": enc, "variant": var, "test": f"{metric}_gan_minus_knn",
                            "vs": "knn", "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                            "p": round(p, 4)})
    write_csv(os.path.join(OUT, "primary.csv"), prim)
    write_csv(os.path.join(OUT, "secondary.csv"), sec)


def kendall_exact(x, y):
    """Kendall tau-b and exact two-sided permutation p-value (all orderings for n <= 9, else 20,000 draws)."""
    from scipy.stats import kendalltau
    x, y = np.asarray(x), np.asarray(y)
    tau = kendalltau(x, y).statistic
    if len(x) <= 9:
        perms = (np.array(p) for p in itertools.permutations(range(len(y))))
        null = [kendalltau(x, y[p]).statistic for p in perms]
    else:
        rng = np.random.default_rng(RNG_SEED)
        null = [kendalltau(x, rng.permutation(y)).statistic for _ in range(20000)]
    null = np.asarray(null)
    return float(tau), float((np.abs(null) >= abs(tau) - 1e-12).mean())


def rq3(runs):
    best = {}
    f = "results/phase4b/analysis/best.csv"
    if os.path.exists(f):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["split_seed"] == "0" and r["task"] == "leakage":
                    best[(r["encoder"], r["variant"])] = float(r["test"])
    pts = []
    for (method, enc, var), r in sorted(runs.items()):
        if method != "gan" or enc in ("pixelstats",):
            continue
        pts.append({"encoder": enc, "variant": var, "ssim": float(r["ssim"].mean()), "lpips": float(r["lpips"].mean()),
                    "detection_auc": C.auc_metric(r["y"], r["detection_score"], np.arange(len(r["y"]))),
                    "mean_iou": float(r["iou_positive"].mean()), "probe_auc": best.get((enc, var))})
    if len(pts) < 4:
        print(f"  RQ3 needs at least 4 GAN runs, found {len(pts)}")
        return
    rows = []
    # LPIPS is a distance: lower = more similar, so its sign is flipped to read as "more leakage"
    for pm, sign in (("ssim", 1), ("lpips", -1)):
        for om in ("detection_auc", "mean_iou", "probe_auc"):
            keep = [p for p in pts if p[om] is not None]
            if len(keep) < 4:
                continue
            tau, p = kendall_exact([sign * k[pm] for k in keep], [k[om] for k in keep])
            rows.append({"pixel_metric": pm + (" (negated)" if sign < 0 else ""), "object_measure": om,
                         "n": len(keep), "kendall_tau": round(tau, 4), "p_exact": round(p, 4)})
    write_csv(os.path.join(OUT, "rq3.csv"), rows)
    write_csv(os.path.join(OUT, "rq3_points.csv"), [{k: (round(v, 4) if isinstance(v, float) else v)
                                                     for k, v in p.items()} for p in pts])


def main():
    sp = C.load_splits(0)
    runs = load_runs()
    if not runs:
        raise SystemExit("No Phase 5 runs found")
    print(f"{len(runs)} inversion runs")
    summary(runs, sp)
    comparisons(runs, sp)
    rq3(runs)


if __name__ == "__main__":
    main()
