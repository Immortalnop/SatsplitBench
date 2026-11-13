"""Phase 6 analysis: the pre-registered localisation tests (results/preregistration.md, Phase 6 section).

Inputs: results/phase6/<encoder>/<variant>/pertile.npz (scripts/localise.py); for the secondary
comparisons, results/phase5/gan/<encoder>/<variant>/pertile.npz and results/phase4b/analysis/best.csv.

Outputs (results/phase6/analysis/)
  summary.csv     every run: mask IoU, location-prior IoU, detection AUC, with 95% CIs
  primary.csv     8 primary tests at full size (vs location prior, vs pixel-statistics control), Holm
  secondary.csv   the same along the compression curve and for the random control; vs the Phase 5 GAN
  detection.csv   detection AUC vs the Phase 4b best attacker (descriptive)

Usage
  python scripts/analysis6.py
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402

ROOT = "results/phase6"
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


def load(root):
    runs = {}
    if not os.path.isdir(root):
        return runs
    for enc in sorted(os.listdir(root)):
        if not os.path.isdir(os.path.join(root, enc)) or enc == "analysis":
            continue
        for var in sorted(os.listdir(os.path.join(root, enc))):
            f = os.path.join(root, enc, var, "pertile.npz")
            if os.path.exists(f):
                runs[(enc, var)] = dict(np.load(f))
    return runs


def iou_test(a_iou, b_iou, pos_rows, sp):
    g = C.patch_groups(sp["patch_id"][pos_rows])
    d = C.boot_stat(lambda idx: a_iou[idx].mean() - b_iou[idx].mean(), g, np.random.default_rng(RNG_SEED))
    return (float(a_iou.mean() - b_iou.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)),
            float(max((d <= 0).mean(), 1 / (len(d) + 1))))


def main():
    sp = C.load_splits(0)
    runs = load(ROOT)
    if not runs:
        raise SystemExit("No Phase 6 runs found")
    print(f"{len(runs)} localisation runs")

    rows = []
    for (enc, var), r in runs.items():
        pos_rows = r["rows"][r["y"] == 1]
        g_pos, g_all = C.patch_groups(sp["patch_id"][pos_rows]), C.patch_groups(sp["patch_id"][r["rows"]])
        rng = lambda: np.random.default_rng(RNG_SEED)
        ci_iou = np.percentile(C.boot_stat(lambda i: r["iou_positive"][i].mean(), g_pos, rng()), [2.5, 97.5])
        auc = C.auc_metric(r["y"], r["detection_score"], np.arange(len(r["y"])))
        ci_auc = np.percentile(C.boot_stat(lambda i: C.auc_metric(r["y"], r["detection_score"], i), g_all, rng()),
                               [2.5, 97.5])
        rows.append({"encoder": enc, "variant": var, "mean_iou_positive": round(float(r["iou_positive"].mean()), 4),
                     "iou_ci": f"[{ci_iou[0]:.4f}, {ci_iou[1]:.4f}]",
                     "prior_iou": round(float(r["prior_iou_positive"].mean()), 4),
                     "detection_auc": round(auc, 4), "auc_ci": f"[{ci_auc[0]:.4f}, {ci_auc[1]:.4f}]"})
    write_csv(os.path.join(OUT, "summary.csv"), rows)

    prim, sec = [], []
    ctrl = runs.get(("pixelstats", "full"))
    gan = load("results/phase5/gan")
    for (enc, var), r in sorted(runs.items()):
        if enc == "pixelstats":
            continue
        pos_rows = r["rows"][r["y"] == 1]
        tests = [("spatial_vs_location_prior", "location_prior", r["prior_iou_positive"])]
        if ctrl is not None and np.array_equal(ctrl["rows"], r["rows"]):
            tests.append(("learned_spatial_vs_pixelstats", "pixelstats", ctrl["iou_positive"]))
        primary = var == "full" and enc in C.AUDITED
        for name, vs, b in tests:
            diff, lo, hi, p = iou_test(r["iou_positive"], b, pos_rows, sp)
            (prim if primary else sec).append({"encoder": enc, "variant": var, "test": name, "vs": vs,
                                               "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                                               "p": round(p, 4)})
        gr = gan.get((enc, var))
        if gr is not None and np.array_equal(gr["rows"], r["rows"]):
            diff, lo, hi, p = iou_test(r["iou_positive"], gr["iou_positive"], pos_rows, sp)
            sec.append({"encoder": enc, "variant": var, "test": "localisation_minus_gan_inverter", "vs": "phase5_gan",
                        "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 4)})
    if prim:
        adj, rej = C.holm([r["p"] for r in prim])
        for r, a, j in zip(prim, adj, rej):
            r.update({"p_holm": round(float(a), 4), "significant_holm": bool(j)})
        if len(prim) != 8:
            print(f"  WARNING: {len(prim)} primary tests found, 8 pre-registered (some runs missing)")
    write_csv(os.path.join(OUT, "primary.csv"), prim)
    write_csv(os.path.join(OUT, "secondary.csv"), sec)

    best = {}
    f = "results/phase4b/analysis/best.csv"
    if os.path.exists(f):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if r["split_seed"] == "0" and r["task"] == "leakage":
                    best[(r["encoder"], r["variant"])] = r
    det = [{"encoder": e, "variant": v, "localisation_detection_auc": row["detection_auc"],
            "phase4b_best_auc": best[(e, v)]["test"] if (e, v) in best else "",
            "phase4b_attacker": best[(e, v)]["attacker"] if (e, v) in best else ""}
           for row in rows for e, v in [(row["encoder"], row["variant"])]]
    write_csv(os.path.join(OUT, "detection.csv"), det)


if __name__ == "__main__":
    main()
