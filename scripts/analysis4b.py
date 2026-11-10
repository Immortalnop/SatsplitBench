"""Phase 4b analysis: the pre-registered tests (results/preregistration.md), from saved predictions.

Leakage estimate: for each (split seed, encoder, variant, task) the attacker with the best VALIDATION
score is selected; its test score is the estimate. Candidates: deep_mlp, lightgbm, svm_rbf (Phase 4b)
and, for split seed 0, Phase 4's linear and MLP probes (results/probes_v2.csv + results/preds/phase4).

Outputs (results/phase4b/analysis/)
  all_attackers.csv    every attacker's val and test score, with the selected one marked
  best.csv             selected attacker per cell, test score and 95% CI
  primary.csv          the 6 primary tests with Holm-adjusted p-values, plus split-seed replications
  per_class.csv        leakage AUC within each land-cover pseudo-class                (secondary)
  spatial.csv          4x4 grid IoU vs the position-conditioned location prior         (secondary)
  quantisation.csv     utility and leakage at 32/8/4/2/1 bits                          (secondary)
  matched_utility.csv  H2 pairs compared at equal BigEarthNet mAP                      (secondary)
  upper_bound.csv      image-level ResNet-18 vs the best embedding                     (secondary)

CIs and p-values: bootstrap with 1,000 resamples; GloSoFarID resamples whole patches (tiles of one
patch are correlated), EuroSAT and BigEarthNet resample images.

Usage
  python scripts/analysis4b.py
"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attackers as A  # noqa: E402  (imports lightgbm before torch)
import common as C  # noqa: E402

import numpy as np  # noqa: E402

OUT = "results/phase4b/analysis"
RUNS = "results/phase4b/runs.csv"
P4_CSV = "results/probes_v2.csv"
P4_PREDS = "results/preds/phase4"
P4B_PREDS = "results/preds/phase4b"
TASK_TYPE = {"leakage": "binary", "leakage_incr": "binary", "spatial": "multilabel",
             "utility_eurosat": "multiclass", "utility_ben": "multilabel"}
H2_PAIRS = [("resnet18_moco", "vits16_moco"), ("resnet50_dino", "vits16_dino")]
RNG_SEED = 0


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def write_csv(path, rows):
    if not rows:
        print(f"  (nothing to write for {path})")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


# ---------------------------------------------------------------- candidates and selection

def test_metric(task, y, s, idx):
    if TASK_TYPE[task] == "binary":
        return C.auc_metric(y, s, idx)
    if TASK_TYPE[task] == "multiclass":
        pred = s.argmax(1) if s.ndim == 2 else s
        return C.acc_metric(y, pred, idx)
    return C.macro_ap_metric(y, s, idx)


def candidates(seed, enc, var, task, runs, p4):
    """All attackers for one cell: dicts with attacker, val, test, scores, y, rows (+ val_scores, val_y)."""
    out = []
    for r in runs:
        if (int(r["split_seed"]), r["encoder"], r["variant"], r["task"]) == (seed, enc, var, task):
            f = os.path.join(P4B_PREDS, f"seed{seed}", enc, var, f"{task}_{r['attacker']}.npz")
            if os.path.exists(f):
                d = np.load(f)
                out.append({"attacker": r["attacker"], "val": float(r["val_score"]), "test": float(r["test_score"]),
                            "scores": d["scores"], "y": d["y"], "rows": d["rows"],
                            "val_scores": d["val_scores"] if "val_scores" in d else None,
                            "val_y": d["val_y"] if "val_y" in d else None})
    p4task = {"leakage": "leakage", "utility_eurosat": "utility"}.get(task)
    if seed == 0 and p4task:
        for r in p4:
            if (r["encoder"], r["variant"], r["task"]) == (enc, var, p4task) and r.get("val_value"):
                f = os.path.join(P4_PREDS, enc, var, f"{p4task}_{r['probe']}.npz")
                if os.path.exists(f):
                    d = np.load(f)
                    s = d["pred"][0]
                    rows = np.where(C.load_splits(0)["split"] == 2)[0] if p4task == "leakage" \
                        else np.arange(len(d["y"]))
                    out.append({"attacker": f"phase4_{r['probe']}", "val": float(r["val_value"]),
                                "test": float(r["value"]) if r["probe"] == "linear" else
                                test_metric(task, d["y"], s, np.arange(len(s))),
                                "scores": s, "y": d["y"], "rows": rows, "val_scores": None, "val_y": None})
    return out


def select(cands):
    return max(cands, key=lambda c: c["val"]) if cands else None


def groups_for(task, cand, sp):
    if task in ("leakage", "leakage_incr", "spatial"):
        return C.patch_groups(sp["patch_id"][cand["rows"]])
    return [np.array([i]) for i in range(len(cand["y"]))]


def ci(task, cand, sp, rng):
    g = groups_for(task, cand, sp)
    d = C.boot_stat(lambda idx: test_metric(task, cand["y"], cand["scores"], idx), g, rng)
    return np.percentile(d, [2.5, 97.5])


# ---------------------------------------------------------------- analyses

def tables(runs, p4):
    cells = sorted({(int(r["split_seed"]), r["encoder"], r["variant"], r["task"]) for r in runs} |
                   {(0, r["encoder"], r["variant"], {"leakage": "leakage", "utility": "utility_eurosat"}[r["task"]])
                    for r in p4 if r.get("val_value")})
    all_rows, best_rows, best = [], [], {}
    for seed, enc, var, task in cells:
        cs = candidates(seed, enc, var, task, runs, p4)
        if not cs:
            continue
        b = select(cs)
        best[(seed, enc, var, task)] = b
        sp = C.load_splits(seed)
        lo, hi = ci(task, b, sp, np.random.default_rng(RNG_SEED))
        for c in cs:
            all_rows.append({"split_seed": seed, "encoder": enc, "variant": var, "task": task,
                             "attacker": c["attacker"], "val": round(c["val"], 4), "test": round(c["test"], 4),
                             "selected": c is b})
        best_rows.append({"split_seed": seed, "encoder": enc, "variant": var, "task": task,
                          "attacker": b["attacker"], "val": round(b["val"], 4), "test": round(b["test"], 4),
                          "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4), "n_candidates": len(cs)})
    write_csv(os.path.join(OUT, "all_attackers.csv"), all_rows)
    write_csv(os.path.join(OUT, "best.csv"), best_rows)
    return best


def aligned(a, b):
    """Reorder b's scores to a's test rows (both cover the same test tiles)."""
    if np.array_equal(a["rows"], b["rows"]):
        return b["scores"]
    pos = {r: i for i, r in enumerate(b["rows"])}
    return b["scores"][[pos[r] for r in a["rows"]]]


def primary(best):
    rows = []
    for seed in (0, 1, 2):
        sp = C.load_splits(seed) if os.path.exists(C.splits_path(seed)) else None
        if sp is None:
            continue
        tests = []
        for enc in C.AUDITED:
            a, b = best.get((seed, enc, "full", "leakage_incr")), best.get((seed, "pixelstats", "full", "leakage"))
            if a and b:
                tests.append(("incremental_leakage", enc, "pixelstats", a, b, "greater"))
        for e1, e2 in H2_PAIRS:
            a, b = best.get((seed, e2, "full", "leakage")), best.get((seed, e1, "full", "leakage"))
            if a and b:
                tests.append(("h2_architecture", e2, e1, a, b, "two-sided"))
        res = []
        for name, ea, eb, a, b, alt in tests:
            g = C.patch_groups(sp["patch_id"][a["rows"]])
            diff, lo, hi, p = C.paired_test(C.auc_metric, a["scores"], aligned(a, b), a["y"], g,
                                            np.random.default_rng(RNG_SEED), alternative=alt)
            res.append({"split_seed": seed, "test": name, "a": ea, "b": eb, "a_attacker": a["attacker"],
                        "b_attacker": b["attacker"], "auc_a": round(a["test"], 4), "auc_b": round(b["test"], 4),
                        "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                        "alternative": alt, "p": round(p, 4)})
        if res and seed == 0:  # Holm across the 6 pre-registered primary tests, on the original split
            adj, rej = C.holm([r["p"] for r in res])
            for r, pa, rj in zip(res, adj, rej):
                r.update({"p_holm": round(float(pa), 4), "significant_holm": bool(rj), "role": "primary"})
        else:
            for r in res:
                r.update({"p_holm": "", "significant_holm": "", "role": "replication"})
        rows += res
    # utility context for the H2 pairs
    for enc in C.AUDITED:
        for task in ("utility_ben", "utility_eurosat"):
            b = best.get((0, enc, "full", task))
            if b:
                rows.append({"split_seed": 0, "test": f"context_{task}", "a": enc, "b": "", "a_attacker": b["attacker"],
                             "b_attacker": "", "auc_a": round(b["test"], 4), "auc_b": "", "diff": "", "ci_lo": "",
                             "ci_hi": "", "alternative": "", "p": "", "p_holm": "", "significant_holm": "",
                             "role": "context"})
    write_csv(os.path.join(OUT, "primary.csv"), rows)


def landcover_pseudo_labels(runs):
    """GloSoFarID tiles labelled with EuroSAT classes by the best (val) EuroSAT attacker on ResNet50 DINO."""
    f = "data/glosofarid/npy/landcover_pseudo.npy"
    if os.path.exists(f):
        return np.load(f)
    cs = [r for r in runs if (r["encoder"], r["variant"], r["task"], r["split_seed"]) ==
          ("resnet50_dino", "full", "utility_eurosat", "0")]
    if not cs:
        return None
    att = max(cs, key=lambda r: float(r["val_score"]))["attacker"]
    print(f"  pseudo-labelling GloSoFarID land cover with {att} on resnet50_dino/full")
    e, ey = C.load_variant("resnet50_dino", "full"), C.eurosat_labels()
    r = A.run_attacker(att, e["eurosat_train"], ey["train"], e["eurosat_val"], ey["val"], e["glosofarid"],
                       "multiclass", device="mps")
    lab = r["test_scores"].argmax(1).astype(np.int8)
    np.save(f, lab)
    return lab


def per_class(best, runs):
    lab = landcover_pseudo_labels(runs)
    if lab is None:
        return
    sp = C.load_splits(0)
    rows = []
    for enc in C.AUDITED + ["pixelstats"]:
        b = best.get((0, enc, "full", "leakage"))
        if not b:
            continue
        cls = lab[b["rows"]]
        for k, name in enumerate(C.EUROSAT_CLASSES):
            idx = np.where(cls == k)[0]
            npos = int(b["y"][idx].sum())
            if npos < 30 or len(idx) - npos < 30:
                continue
            sub = {**b, "scores": b["scores"][idx], "y": b["y"][idx], "rows": b["rows"][idx]}
            lo, hi = ci("leakage", sub, sp, np.random.default_rng(RNG_SEED))
            rows.append({"encoder": enc, "attacker": b["attacker"], "class": name, "n": len(idx), "n_pos": npos,
                         "auc": round(C.auc_metric(sub["y"], sub["scores"], np.arange(len(idx))), 4),
                         "ci_lo": round(float(lo), 4), "ci_hi": round(float(hi), 4)})
    write_csv(os.path.join(OUT, "per_class.csv"), rows)


def grid_iou(scores, y, thr):
    p = scores > thr
    t = y > 0
    inter, union = (p & t).sum(1), (p | t).sum(1)
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def best_threshold(scores, y):
    qs = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 37)))
    return max(qs, key=lambda t: grid_iou(scores, y, t).mean())


def spatial(best):
    import attack4b
    sp = C.load_splits(0)
    cells = attack4b.cell_labels()
    _, prior = C.position_prior(sp, cells)
    rows = []
    for (seed, enc, var, task), b in sorted(best.items()):
        if task != "spatial" or seed != 0 or b["val_scores"] is None:
            continue
        thr = best_threshold(b["val_scores"], b["val_y"])
        iou = grid_iou(b["scores"], b["y"], thr)
        va = np.where((sp["split"] == 1) & (sp["y"] == 1))[0]
        pthr = best_threshold(prior[sp["tile_idx"][va]], cells[va])
        piou = grid_iou(prior[sp["tile_idx"][b["rows"]]], b["y"], pthr)
        g = C.patch_groups(sp["patch_id"][b["rows"]])
        d = C.boot_stat(lambda idx: iou[idx].mean() - piou[idx].mean(), g, np.random.default_rng(RNG_SEED))
        rows.append({"encoder": enc, "variant": var, "attacker": b["attacker"], "macro_ap": round(b["test"], 4),
                     "grid_iou": round(float(iou.mean()), 4), "prior_grid_iou": round(float(piou.mean()), 4),
                     "diff": round(float(iou.mean() - piou.mean()), 4),
                     "ci_lo": round(float(np.percentile(d, 2.5)), 4), "ci_hi": round(float(np.percentile(d, 97.5)), 4),
                     "p_greater": round(float(max((d <= 0).mean(), 1 / (len(d) + 1))), 4)})
    write_csv(os.path.join(OUT, "spatial.csv"), rows)


def quantisation(p4):
    q = read_csv("results/probes_quant.csv")
    rows = []
    for r in p4 + q:
        var = r["variant"]
        if var.endswith("_int8"):
            continue
        level, bits = (var.rsplit("_q", 1)[0], int(var.rsplit("_q", 1)[1])) if "_q" in var else (var, 32)
        rows.append({"encoder": r["encoder"], "level": level, "bits": bits, "dims": r["dims"],
                     "bytes": float(r["dims"]) * bits / 8, "task": r["task"], "probe": r["probe"],
                     "test": r["value"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"]})
    write_csv(os.path.join(OUT, "quantisation.csv"), sorted(rows, key=lambda r: (r["encoder"], r["level"], -r["bits"])))


def matched_utility(best):
    rows = []
    for e1, e2 in H2_PAIRS:
        curves = {}
        for enc in (e1, e2):
            pts = [(best[(0, enc, v, "utility_ben")]["test"], best[(0, enc, v, "leakage")]["test"])
                   for v in C.FLOAT_VARIANTS if (0, enc, v, "utility_ben") in best and (0, enc, v, "leakage") in best]
            curves[enc] = sorted(pts)
        if min(len(c) for c in curves.values()) < 2:
            continue
        lo = max(c[0][0] for c in curves.values())
        hi = min(c[-1][0] for c in curves.values())
        for u in (np.linspace(lo, hi, 5) if hi > lo else []):
            l1 = np.interp(u, *zip(*curves[e1]))
            l2 = np.interp(u, *zip(*curves[e2]))
            rows.append({"pair": f"{e2} vs {e1}", "ben_map": round(float(u), 4), f"leak_a": round(float(l2), 4),
                         f"leak_b": round(float(l1), 4), "diff": round(float(l2 - l1), 4)})
    write_csv(os.path.join(OUT, "matched_utility.csv"), rows)


def upper_bound(best):
    ub = best.get((0, "image_resnet18", "raw", "leakage"))
    if not ub:
        return
    sp = C.load_splits(0)
    rows = []
    for enc in C.AUDITED + ["pixelstats"]:
        b = best.get((0, enc, "full", "leakage"))
        if not b:
            continue
        g = C.patch_groups(sp["patch_id"][ub["rows"]])
        diff, lo, hi, p = C.paired_test(C.auc_metric, ub["scores"], aligned(ub, b), ub["y"], g,
                                        np.random.default_rng(RNG_SEED))
        rows.append({"image_auc": round(ub["test"], 4), "encoder": enc, "embedding_auc": round(b["test"], 4),
                     "gap": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 4)})
    write_csv(os.path.join(OUT, "upper_bound.csv"), rows)


def main():
    import argparse
    global OUT, RUNS, P4B_PREDS, P4_CSV, P4_PREDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--preds", default=P4B_PREDS)
    ap.add_argument("--phase4-csv", default=P4_CSV)
    ap.add_argument("--phase4-preds", default=P4_PREDS)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    OUT, RUNS, P4B_PREDS, P4_CSV, P4_PREDS = args.out, args.runs, args.preds, args.phase4_csv, args.phase4_preds

    runs, p4 = read_csv(RUNS), read_csv(P4_CSV)
    if not runs:
        raise SystemExit(f"No Phase 4b runs yet ({RUNS})")
    print("tables")
    best = tables(runs, p4)
    print("primary tests")
    primary(best)
    print("per-class leakage")
    per_class(best, runs)
    print("spatial")
    spatial(best)
    print("quantisation")
    quantisation(p4)
    print("matched utility")
    matched_utility(best)
    print("upper bound")
    upper_bound(best)
    with open(os.path.join(OUT, "README.json"), "w") as fh:
        json.dump({"selection": "best validation score among attackers", "bootstrap": 1000,
                   "cluster": "GloSoFarID patch", "holm": "6 primary tests, split seed 0"}, fh, indent=2)


if __name__ == "__main__":
    main()
