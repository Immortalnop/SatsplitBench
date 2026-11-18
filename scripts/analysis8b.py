"""Phase 8b analysis: split points inside the encoders (tests pre-registered in results/preregistration.md,
section "Phase 8b split-point settings").

Inputs
  results/phase8/runs8b.csv + results/preds/phase8b/seed0/<encoder>@<split>/<form>/<task>_deep_mlp.npz
  results/phase8/compute.csv (8a; optional, adds bytes and FLOPs to the tables)
  Phase 4b predictions of the pixel-statistics control (results/preds/phase4b/seed0/pixelstats/full/)

Primary tests (Holm over 8, alpha 0.05; paired cluster bootstrap over test patches, 1,000 resamples, one-sided)
  P8b.1-4  depth adds presence leakage    AUC(gap, final split) - AUC(gap, earliest split) > 0
           (earliest = layer1 / block3; final = layer4 / block12), one test per audited encoder
  P8b.5-8  early splits leak location      macro AP(map4 spatial, earliest) - macro AP(map4 spatial, final) > 0
Verdict per family of 4: supported (>= 3 significant), not supported (<= 1), mixed (2).
A missing input counts as p = 1 in the Holm family and is reported as not run.

Secondary (exploratory, no multiplicity correction; --boot-secondary resamples, default 300)
  cells.csv        every encoder x split x form x task: test score, 95% CI, dims, payload bytes, head FLOPs
  vs_controls.csv  leakage minus pixel statistics (gap leakage, and leakage_incr minus pixelstats) per split;
                   spatial grid IoU minus the location prior per split (map4)
  frontier.csv     per encoder: splits not dominated on (int8 map bytes, head FLOPs, -BigEarthNet AP),
                   with their leakage
  split_curves.png leakage, spatial AP and BigEarthNet AP against split, one panel per measure

Outputs: results/phase8/analysis/{primary,cells,vs_controls,frontier}.csv, README.json, split_curves.png

Usage
  python scripts/analysis8b.py
  python scripts/analysis8b.py --boot 100 --out /tmp/x      # code test
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis4b as B  # noqa: E402  (imports lightgbm before torch)
import common as C  # noqa: E402

import numpy as np  # noqa: E402

import splitpoints as S  # noqa: E402

RUNS = "results/phase8/runs8b.csv"
PREDS = "results/preds/phase8b/seed0"
COMPUTE = "results/phase8/compute.csv"
ENCODERS = C.AUDITED + ["resnet18_random"]
CELLS = [("gap", "leakage"), ("gap", "leakage_incr"), ("gap", "utility_eurosat"), ("gap", "utility_ben"),
         ("map4", "spatial"), ("map4", "leakage")]
RNG_SEED = 0


def rng():
    return np.random.default_rng(RNG_SEED)


def load_cell(enc, split, form, task, runs):
    r = runs.get((f"{enc}@{split}", form, task))
    f = os.path.join(PREDS, f"{enc}@{split}", form, f"{task}_deep_mlp.npz")
    if r is None or not os.path.exists(f):
        return None
    d = np.load(f)
    test = float(B.test_metric(task, d["y"], d["scores"], np.arange(len(d["y"]))))  # recomputed from the predictions
    return {"val": float(r["val_score"]), "test": test, "csv_test": float(r["test_score"]), "dims": int(r["dims"]),
            "scores": d["scores"], "y": d["y"], "rows": d["rows"], "val_scores": d["val_scores"],
            "val_y": d["val_y"]}


def groups(task, cell, sp):
    return B.groups_for(task, cell, sp)


def paired(task, a, b, sp, n, alternative):
    assert np.array_equal(a["rows"], b["rows"]), "test rows differ"
    metric = lambda y, s, idx: B.test_metric(task, y, s, idx)  # noqa: E731
    return C.paired_test(metric, a["scores"], b["scores"], a["y"], groups(task, a, sp), rng(), n, alternative)


def primary(runs, sp, n):
    out, pvals = [], []
    for fam, form, task, sign in (("depth adds presence leakage", "gap", "leakage", "final - earliest"),
                                  ("early splits leak location", "map4", "spatial", "earliest - final")):
        for enc in C.AUDITED:
            sp_names = S.SPLITS[S.ARCH[enc]]
            first, last = sp_names[0], sp_names[-1]
            a_name, b_name = (last, first) if sign == "final - earliest" else (first, last)
            a, b = load_cell(enc, a_name, form, task, runs), load_cell(enc, b_name, form, task, runs)
            pid = f"P8b.{len(out) + 1}"
            if a is None or b is None:
                print(f"  {pid} {enc}: missing inputs, counted as p = 1")
                out.append({"id": pid, "family": fam, "encoder": enc, "measure": f"{task} ({form})",
                            "hypothesis": f"{sign} > 0", "status": "not run"})
                pvals.append(1.0)
                continue
            diff, lo, hi, p = paired(task, a, b, sp, n, "greater")
            out.append({"id": pid, "family": fam, "encoder": enc, "measure": f"{task} ({form})",
                        "a": f"{enc}@{a_name}", "b": f"{enc}@{b_name}", "score_a": round(a["test"], 4),
                        "score_b": round(b["test"], 4), "diff": round(diff, 4), "ci_lo": round(lo, 4),
                        "ci_hi": round(hi, 4), "hypothesis": f"{sign} > 0", "p": round(p, 4), "status": "run"})
            pvals.append(p)
    adj, rej = C.holm(pvals)
    for r, pa, rj in zip(out, adj, rej):
        r["p_holm"], r["significant_holm"] = round(float(pa), 4), bool(rj) and r["status"] == "run"
    verdicts = {}
    for fam in dict.fromkeys(r["family"] for r in out):
        k = sum(r["significant_holm"] for r in out if r["family"] == fam)
        verdicts[fam] = f"supported ({k}/4)" if k >= 3 else f"not supported ({k}/4)" if k <= 1 else f"mixed ({k}/4)"
    return out, verdicts


def ci(task, c, sp, n):
    """95% cluster-bootstrap CI of one cell's test metric (n resamples; Phase 4b's B.ci always uses 1,000)."""
    d = C.boot_stat(lambda idx: B.test_metric(task, c["y"], c["scores"], idx), groups(task, c, sp), rng(), n)
    return np.percentile(d, [2.5, 97.5])


def compute_table():
    if not os.path.exists(COMPUTE):
        return {}
    return {(r["arch"], r["split"]): r for r in csv.DictReader(open(COMPUTE))}


def cells(runs, sp, n, comp):
    out = []
    for enc in ENCODERS:
        arch = S.ARCH[enc]
        for k, split in enumerate(S.SPLITS[arch]):
            c8a = comp.get((arch, split), {})
            for form, task in CELLS:
                c = load_cell(enc, split, form, task, runs)
                if c is None:
                    continue
                lo, hi = ci(task, c, sp, n) if n else (np.nan, np.nan)
                out.append({"encoder": enc, "split": split, "split_index": k + 1, "form": form, "task": task,
                            "metric": "",
                            "val": round(c["val"], 4), "test": round(c["test"], 4), "ci_lo": round(float(lo), 4),
                            "ci_hi": round(float(hi), 4), "dims": c["dims"],
                            "form_fp32_bytes": c["dims"] * 4,
                            "map_int8_bytes": c8a.get("map_int8_bytes", ""),
                            "head_flops": c8a.get("head_flops", ""),
                            "head_flops_share": c8a.get("head_flops_share", "")})
    for r in out:
        r["metric"] = {"leakage": "roc_auc", "leakage_incr": "roc_auc", "spatial": "macro_ap",
                       "utility_eurosat": "accuracy", "utility_ben": "macro_ap"}[r["task"]]
    return out


def pixelstats_leakage():
    f = os.path.join(B.P4B_PREDS, "seed0", "pixelstats", "full", "leakage_deep_mlp.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def vs_controls(runs, sp, n):
    import attack4b
    out = []
    px = pixelstats_leakage()
    cl = attack4b.cell_labels()
    _, prior = C.position_prior(sp, cl)
    va = np.where((sp["split"] == 1) & (sp["y"] == 1))[0]
    pthr = B.best_threshold(prior[sp["tile_idx"][va]], cl[va])
    for enc in ENCODERS:
        for split in S.SPLITS[S.ARCH[enc]]:
            for task in ("leakage", "leakage_incr"):
                c = load_cell(enc, split, "gap", task, runs)
                if c is None or px is None:
                    continue
                pos = {r: i for i, r in enumerate(px["rows"])}
                ps = px["scores"][[pos[r] for r in c["rows"]]]
                diff, lo, hi, p = C.paired_test(C.auc_metric, c["scores"], ps, c["y"],
                                                C.patch_groups(sp["patch_id"][c["rows"]]), rng(), n, "two-sided")
                out.append({"encoder": enc, "split": split, "comparison": f"{task} (gap) - pixelstats leakage",
                            "diff": round(diff, 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 4)})
            c = load_cell(enc, split, "map4", "spatial", runs)
            if c is None:
                continue
            thr = B.best_threshold(c["val_scores"], c["val_y"])
            iou = B.grid_iou(c["scores"], c["y"], thr)
            piou = B.grid_iou(prior[sp["tile_idx"][c["rows"]]], c["y"], pthr)
            g = C.patch_groups(sp["patch_id"][c["rows"]])
            d = C.boot_stat(lambda idx: iou[idx].mean() - piou[idx].mean(), g, rng(), n)
            lo, hi = np.percentile(d, [2.5, 97.5])
            out.append({"encoder": enc, "split": split, "comparison": "spatial grid IoU (map4) - location prior",
                        "score": round(float(iou.mean()), 4), "reference": round(float(piou.mean()), 4),
                        "diff": round(float(iou.mean() - piou.mean()), 4), "ci_lo": round(float(lo), 4),
                        "ci_hi": round(float(hi), 4), "p": round(float(max((d <= 0).mean(), 1 / (len(d) + 1))), 4)})
    return out


def frontier(cell_rows):
    """Per encoder: splits not dominated on (map int8 bytes, head FLOPs) at equal or better BigEarthNet AP."""
    out = []
    by = {}
    for r in cell_rows:
        by.setdefault((r["encoder"], r["split"]), {})[(r["form"], r["task"])] = r
    for enc in ENCODERS:
        pts = []
        for split in S.SPLITS[S.ARCH[enc]]:
            d = by.get((enc, split), {})
            u = d.get(("gap", "utility_ben"))
            if u is None or u["map_int8_bytes"] == "":
                continue
            pts.append({"encoder": enc, "split": split, "map_int8_bytes": int(u["map_int8_bytes"]),
                        "head_flops": int(u["head_flops"]), "utility_ben": u["test"],
                        "leakage_gap": d.get(("gap", "leakage"), {}).get("test", ""),
                        "spatial_map4": d.get(("map4", "spatial"), {}).get("test", "")})
        for p in pts:
            p["pareto"] = not any(q is not p and q["map_int8_bytes"] <= p["map_int8_bytes"]
                                  and q["head_flops"] <= p["head_flops"] and q["utility_ben"] >= p["utility_ben"]
                                  and (q["map_int8_bytes"], q["head_flops"], q["utility_ben"]) !=
                                  (p["map_int8_bytes"], p["head_flops"], p["utility_ben"]) for q in pts)
        out += pts
    return out


def figure(cell_rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib missing: no figure")
        return
    panels = [("gap", "leakage", "presence leakage AUC (gap)"), ("map4", "spatial", "spatial macro AP (map4)"),
              ("gap", "utility_ben", "BigEarthNet macro AP (gap)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (form, task, title) in zip(axes, panels):
        for enc in ENCODERS:
            pts = sorted((r["split_index"], r["test"], r["ci_lo"], r["ci_hi"]) for r in cell_rows
                         if r["encoder"] == enc and r["form"] == form and r["task"] == task)
            if not pts:
                continue
            x, y, lo, hi = map(np.array, zip(*pts))
            ax.errorbar(x, y, yerr=[y - lo, hi - y], marker="o", capsize=2, label=enc,
                        linestyle="--" if enc == "resnet18_random" else "-")
        ax.set_title(title, fontsize=10)
        ax.set_xticks([1, 2, 3, 4], ["1st", "2nd", "3rd", "final"])
        ax.set_xlabel("split point (stage / block group)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"  wrote {path}")


def write_csv(path, rows):
    if not rows:
        print(f"  {path}: no rows")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {path} ({len(rows)} rows)")


def main():
    global RUNS, PREDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000, help="resamples for the 8 primary tests (pre-registered)")
    ap.add_argument("--boot-secondary", type=int, default=300, help="resamples for exploratory CIs")
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--preds", default=PREDS)
    ap.add_argument("--out", default="results/phase8/analysis")
    args = ap.parse_args()
    RUNS, PREDS = args.runs, args.preds
    if not os.path.exists(RUNS):
        raise SystemExit(f"Missing {RUNS}: run the 8b attackers first (scripts/run_phase8.sh 8b)")
    os.makedirs(args.out, exist_ok=True)
    runs = {(r["encoder"], r["variant"], r["task"]): r for r in csv.DictReader(open(RUNS))
            if r["attacker"] == "deep_mlp" and r["split_seed"] == "0"}
    sp = C.load_splits(0)

    prim, verdicts = primary(runs, sp, args.boot)
    write_csv(os.path.join(args.out, "primary.csv"), prim)
    for r in prim:
        print(f"  {r['id']} {r['encoder']:>15} {r['measure']:>16}: "
              + (f"diff {r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] p_holm {r['p_holm']}"
                 if r["status"] == "run" else "not run"))
    for fam, v in verdicts.items():
        print(f"  verdict, {fam}: {v}")
    cr = cells(runs, sp, args.boot_secondary, compute_table())
    write_csv(os.path.join(args.out, "cells.csv"), cr)
    write_csv(os.path.join(args.out, "vs_controls.csv"), vs_controls(runs, sp, args.boot_secondary))
    write_csv(os.path.join(args.out, "frontier.csv"), frontier(cr))
    figure(cr, os.path.join(args.out, "split_curves.png"))
    json.dump({"verdicts": verdicts, "bootstrap_primary": args.boot, "bootstrap_secondary": args.boot_secondary, "attacker": "deep_mlp (hyperparameters on val)",
               "holm": "8 primary tests, alpha 0.05", "earliest": "layer1 / block3", "final": "layer4 / block12",
               "compute_csv": os.path.exists(COMPUTE)}, open(os.path.join(args.out, "README.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
