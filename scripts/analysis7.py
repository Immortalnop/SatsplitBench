"""Phase 7 analysis: baseline vs defended own encoder (the tests are pre-registered in
results/preregistration.md, section "Phase 7 defence settings").

Both encoders went through exactly the audited encoders' pipeline, so every measure comes from files the
earlier phases already write:
  results/phase4b/runs.csv + results/preds/phase4b/...     Phase 4b attackers (deep MLP, LightGBM, RBF SVM)
  results/phase6/<enc>/full/pertile.npz                      Phase 6 localisation attack
  results/phase5/gan/<enc>/full/pertile.npz                  Phase 5 GAN inverter
  results/phase7/<enc>_train.json                            training log (adversary monitor)

Leakage estimate = test score of the attacker with the best VALIDATION score. Only the three Phase 4b
attackers are candidates for every encoder here (Phase 4's probes were never run on the own encoders), so
the baseline and the defended encoder are treated identically.

Primary tests (Holm over 5, alpha 0.05; paired cluster bootstrap over test patches / images, 1,000 resamples):
  P1  leakage AUC, Phase 4b best attacker, full size          baseline - defended > 0
  P2  detection AUC of the Phase 6 localisation attack        baseline - defended > 0
  P3  detection AUC of the Phase 5 GAN reconstructions        baseline - defended > 0
  P4  BigEarthNet macro AP, non-inferiority                   defended - baseline > -0.02
  P5  EuroSAT accuracy, non-inferiority                       defended - baseline > -0.01
Verdict (printed and written to README.json):
  works                 P1 significant and P4, P5 significant
  leakage cut at a cost P1 significant, P4 or P5 not significant
  no reduction          P1 not significant

Outputs (results/phase7/analysis/)
  summary.csv    best attacker per cell for the two encoders, the audited ViT-S/16s and the controls
  primary.csv    the 5 tests, Holm-adjusted
  secondary.csv  compression curve, learned excess over pixel statistics, localisation, inversion  (exploratory)
  training.csv   last-epoch training losses and the adversary's own accuracy
  README.json    verdict and definitions

Usage
  python scripts/analysis7.py
  python scripts/analysis7.py --baseline vits16_moco --defended vits16_dino --out /tmp/x --boot 100   # code test
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis4b as B  # noqa: E402  (imports lightgbm before torch)
import common as C  # noqa: E402

import numpy as np  # noqa: E402

MARGIN_BEN, MARGIN_EUROSAT = 0.02, 0.01
REFERENCE = ["vits16_moco", "vits16_dino", "pixelstats"]
VARIANTS = ["full", "pca256", "pca64", "pca16"]
TASKS = ["leakage", "leakage_incr", "spatial", "utility_eurosat", "utility_ben"]


def rng():
    return np.random.default_rng(B.RNG_SEED)


def best_cell(runs, enc, var, task):
    return B.select(B.candidates(0, enc, var, task, runs, []))


def pertile(root, enc):
    f = os.path.join(root, enc, "full", "pertile.npz")
    return dict(np.load(f)) if os.path.exists(f) else None


def acc_from_scores(y, s, idx):
    return C.acc_metric(y, s.argmax(1), idx)


def metric_for(task):
    return {"leakage": C.auc_metric, "leakage_incr": C.auc_metric, "utility_eurosat": acc_from_scores,
            "utility_ben": C.macro_ap_metric, "spatial": C.macro_ap_metric}[task]


def groups(task, cell, sp):
    return B.groups_for(task, cell, sp)


def pair_diff(metric, a, b, y, g, n):
    """Statistic = metric(a) - metric(b) on the same rows; bootstrap distribution over groups."""
    full = np.arange(len(y))
    diff = metric(y, a, full) - metric(y, b, full)
    d = C.boot_stat(lambda idx: metric(y, a, idx) - metric(y, b, idx), g, rng(), n)
    return float(diff), d


def superiority(diff, d, alternative):
    lo, hi = np.percentile(d, [2.5, 97.5])
    le, ge = (d <= 0).mean(), (d >= 0).mean()
    p = le if alternative == "greater" else min(1.0, 2 * min(le, ge))
    return float(lo), float(hi), float(max(p, 1.0 / (len(d) + 1)))


def non_inferiority(d, margin):
    """H1: true difference (defended - baseline) > -margin; p = P(bootstrap diff <= -margin)."""
    return float(max((d <= -margin).mean(), 1.0 / (len(d) + 1)))


def row(**kw):
    base = {"id": "", "test": "", "measure": "", "baseline": "", "defended": "", "diff": "", "ci_lo": "",
            "ci_hi": "", "hypothesis": "", "p": "", "p_holm": "", "significant_holm": "", "role": ""}
    base.update({k: (round(v, 4) if isinstance(v, float) else v) for k, v in kw.items()})
    return base


# ---------------------------------------------------------------- primary

def primary(runs, base, dfn, sp, n):
    rows, pvals = [], []

    def leak(pid, name, measure, y, sb, sd, rows_b, rows_d, hi_is_better=False):
        if sb is None or sd is None:
            print(f"  {pid}: missing inputs, counted as p = 1 in the Holm family")
            rows.append(row(id=pid, test=name, measure=measure, role="primary", hypothesis="MISSING"))
            pvals.append(1.0)
            return
        g = C.patch_groups(sp["patch_id"][rows_b])
        diff, d = pair_diff(C.auc_metric, sb, sd, y, g, n)
        lo, hi, p = superiority(diff, d, "greater")
        vb, vd = C.auc_metric(y, sb, np.arange(len(y))), C.auc_metric(y, sd, np.arange(len(y)))
        rows.append(row(id=pid, test=name, measure=measure, baseline=vb, defended=vd, diff=diff, ci_lo=lo,
                        ci_hi=hi, hypothesis="baseline - defended > 0", p=p, role="primary"))
        pvals.append(p)

    # P1
    cb, cd = best_cell(runs, base, "full", "leakage"), best_cell(runs, dfn, "full", "leakage")
    if cb and cd:
        leak("P1", "leakage_phase4b", "AUC, best attacker", cb["y"], cb["scores"], B.aligned(cb, cd),
             cb["rows"], cd["rows"])
        rows[-1]["hypothesis"] += f"  [attackers: {cb['attacker']} / {cd['attacker']}]"
    else:
        leak("P1", "leakage_phase4b", "AUC, best attacker", None, None, None, None, None)
    # P2, P3
    for pid, name, root in (("P2", "detection_localisation", "results/phase6"),
                            ("P3", "detection_gan_inversion", "results/phase5/gan")):
        tb, td = pertile(root, base), pertile(root, dfn)
        if tb is not None and td is not None and np.array_equal(tb["rows"], td["rows"]):
            leak(pid, name, "detection AUC", tb["y"], tb["detection_score"], td["detection_score"], tb["rows"],
                 td["rows"])
        else:
            leak(pid, name, "detection AUC", None, None, None, None, None)
    # P4, P5 (non-inferiority)
    for pid, task, margin, name in (("P4", "utility_ben", MARGIN_BEN, "utility_bigearthnet_noninferiority"),
                                    ("P5", "utility_eurosat", MARGIN_EUROSAT, "utility_eurosat_noninferiority")):
        cb, cd = best_cell(runs, base, "full", task), best_cell(runs, dfn, "full", task)
        if not (cb and cd):
            print(f"  {pid}: missing inputs, counted as p = 1 in the Holm family")
            rows.append(row(id=pid, test=name, role="primary", hypothesis="MISSING"))
            pvals.append(1.0)
            continue
        m, sd_ = metric_for(task), B.aligned(cb, cd)
        g = groups(task, cb, sp)
        diff, d = pair_diff(m, sd_, cb["scores"], cb["y"], g, n)   # defended - baseline
        lo, hi, _ = superiority(diff, d, "greater")
        p = non_inferiority(d, margin)
        rows.append(row(id=pid, test=name, measure="macro AP" if task == "utility_ben" else "accuracy",
                        baseline=cb["test"], defended=cd["test"], diff=diff, ci_lo=lo, ci_hi=hi,
                        hypothesis=f"defended - baseline > -{margin}", p=p, role="primary"))
        pvals.append(p)
    adj, rej = C.holm(pvals)
    for r, pa, rj in zip(rows, adj, rej):
        r["p_holm"], r["significant_holm"] = round(float(pa), 4), bool(rj)
    return rows


def verdict(rows):
    s = {r["id"]: r["significant_holm"] is True for r in rows}
    if s.get("P1") and s.get("P4") and s.get("P5"):
        return "works: leakage reduced significantly with utility within the pre-registered margins"
    if s.get("P1"):
        return "leakage cut at a cost: P1 significant but a utility non-inferiority test is not"
    return "no reduction: P1 (leakage at full size) not significant"


# ---------------------------------------------------------------- secondary

def secondary(runs, base, dfn, sp, n):
    out = []
    # learned excess over pixel statistics, and the compression curve
    px = best_cell(runs, "pixelstats", "full", "leakage")
    for enc in (base, dfn):
        for task in ("leakage", "leakage_incr"):
            c = best_cell(runs, enc, "full", task)
            if c and px:
                g = C.patch_groups(sp["patch_id"][c["rows"]])
                diff, lo, hi, p = C.paired_test(C.auc_metric, c["scores"], B.aligned(c, px), c["y"], g, rng(), n,
                                                "two-sided")
                out.append(row(id="S1", test=f"{task}_minus_pixelstats", measure="AUC", baseline=enc if enc == base else "",
                               defended=enc if enc == dfn else "", diff=diff, ci_lo=lo, ci_hi=hi,
                               hypothesis=f"{enc}: {c['attacker']} {c['test']:.4f} vs pixelstats {px['test']:.4f}",
                               p=p, role="exploratory"))
    for var in VARIANTS:
        cb, cd = best_cell(runs, base, var, "leakage"), best_cell(runs, dfn, var, "leakage")
        if cb and cd:
            g = C.patch_groups(sp["patch_id"][cb["rows"]])
            diff, lo, hi, p = C.paired_test(C.auc_metric, cb["scores"], B.aligned(cb, cd), cb["y"], g, rng(), n,
                                            "two-sided")
            out.append(row(id="S2", test=f"leakage_baseline_minus_defended_{var}", measure="AUC",
                           baseline=cb["test"], defended=cd["test"], diff=diff, ci_lo=lo, ci_hi=hi,
                           hypothesis="two-sided", p=p, role="exploratory"))
        for task, name in (("utility_ben", "BEN macro AP"), ("utility_eurosat", "EuroSAT accuracy"),
                           ("spatial", "spatial macro AP")):
            ub, ud = best_cell(runs, base, var, task), best_cell(runs, dfn, var, task)
            if ub and ud:
                out.append(row(id="S2", test=f"{task}_{var}", measure=name, baseline=ub["test"], defended=ud["test"],
                               diff=ud["test"] - ub["test"], hypothesis="defended - baseline, point estimates",
                               role="exploratory"))
    # localisation: mask IoU
    lb, ld = pertile("results/phase6", base), pertile("results/phase6", dfn)
    if lb is not None and ld is not None:
        pos = lb["rows"][lb["y"] == 1]
        g = C.patch_groups(sp["patch_id"][pos])
        for name, a, b in (("localisation_iou_baseline_minus_defended", lb["iou_positive"], ld["iou_positive"]),
                           ("localisation_iou_baseline_minus_prior", lb["iou_positive"], lb["prior_iou_positive"]),
                           ("localisation_iou_defended_minus_prior", ld["iou_positive"], ld["prior_iou_positive"])):
            d = C.boot_stat(lambda i: a[i].mean() - b[i].mean(), g, rng(), n)
            lo, hi, p = superiority(0, d, "two-sided")
            out.append(row(id="S3", test=name, measure="mask IoU (positive tiles)", baseline=float(a.mean()),
                           defended=float(b.mean()), diff=float(a.mean() - b.mean()), ci_lo=lo, ci_hi=hi,
                           hypothesis="two-sided", p=p, role="exploratory"))
    # inversion: GAN pixel metrics and mask IoU
    gb, gd = pertile("results/phase5/gan", base), pertile("results/phase5/gan", dfn)
    if gb is not None and gd is not None and np.array_equal(gb["rows"], gd["rows"]):
        g_all = C.patch_groups(sp["patch_id"][gb["rows"]])
        for m in ("l1", "ssim", "lpips", "sam"):
            a, b = gb[m], gd[m]
            d = C.boot_stat(lambda i: a[i].mean() - b[i].mean(), g_all, rng(), n)
            lo, hi, p = superiority(0, d, "two-sided")
            out.append(row(id="S4", test=f"gan_{m}_baseline_minus_defended", measure=m, baseline=float(a.mean()),
                           defended=float(b.mean()), diff=float(a.mean() - b.mean()), ci_lo=lo, ci_hi=hi,
                           hypothesis="two-sided", p=p, role="exploratory"))
        pos = gb["rows"][gb["y"] == 1]
        g = C.patch_groups(sp["patch_id"][pos])
        a, b = gb["iou_positive"], gd["iou_positive"]
        d = C.boot_stat(lambda i: a[i].mean() - b[i].mean(), g, rng(), n)
        lo, hi, p = superiority(0, d, "two-sided")
        out.append(row(id="S4", test="gan_iou_baseline_minus_defended", measure="mask IoU (positive tiles)",
                       baseline=float(a.mean()), defended=float(b.mean()), diff=float(a.mean() - b.mean()), ci_lo=lo,
                       ci_hi=hi, hypothesis="two-sided", p=p, role="exploratory"))
    return out


def summary(runs, encoders):
    rows = []
    for enc in encoders:
        for var in VARIANTS:
            for task in TASKS:
                c = best_cell(runs, enc, var, task)
                if c:
                    rows.append({"encoder": enc, "variant": var, "task": task, "attacker": c["attacker"],
                                 "val": round(c["val"], 4), "test": round(c["test"], 4)})
    return rows


def training(names):
    rows = []
    for n in names:
        f = f"results/phase7/{n}_train.json"
        if not os.path.exists(f):
            continue
        t = json.load(open(f))
        last, first = t["log"][-1], t["log"][0]
        rows.append({"encoder": n, "adv_weight": t["adv_weight"], "epochs": t["epochs"],
                     "mae_first": round(first["mae"], 4), "mae_last": round(last["mae"], 4),
                     "utility_ce_last": round(last["utility_ce"], 4),
                     "adversary_bce_last": round(last["adversary_bce"], 4),
                     "adversary_acc_last": round(last["adversary_acc"], 4), "seconds": t["seconds"]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="own_vit_baseline")
    ap.add_argument("--defended", default="own_vit_defended")
    ap.add_argument("--out", default="results/phase7/analysis")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    runs = B.read_csv(B.RUNS)
    sp = C.load_splits(0)
    prim = primary(runs, args.baseline, args.defended, sp, args.boot)
    B.write_csv(os.path.join(args.out, "primary.csv"), prim)
    B.write_csv(os.path.join(args.out, "secondary.csv"), secondary(runs, args.baseline, args.defended, sp, args.boot))
    B.write_csv(os.path.join(args.out, "summary.csv"), summary(runs, [args.baseline, args.defended] + REFERENCE))
    B.write_csv(os.path.join(args.out, "training.csv"), training([args.baseline, args.defended]))
    v = verdict(prim)
    for r in prim:
        print(f"  {r['id']} {r['test']:42s} diff {r['diff']}  p_holm {r['p_holm']}  significant {r['significant_holm']}")
    print(f"  verdict: {v}")
    with open(os.path.join(args.out, "README.json"), "w") as fh:
        json.dump({"baseline": args.baseline, "defended": args.defended, "verdict": v, "bootstrap": args.boot,
                   "leakage_diff": "baseline - defended (positive = the defence reduced leakage)",
                   "utility_diff": "defended - baseline (negative = the defence cost utility)",
                   "margins": {"utility_bigearthnet_macro_ap": MARGIN_BEN, "utility_eurosat_accuracy": MARGIN_EUROSAT},
                   "holm": "5 primary tests, alpha 0.05"}, fh, indent=2)


if __name__ == "__main__":
    main()
