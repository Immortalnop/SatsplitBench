"""Phase 8d: follow-ups to 8b (exploratory: decided after the 8b results were seen; see results/preregistration.md,
section "Phase 8d follow-ups").

  1. Pixel control for location. The raw tile, average-pooled to the same 4x4 grid as `map4` (13 bands x 16 cells
     = 208 dims), as pseudo-encoder `pixels`, variant `map4`. Question: do encoder maps locate panels better than the
     pixels pooled the same way?
  2. Payload form. Spatial attack on the pooled `gap` vector vs the `map4` map, at the earliest and final split.
     Question: is the location signal carried by the map form, on the same test tiles?
  3. Replication. The 8 pre-registered 8b contrasts re-run on GloSoFarID split seeds 1 and 2.

All comparisons are paired on the same test tiles (cluster bootstrap over patches) and Holm-corrected within each family.

Usage
  python scripts/deepen8.py --build                 # writes data/compressed/pixels/map4/glosofarid.npy (~1 min)
  python scripts/deepen8.py --analyse               # after the attackers (scripts/run_phase8.sh 8d)
  python scripts/deepen8.py --analyse --boot 50 --out /tmp/x    # code test
Outputs: results/phase8/deepen/{vs_pixels,form,replication}.csv, README.json
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

import analysis8b as A8  # noqa: E402
import splitpoints as S  # noqa: E402

PIX = os.path.join(C.COMP, "pixels", "map4")
SEEDS = (0, 1, 2)


def build():
    C.build_tile_cache()
    x = np.load(C.TILES_X, mmap_mode="r")
    os.makedirs(PIX, exist_ok=True)
    out = np.lib.format.open_memmap(os.path.join(PIX, "glosofarid.npy"), "w+", np.float16, (len(x), 13 * 16))
    for i in range(0, len(x), 2048):
        b = np.asarray(x[i:i + 2048], dtype=np.float32) / 10000.0
        out[i:i + len(b)] = b.reshape(len(b), 13, 4, 16, 4, 16).mean((3, 5)).reshape(len(b), -1).astype(np.float16)
    out.flush()
    print(f"wrote {PIX}/glosofarid.npy {out.shape}")


_RUNS = {}


def runs_for(seed):
    if seed not in _RUNS:
        _RUNS[seed] = {(r["encoder"], r["variant"], r["task"]): r for r in csv.DictReader(open(A8.RUNS))
                       if r["attacker"] == "deep_mlp" and r["split_seed"] == str(seed)}
    return _RUNS[seed]


def cell(name, form, task, seed):
    """Test predictions of one deep-MLP cell (any pseudo-encoder, any split seed); test score recomputed."""
    r = runs_for(seed).get((name, form, task))
    f = os.path.join(f"results/preds/phase8b/seed{seed}", name, form, f"{task}_deep_mlp.npz")
    if r is None or not os.path.exists(f):
        return None
    d = np.load(f)
    return {"test": float(B.test_metric(task, d["y"], d["scores"], np.arange(len(d["y"])))), "scores": d["scores"],
            "y": d["y"], "rows": d["rows"], "val_scores": d["val_scores"], "val_y": d["val_y"]}


def iou_of(c):
    return B.grid_iou(c["scores"], c["y"], B.best_threshold(c["val_scores"], c["val_y"]))


def compare(task, a, b, sp, n, label):
    """a - b on the same test tiles: metric (AUC or macro AP) and, for spatial, grid IoU."""
    assert np.array_equal(a["rows"], b["rows"]), f"{label}: test rows differ"
    g = C.patch_groups(sp["patch_id"][a["rows"]])
    metric = lambda y, s, idx: B.test_metric(task, y, s, idx)  # noqa: E731
    diff, lo, hi, p = C.paired_test(metric, a["scores"], b["scores"], a["y"], g, A8.rng(), n, "two-sided")
    r = {"comparison": label, "task": task, "a": round(a["test"], 4), "b": round(b["test"], 4), "diff": round(diff, 4),
         "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 4)}
    if task == "spatial":
        ia, ib = iou_of(a), iou_of(b)
        d = C.boot_stat(lambda idx: ia[idx].mean() - ib[idx].mean(), g, A8.rng(), n)
        r.update({"iou_a": round(float(ia.mean()), 4), "iou_b": round(float(ib.mean()), 4),
                  "iou_diff": round(float(ia.mean() - ib.mean()), 4),
                  "iou_ci_lo": round(float(np.percentile(d, 2.5)), 4),
                  "iou_ci_hi": round(float(np.percentile(d, 97.5)), 4),
                  "iou_p": round(float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / (len(d) + 1))), 4)})
    return r


def holm_on(rows, key="p"):
    run = [r for r in rows if key in r]
    if run:
        adj, rej = C.holm([r[key] for r in run])
        for r, pa, rj in zip(run, adj, rej):
            r[f"{key}_holm"], r[f"{key}_sig"] = round(float(pa), 4), bool(rj)
    return rows


def ends(enc):
    s = S.SPLITS[S.ARCH[enc]]
    return s[0], s[-1]


def vs_pixels(n):
    sp, rows = C.load_splits(0), []
    for task in ("spatial", "leakage"):
        px = cell("pixels", "map4", task, 0)
        if px is None:
            print(f"  pixels/map4 {task}: missing, skipped")
            continue
        for enc in A8.ENCODERS:
            for split in S.SPLITS[S.ARCH[enc]]:
                c = cell(f"{enc}@{split}", "map4", task, 0)
                if c is not None:
                    rows.append({"encoder": enc, "split": split,
                                 **compare(task, c, px, sp, n, f"{task}: map4 - pixel map4")})
    for task in ("spatial", "leakage"):  # Holm per task family, audited encoders only (random is a control)
        holm_on([r for r in rows if r["task"] == task and r["encoder"] in C.AUDITED],
                "iou_p" if task == "spatial" else "p")
    return rows


def form(n):
    sp, rows = C.load_splits(0), []
    for enc in C.AUDITED:
        for split in ends(enc):
            m, g = cell(f"{enc}@{split}", "map4", "spatial", 0), cell(f"{enc}@{split}", "gap", "spatial", 0)
            if m is not None and g is not None:
                rows.append({"encoder": enc, "split": split, **compare("spatial", m, g, sp, n, "spatial: map4 - gap")})
    return holm_on(rows, "iou_p")


def replication(n):
    rows = []
    for seed in SEEDS:
        sp = C.load_splits(seed)
        for fam, frm, task in (("depth adds presence leakage", "gap", "leakage"),
                               ("early splits leak location", "map4", "spatial")):
            for enc in C.AUDITED:
                first, last = ends(enc)
                hi_, lo_ = (last, first) if task == "leakage" else (first, last)
                a, b = cell(f"{enc}@{hi_}", frm, task, seed), cell(f"{enc}@{lo_}", frm, task, seed)
                if a is None or b is None:
                    continue
                g = C.patch_groups(sp["patch_id"][a["rows"]])
                metric = lambda y, s, idx, t=task: B.test_metric(t, y, s, idx)  # noqa: E731
                diff, lo, hi, p = C.paired_test(metric, a["scores"], b["scores"], a["y"], g, A8.rng(), n, "greater")
                rows.append({"seed": seed, "family": fam, "encoder": enc, "contrast": f"{hi_} - {lo_}",
                             "a": round(a["test"], 4), "b": round(b["test"], 4), "diff": round(diff, 4),
                             "ci_lo": round(lo, 4), "ci_hi": round(hi, 4), "p": round(p, 4)})
    for seed in SEEDS:
        holm_on([r for r in rows if r["seed"] == seed])
    return rows


def summary(rep):
    out = {}
    for fam in dict.fromkeys(r["family"] for r in rep):
        for enc in C.AUDITED:
            rr = [r for r in rep if r["family"] == fam and r["encoder"] == enc]
            if rr:
                out[f"{fam} | {enc}"] = {"seeds": len(rr), "significant (Holm within seed)": sum(r.get("p_sig", False) for r in rr),
                                         "mean diff": round(float(np.mean([r["diff"] for r in rr])), 4),
                                         "diff range": [min(r["diff"] for r in rr), max(r["diff"] for r in rr)]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyse", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/phase8/deepen")
    args = ap.parse_args()
    if args.build:
        build()
    if not args.analyse:
        return
    os.makedirs(args.out, exist_ok=True)
    res = {"vs_pixels": vs_pixels(args.boot), "form": form(args.boot), "replication": replication(args.boot)}
    for k, rows in res.items():
        A8.write_csv(os.path.join(args.out, f"{k}.csv"), rows)
    for r in res["vs_pixels"]:
        extra = f", IoU {r['iou_diff']:+.4f} [{r['iou_ci_lo']:+.4f}, {r['iou_ci_hi']:+.4f}]" if "iou_diff" in r else ""
        print(f"  {r['comparison']:>26} {r['encoder']:>15} {r['split']:>7}: {r['diff']:+.4f} "
              f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]{extra}")
    for r in res["form"]:
        print(f"  {r['comparison']} {r['encoder']:>15} {r['split']:>7}: AP {r['diff']:+.4f}, "
              f"IoU {r['iou_diff']:+.4f} [{r['iou_ci_lo']:+.4f}, {r['iou_ci_hi']:+.4f}] p_holm {r.get('iou_p_holm')}")
    s = summary(res["replication"])
    for k, v in s.items():
        print(f"  {k}: {v}")
    json.dump({"exploratory": True, "boot": args.boot, "seeds": SEEDS, "replication_summary": s,
               "pixel_control": "raw tile / 10000, 4x4 average pool, 13 x 16 = 208 dims"},
              open(os.path.join(args.out, "README.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
