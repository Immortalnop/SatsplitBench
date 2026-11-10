"""Phase 4b runner: the three research-grade attackers on every task, encoder and variant.

Tasks
  leakage          GloSoFarID tile "solar farm present"                binary,     ROC-AUC
  leakage_incr     same, features = pixel statistics + embedding       binary,     ROC-AUC
                   (compare with encoder `pixelstats` on `leakage` for learned leakage)
  spatial          positive tiles only: which of the 16 cells (16x16 px) of a 4x4 grid hold panels
                                                                       multilabel, macro AP (grid IoU in analysis)
  utility_eurosat  EuroSAT land cover, 10 classes                      multiclass, accuracy
  utility_ben      BigEarthNet, 19 labels (needs scripts/bigearthnet.py) multilabel, macro AP

Attackers: deep_mlp, lightgbm, svm_rbf (scripts/attackers.py). All hyperparameters on val.
GloSoFarID tasks use the train/val/test split of --split-seed (EuroSAT and BigEarthNet always use
their official splits).

Outputs
  results/phase4b/runs.csv     one row per (split_seed, encoder, variant, task, attacker):
                               val score, test score, chosen hyperparameters, seconds
  results/preds/phase4b/seed<k>/<encoder>/<variant>/<task>_<attacker>.npz
                               scores (N_test, ...), y, rows (test row indices), val_scores, val_y
Rows already in runs.csv are skipped, so an interrupted run resumes.

Usage
  python scripts/attack4b.py --tasks leakage leakage_incr
  python scripts/attack4b.py --tasks leakage --encoders pixelstats
  python scripts/attack4b.py --tasks leakage leakage_incr --variants full --split-seed 1
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attackers as A  # noqa: E402  (imports lightgbm before torch; keep first)
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

FIELDS = ["split_seed", "encoder", "variant", "task", "attacker", "metric", "val_score", "test_score",
          "n_train", "n_test", "dims", "params", "seconds"]
METRIC = {"leakage": "roc_auc", "leakage_incr": "roc_auc", "spatial": "macro_ap",
          "utility_eurosat": "accuracy", "utility_ben": "macro_ap"}
TASK_TYPE = {"leakage": "binary", "leakage_incr": "binary", "spatial": "multilabel",
             "utility_eurosat": "multiclass", "utility_ben": "multilabel"}
CELLS_FILE = "data/glosofarid/npy/cells_4x4.npy"


def cell_labels():
    """(N_tiles, 16) uint8: does each 16x16-px cell of each 64x64 tile contain a panel pixel."""
    if os.path.exists(CELLS_FILE):
        return np.load(CELLS_FILE)
    masks = np.load("data/glosofarid/npy/solar_2023_global_mask.npy", mmap_mode="r")
    out = []
    for i in range(0, len(masks), 256):
        m = np.asarray(masks[i:i + 256]).astype(bool)
        b = len(m)
        # patch (256x256) -> 16 tiles (64x64) row-major -> 16 cells (16x16) row-major per tile
        m = m.reshape(b, 4, 64, 4, 64).transpose(0, 1, 3, 2, 4).reshape(b * 16, 64, 64)
        out.append(m.reshape(-1, 4, 16, 4, 16).any(axis=(2, 4)).reshape(-1, 16))
    cells = np.concatenate(out).astype(np.uint8)
    np.save(CELLS_FILE, cells)
    return cells


def ben_labels():
    d = "data/bigearthnet/npy"
    if not os.path.exists(os.path.join(d, "train_y.npy")):
        raise SystemExit("BigEarthNet subset missing: run scripts/bigearthnet.py first")
    return {s: np.load(os.path.join(d, f"{s}_y.npy")) for s in ("train", "val", "test")}


def task_data(task, enc, var, sp):
    """Returns xtr, ytr, xva, yva, xte, yte, test_rows."""
    if task in ("leakage", "leakage_incr", "spatial"):
        x = C.load_variant(enc, var, ["glosofarid"])["glosofarid"]
        if task == "leakage_incr":
            px = C.load_variant("pixelstats", "full", ["glosofarid"])["glosofarid"]
            x = np.hstack([px, x])
        y = sp["y"].astype(np.int64)
        keep = np.ones(len(y), bool)
        if task == "spatial":
            y = cell_labels()
            keep = sp["y"] == 1
        rows = [np.where((sp["split"] == k) & keep)[0] for k in range(3)]
        return x[rows[0]], y[rows[0]], x[rows[1]], y[rows[1]], x[rows[2]], y[rows[2]], rows[2]
    if task == "utility_eurosat":
        e, ey = C.load_variant(enc, var, ["eurosat_train", "eurosat_val", "eurosat_test"]), C.eurosat_labels()
        return (e["eurosat_train"], ey["train"], e["eurosat_val"], ey["val"], e["eurosat_test"], ey["test"],
                np.arange(len(ey["test"])))
    if task == "utility_ben":
        b, by = C.load_variant(enc, var, ["ben_train", "ben_val", "ben_test"]), ben_labels()
        if "ben_train" not in b:
            raise FileNotFoundError(f"no BigEarthNet embeddings for {enc}/{var}")
        return b["ben_train"], by["train"], b["ben_val"], by["val"], b["ben_test"], by["test"], \
            np.arange(len(by["test"]))
    raise ValueError(task)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", required=True, choices=list(METRIC))
    ap.add_argument("--encoders", nargs="+", default=C.AUDITED + C.CONTROLS)
    ap.add_argument("--variants", nargs="+", default=C.FLOAT_VARIANTS)
    ap.add_argument("--attackers", nargs="+", default=A.ATTACKERS, choices=A.ATTACKERS)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="attacker seed")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="results/phase4b/runs.csv")
    ap.add_argument("--preds", default="results/preds/phase4b")
    args = ap.parse_args()

    sp = C.load_splits(args.split_seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(int(r["split_seed"]), r["encoder"], r["variant"], r["task"], r["attacker"])
                    for r in csv.DictReader(fh)}
    new = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        w.writeheader()

    for task in args.tasks:
        if task.startswith("utility") and args.split_seed != 0:
            print(f"skip {task}: utility uses official splits, run it with --split-seed 0")
            continue
        for enc in args.encoders:
            variants = ["full"] if enc == "pixelstats" else args.variants
            if enc == "pixelstats" and task == "leakage_incr":
                continue  # pixelstats + pixelstats is meaningless; its `leakage` row is the baseline
            for var in variants:
                todo = [a for a in args.attackers if (args.split_seed, enc, var, task, a) not in done]
                if not todo:
                    continue
                try:
                    xtr, ytr, xva, yva, xte, yte, rows = task_data(task, enc, var, sp)
                except FileNotFoundError as e:
                    print(f"skip {enc}/{var}/{task}: {e}")
                    continue
                for att in todo:
                    r = A.run_attacker(att, xtr, ytr, xva, yva, xte, TASK_TYPE[task], args.seed, args.device)
                    s = r["test_scores"]
                    test = A.val_metric(TASK_TYPE[task], yte, s)
                    pd = os.path.join(args.preds, f"seed{args.split_seed}", enc, var)
                    os.makedirs(pd, exist_ok=True)
                    np.savez_compressed(os.path.join(pd, f"{task}_{att}.npz"), scores=s, y=yte, rows=rows,
                                        val_scores=r["val_scores"], val_y=yva)
                    w.writerow({"split_seed": args.split_seed, "encoder": enc, "variant": var, "task": task,
                                "attacker": att, "metric": METRIC[task], "val_score": round(r["val_score"], 4),
                                "test_score": round(float(test), 4), "n_train": len(xtr), "n_test": len(xte),
                                "dims": xtr.shape[1], "params": json.dumps(r["params"]),
                                "seconds": round(r["seconds"])})
                    fh.flush()
                    print(f"seed{args.split_seed} {enc:>16} {var:>9} {task:>15} {att:>8}: "
                          f"val {r['val_score']:.4f} test {test:.4f} ({r['seconds']:.0f}s)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
