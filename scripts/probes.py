"""Phase 4: utility and attribute (leakage) probes on every encoder x compression variant.

For each embedding set in data/compressed/<encoder>/<variant>/:
  utility  = EuroSAT land cover, 10 classes            -> test accuracy (chance 10%)
  leakage  = GloSoFarID tile "solar farm present"      -> test ROC-AUC  (chance 0.5)
Two attackers per task:
  linear = standardised logistic regression; C chosen on val from --cs (class_weight balanced for leakage)
  mlp    = one hidden layer (256, ReLU, dropout 0.1), AdamW, early stopping on val loss; one run per seed
Model selection uses val only; every reported number is on test.

Uncertainty:
  95% bootstrap CIs (1,000 resamples). Leakage resamples whole GloSoFarID PATCHES, not tiles,
  because the 16 tiles of a patch are correlated. Utility resamples EuroSAT test images.
  MLP: mean +/- std over seeds; its CI is from the first seed.

Controls (all at native resolution, nothing resized):
  resnet18_random = same architecture as resnet18_moco, random weights (from Phase 2/3)
  pixelstats      = per-band mean, std, p10, p50, p90 of each 64x64 image (13 x 5 = 65 features),
                    built here into data/compressed/pixelstats/full/ on first run

Output: results/probes.csv, one row per (encoder, variant, task, probe). Rows already in the CSV are
skipped, so an interrupted run resumes.
With --preds-dir, per-sample test predictions are saved for paired tests (Phase 4b):
  <preds-dir>/<encoder>/<variant>/<task>_<probe>.npz
    pred   (n_seeds, N_test)  leakage: P(solar) as float32; utility: predicted class as int16
    y      (N_test,)          true labels, in test-row order
    seeds  (n_seeds,)

Usage:
  python scripts/probes.py
  python scripts/probes.py --out results/probes_v2.csv --preds-dir results/preds/phase4
  python scripts/probes.py --encoders resnet18_moco --variants full pca16
"""
import argparse
import csv
import os
import time

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

COMP = "data/compressed"
FIELDS = ["encoder", "variant", "dims", "bytes", "task", "probe", "metric", "value", "ci_lo", "ci_hi",
          "seed_std", "n_seeds", "chosen_C", "chance", "val_value"]


# ---------------------------------------------------------------- data

def load_variant(enc, var):
    """Dequantises int8 (scale.npy) and Phase 4b quantile codes (codebook.npy)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from common import dequantise

    d = os.path.join(COMP, enc, var)
    return {n: dequantise(d, np.load(os.path.join(d, f"{n}.npy")))
            for n in ("eurosat_train", "eurosat_val", "eurosat_test", "glosofarid")}


def tile_stats(x):
    """x: (N, 13, 64, 64) -> (N, 65) per-band mean, std, p10, p50, p90 of reflectance."""
    f = x.reshape(len(x), x.shape[1], -1).astype(np.float32) / 10000.0
    p = np.percentile(f, [10, 50, 90], axis=2)  # (3, N, 13)
    return np.concatenate([f.mean(2), f.std(2), p[0], p[1], p[2]], axis=1).astype(np.float32)


def build_pixelstats():
    out = os.path.join(COMP, "pixelstats", "full")
    if os.path.exists(os.path.join(out, "glosofarid.npy")):
        return
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from embed import patches_to_tiles

    os.makedirs(out, exist_ok=True)
    print("building pixelstats control (65 per-band statistics, no resizing)")
    for s in ("train", "val", "test"):
        x = np.load(f"data/eurosat/npy/{s}_x.npy", mmap_mode="r")
        np.save(os.path.join(out, f"eurosat_{s}.npy"),
                np.concatenate([tile_stats(np.asarray(x[i:i + 2048])) for i in range(0, len(x), 2048)]))
    x = np.load("data/glosofarid/npy/solar_2023_global_x.npy", mmap_mode="r")
    feats = [tile_stats(patches_to_tiles(np.asarray(x[i:i + 64]))) for i in range(0, len(x), 64)]
    np.save(os.path.join(out, "glosofarid.npy"), np.concatenate(feats))


# ---------------------------------------------------------------- metrics

def boot_ci(stat, groups, rng, n=1000):
    """Percentile CI of stat(idx) resampling `groups` (list of index arrays) with replacement."""
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[g] for g in pick])
        try:
            vals.append(stat(idx))
        except ValueError:  # a resample with one class only
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ---------------------------------------------------------------- probes

def linear_probe(xtr, ytr, xva, yva, xte, cs, binary):
    best = None
    for c in cs:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced" if binary else None)
        clf.fit(xtr, ytr)
        score = roc_auc_score(yva, clf.predict_proba(xva)[:, 1]) if binary else clf.score(xva, yva)
        if best is None or score > best[0]:
            best = (score, c, clf)
    val_score, c, clf = best
    out = clf.predict_proba(xte)[:, 1] if binary else clf.predict(xte)
    return out, c, float(val_score)


def mlp_probe(xtr, ytr, xva, yva, xte, n_classes, seed, device, epochs=100, patience=8):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = xtr.shape[1]
    out_dim = 1 if n_classes == 2 else n_classes
    model = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                torch.nn.Linear(256, out_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if n_classes == 2:
        pos = float(ytr.mean())
        lossf = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32, device=device))
        tgt = lambda y: torch.as_tensor(y, dtype=torch.float32, device=device)[:, None]
    else:
        lossf = torch.nn.CrossEntropyLoss()
        tgt = lambda y: torch.as_tensor(y, dtype=torch.long, device=device)
    T = lambda a: torch.as_tensor(a, device=device)
    xtr_t, xva_t, xte_t, yva_t = T(xtr), T(xva), T(xte), tgt(yva)

    best, best_state, bad = np.inf, None, 0
    for _ in range(epochs):
        model.train()
        for b in np.array_split(rng.permutation(len(xtr)), max(1, len(xtr) // 512)):
            opt.zero_grad()
            lossf(model(xtr_t[b]), tgt(ytr[b])).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(xva_t), yva_t).item()
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, vlogits = model(xte_t), model(xva_t)
    if n_classes == 2:
        val = roc_auc_score(yva, torch.sigmoid(vlogits[:, 0]).cpu().numpy())
        return torch.sigmoid(logits[:, 0]).cpu().numpy(), float(val)
    val = float((vlogits.argmax(1).cpu().numpy() == yva).mean())
    return logits.argmax(1).cpu().numpy(), val


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=None, help="default: all in data/compressed")
    ap.add_argument("--variants", nargs="+", default=None, help="default: all present per encoder")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--cs", nargs="+", type=float, default=[0.01, 0.1, 1.0])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="results/probes.csv")
    ap.add_argument("--preds-dir", default=None, help="save per-sample test predictions here")
    ap.add_argument("--comp-root", default="data/compressed",
                    help="data/compressed (Phase 3 variants) or data/quantised (Phase 4b q8/q4/q2/q1)")
    args = ap.parse_args()
    global COMP
    COMP = args.comp_root

    if COMP == "data/compressed":  # the pixel-statistics control lives with the Phase 3 variants only
        build_pixelstats()
    sp = np.load("data/embeddings/splits_glosofarid.npz")
    g_y, g_split, g_patch = sp["y"].astype(np.int64), sp["split"], sp["patch_id"]
    e_y = {s: np.load(f"data/eurosat/npy/{s}_y.npy") for s in ("train", "val", "test")}
    te = np.where(g_split == 2)[0]
    te_patch_groups = [np.where(g_patch[te] == p)[0] for p in np.unique(g_patch[te])]
    e_groups = [np.array([i]) for i in range(len(e_y["test"]))]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        with open(args.out) as fh:
            done = {(r["encoder"], r["variant"], r["task"], r["probe"]) for r in csv.DictReader(fh)}
    new_file = not os.path.exists(args.out)
    fh = open(args.out, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    encoders = args.encoders or sorted(os.listdir(COMP))
    for enc in encoders:
        variants = args.variants or sorted(v for v in os.listdir(os.path.join(COMP, enc))
                                           if os.path.isdir(os.path.join(COMP, enc, v)))
        for var in variants:
            if not os.path.isdir(os.path.join(COMP, enc, var)):
                continue
            todo = [(t, p) for t in ("utility", "leakage") for p in ("linear", "mlp")
                    if (enc, var, t, p) not in done]
            if not todo:
                continue
            t0 = time.time()
            e = load_variant(enc, var)
            dims = e["glosofarid"].shape[1]
            nbytes = dims * (1 if var.endswith("_int8") else 4)
            g = e["glosofarid"]
            data = {
                "utility": (e["eurosat_train"], e_y["train"], e["eurosat_val"], e_y["val"],
                            e["eurosat_test"], e_y["test"], 10),
                "leakage": (g[g_split == 0], g_y[g_split == 0], g[g_split == 1], g_y[g_split == 1],
                            g[te], g_y[te], 2),
            }
            for task, probe in todo:
                xtr, ytr, xva, yva, xte, yte, k = data[task]
                sc = StandardScaler().fit(xtr)
                xtr_s, xva_s, xte_s = (sc.transform(a).astype(np.float32) for a in (xtr, xva, xte))
                binary = k == 2
                metric = (lambda pr, idx: roc_auc_score(yte[idx], pr[idx])) if binary else \
                         (lambda pr, idx: float((pr[idx] == yte[idx]).mean()))
                groups = te_patch_groups if binary else e_groups
                rng = np.random.default_rng(0)
                if probe == "linear":
                    pred, c, val_value = linear_probe(xtr_s, ytr, xva_s, yva, xte_s, args.cs, binary)
                    preds, seeds_used = [pred], [0]
                    vals, chosen = [metric(pred, np.arange(len(yte)))], c
                    first = pred
                else:
                    outs = [mlp_probe(xtr_s, ytr, xva_s, yva, xte_s, k, s, args.device) for s in args.seeds]
                    preds, val_value = [o[0] for o in outs], float(np.mean([o[1] for o in outs]))
                    seeds_used = list(args.seeds)
                    vals, chosen, first = [metric(p, np.arange(len(yte))) for p in preds], "", preds[0]
                if args.preds_dir:
                    pd_dir = os.path.join(args.preds_dir, enc, var)
                    os.makedirs(pd_dir, exist_ok=True)
                    np.savez(os.path.join(pd_dir, f"{task}_{probe}.npz"),
                             pred=np.stack(preds).astype(np.float32 if binary else np.int16),
                             y=yte, seeds=np.array(seeds_used))
                lo, hi = boot_ci(lambda idx: metric(first, idx), groups, rng)
                row = {"encoder": enc, "variant": var, "dims": dims, "bytes": nbytes, "task": task,
                       "probe": probe, "metric": "roc_auc" if binary else "accuracy",
                       "value": round(float(np.mean(vals)), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                       "seed_std": round(float(np.std(vals)), 4) if len(vals) > 1 else "",
                       "n_seeds": len(vals), "chosen_C": chosen, "chance": 0.5 if binary else 0.1,
                       "val_value": round(val_value, 4)}
                w.writerow(row)
                fh.flush()
                print(f"{enc:>16} {var:>12} {task:>8} {probe:>6}: {row['metric']} {row['value']:.3f} "
                      f"[{lo:.3f}, {hi:.3f}]", flush=True)
            print(f"  ({time.time() - t0:.0f}s)", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
