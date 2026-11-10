"""Phase 4b: coarser quantisation (8/4/2/1 bits) of the float compression levels.

Per-dimension quantile (equal-frequency) scalar quantisation:
  - for each dimension, 2^b bins with edges at training quantiles;
  - each code decodes to the mean of the training values in its bin (Lloyd-style reconstruction).
Phase 3's int8 scales by the per-dimension maximum; at 4 bits or fewer that collapses most values to
zero, which would measure a poor encoder rather than compression. q8 here bridges to Phase 3's int8.

Fitted on exactly the rows Phase 3 fitted PCA on (EuroSAT train + GloSoFarID train, equal numbers,
same seed), and applied unchanged to val/test and BigEarthNet.

Inputs : data/compressed/<encoder>/<level>/        (float levels: full, pca256, pca64, pca16)
Outputs: data/quantised/<encoder>/<level>_q<b>/<split>.npy   uint8 codes
         data/quantised/<encoder>/<level>_q<b>/codebook.npy  (D, 2^b) float32
         data/quantised/<encoder>/meta.json
Written to a temporary folder and renamed, so readers never see half-written variants.

Usage:
  python scripts/quantise.py
  python scripts/quantise.py --encoders resnet18_moco --bits 4 1
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

LEVELS = ["full", "pca256", "pca64", "pca16"]


def fit_rows_index(gl_split, n_eurosat, seed):
    """Reproduce compress.fit_rows exactly: same rng, same draws."""
    rng = np.random.default_rng(seed)
    gl_tr = np.where(gl_split == 0)[0]
    n = min(n_eurosat, len(gl_tr))
    a = rng.choice(n_eurosat, n, replace=False)
    b = rng.choice(len(gl_tr), n, replace=False)
    return a, gl_tr[b]


def fit_codebook(fit, bits):
    L = 2 ** bits
    d = fit.shape[1]
    edges = np.quantile(fit, np.linspace(0, 1, L + 1)[1:-1], axis=0).T  # (D, L-1)
    codebook = np.zeros((d, L), np.float32)
    for j in range(d):
        codes = np.searchsorted(edges[j], fit[:, j], side="right")
        sums = np.bincount(codes, weights=fit[:, j], minlength=L)
        cnt = np.bincount(codes, minlength=L)
        filled = cnt > 0
        codebook[j, filled] = sums[filled] / cnt[filled]
        if not filled.all():  # ties in the data leave empty bins: copy the nearest filled bin
            idx = np.where(filled)[0]
            for k in np.where(~filled)[0]:
                codebook[j, k] = codebook[j, idx[np.argmin(np.abs(idx - k))]]
    return edges, codebook


def encode(x, edges):
    out = np.empty(x.shape, np.uint8)
    for j in range(x.shape[1]):
        out[:, j] = np.searchsorted(edges[j], x[:, j], side="right")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=C.AUDITED + ["resnet18_random"])
    ap.add_argument("--bits", nargs="+", type=int, default=[8, 4, 2, 1])
    ap.add_argument("--out", default="data/quantised")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gl_split = C.load_splits(0)["split"]
    for enc in args.encoders:
        meta = {"encoder": enc, "scheme": "per-dimension quantile, bin-mean reconstruction",
                "fit_on": "Phase 3 PCA fit rows (EuroSAT train + GloSoFarID train, equal numbers)",
                "seed": args.seed, "variants": {}}
        print(f"\n== {enc}")
        for level in LEVELS:
            src = os.path.join(C.COMP, enc, level)
            if not os.path.isdir(src):
                continue
            arrays = {n[:-4]: np.load(os.path.join(src, n)) for n in sorted(os.listdir(src)) if n.endswith(".npy")}
            ia, ib = fit_rows_index(gl_split, len(arrays["eurosat_train"]), args.seed)
            fit = np.concatenate([arrays["eurosat_train"][ia], arrays["glosofarid"][ib]])
            dims = fit.shape[1]
            for bits in args.bits:
                name = f"{level}_q{bits}"
                dst = os.path.join(args.out, enc, name)
                if os.path.exists(dst):
                    print(f"  {name}: exists, skipped")
                    continue
                edges, cb = fit_codebook(fit, bits)
                tmp = dst + ".tmp"
                shutil.rmtree(tmp, ignore_errors=True)
                os.makedirs(tmp)
                for split, a in arrays.items():
                    np.save(os.path.join(tmp, f"{split}.npy"), encode(a, edges))
                np.save(os.path.join(tmp, "codebook.npy"), cb)
                np.save(os.path.join(tmp, "edges.npy"), edges.astype(np.float32))
                os.replace(tmp, dst)
                # reconstruction quality on the fit rows, relative to their variance
                rec = cb[np.arange(dims)[None, :], encode(fit, edges).astype(np.int64)]
                rel_mse = float(((rec - fit) ** 2).mean() / fit.var(axis=0).mean())
                meta["variants"][name] = {"dims": dims, "bits": bits, "bytes": dims * bits / 8,
                                          "relative_mse_on_fit_rows": rel_mse}
                print(f"  {name:>11}: {dims:>4} dims x {bits} bit = {dims * bits / 8:>6.0f} B, "
                      f"relative MSE {rel_mse:.4f}")
        os.makedirs(os.path.join(args.out, enc), exist_ok=True)
        mf = os.path.join(args.out, enc, "meta.json")
        old = json.load(open(mf))["variants"] if os.path.exists(mf) else {}
        meta["variants"] = {**old, **meta["variants"]}
        with open(mf, "w") as fh:
            json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
