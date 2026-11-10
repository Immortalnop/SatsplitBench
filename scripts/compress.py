"""Phase 3: compress embeddings with PCA (16/64/256 dims) and int8 quantisation.

Everything that is fitted (PCA mean and components, int8 scales) is fitted on TRAINING rows only:
EuroSAT train + GloSoFarID train tiles, subsampled to equal size so neither dataset dominates.
Val/test rows are only transformed, as a released compressor would be applied to new data.

Variants per encoder:
  full, full_int8, pca16, pca16_int8, pca64, pca64_int8, pca256, pca256_int8
  (`full` = the uncompressed reference point; PCA dims above the embedding size are skipped.)

int8: symmetric per-dimension scale = max |value| over training rows / 127;
q = clip(round(v / scale), -127, 127). Val/test values beyond the training range are clipped.

Outputs (under --out, default data/compressed):
  <encoder>/<variant>/eurosat_{train,val,test}.npy, glosofarid.npy
      float32 for float variants; int8 for *_int8 (multiply by scale.npy to dequantise)
  <encoder>/<variant>/scale.npy      (*_int8 only)
  <encoder>/pca<k>.npz               mean, components, explained_variance_ratio
  <encoder>/meta.json                per variant: dims, bytes per embedding, variance kept, int8 clip rate

Usage:
  python scripts/compress.py
  python scripts/compress.py --encoders resnet18_moco --dims 16 64
"""
import argparse
import json
import os

import numpy as np
from sklearn.decomposition import PCA

SPLITS = ("train", "val", "test")


def load_encoder(emb_dir, enc, gl_split):
    e = {f"eurosat_{s}": np.load(os.path.join(emb_dir, enc, f"eurosat_{s}.npy")) for s in SPLITS}
    e["glosofarid"] = np.load(os.path.join(emb_dir, enc, "glosofarid.npy"))
    assert len(e["glosofarid"]) == len(gl_split), "glosofarid rows do not match splits file"
    return e


def fit_rows(e, gl_split, rng):
    """Training rows only, EuroSAT and GloSoFarID in equal numbers."""
    a = e["eurosat_train"]
    b = e["glosofarid"][gl_split == 0]
    n = min(len(a), len(b))
    return np.concatenate([a[rng.choice(len(a), n, replace=False)],
                           b[rng.choice(len(b), n, replace=False)]])


def quantise(arrays, fit):
    scale = np.abs(fit).max(axis=0) / 127.0
    scale[scale == 0] = 1.0
    out, clipped, total = {}, 0, 0
    for k, v in arrays.items():
        r = np.rint(v / scale)
        clipped += int((np.abs(r) > 127).sum())
        total += r.size
        out[k] = np.clip(r, -127, 127).astype(np.int8)
    return out, scale.astype(np.float32), clipped / total


def save_variant(d, arrays, scale=None):
    os.makedirs(d, exist_ok=True)
    for k, v in arrays.items():
        np.save(os.path.join(d, f"{k}.npy"), v)
    if scale is not None:
        np.save(os.path.join(d, "scale.npy"), scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=None, help="default: every folder in --emb")
    ap.add_argument("--dims", nargs="+", type=int, default=[16, 64, 256])
    ap.add_argument("--emb", default="data/embeddings")
    ap.add_argument("--out", default="data/compressed")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    gl_split = np.load(os.path.join(args.emb, "splits_glosofarid.npz"))["split"]
    encoders = args.encoders or sorted(
        d for d in os.listdir(args.emb) if os.path.isfile(os.path.join(args.emb, d, "glosofarid.npy")))

    for enc in encoders:
        rng = np.random.default_rng(args.seed)
        e = load_encoder(args.emb, enc, gl_split)
        dim = e["glosofarid"].shape[1]
        fit = fit_rows(e, gl_split, rng)
        enc_out = os.path.join(args.out, enc)
        meta = {"encoder": enc, "embedding_dim": dim, "fit_rows": len(fit),
                "fit_on": "EuroSAT train + GloSoFarID train tiles, equal numbers", "seed": args.seed,
                "variants": {}}
        print(f"\n== {enc} (dim {dim}, PCA/int8 fitted on {len(fit)} training rows)")

        levels = [("full", None)] + [(f"pca{k}", k) for k in args.dims if k < dim]
        for name, k in levels:
            if k is None:
                arrays, fit_k, kept = e, fit, 1.0
            else:
                pca = PCA(n_components=k, random_state=args.seed).fit(fit)
                arrays = {n: pca.transform(v).astype(np.float32) for n, v in e.items()}
                fit_k = pca.transform(fit)
                kept = float(pca.explained_variance_ratio_.sum())
                os.makedirs(enc_out, exist_ok=True)
                np.savez(os.path.join(enc_out, f"{name}.npz"), mean=pca.mean_, components=pca.components_,
                         explained_variance_ratio=pca.explained_variance_ratio_)
            save_variant(os.path.join(enc_out, name), arrays)
            q, scale, clip_rate = quantise(arrays, fit_k)
            save_variant(os.path.join(enc_out, f"{name}_int8"), q, scale)

            d = k or dim
            meta["variants"][name] = {"dims": d, "bytes": d * 4, "variance_kept": kept}
            meta["variants"][f"{name}_int8"] = {"dims": d, "bytes": d, "variance_kept": kept,
                                                "int8_clip_rate": clip_rate}
            print(f"  {name:>8}: {d:>4} dims, {d * 4:>5} B float32 / {d:>4} B int8, "
                  f"variance kept {kept:.1%}, int8 clipped {clip_rate:.4%}")

        with open(os.path.join(enc_out, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)


if __name__ == "__main__":
    main()
