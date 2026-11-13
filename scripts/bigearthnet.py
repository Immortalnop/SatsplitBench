"""Phase 4b: BigEarthNet v2 utility task (19 labels) for all encoders and compression variants.

Subset: random draw from the official splits (default 30,000 train / 10,000 val / 10,000 test),
excluding patches flagged for seasonal snow or cloud/shadow.

Preprocessing, per patch (no change of ground resolution or extent beyond the crop):
  - BigEarthNet stores 20 m bands at 60x60 and 60 m bands at 20x20. They are put on the 10 m grid
    (120x120) by nearest-neighbour replication (each value copied 2x2 or 6x6; no values invented).
    EuroSAT, GloSoFarID and the encoders' pretraining data already have every band on the 10 m grid.
  - B10 is absent from L2A products: filled with zeros, in its slot between B09 and B11.
  - Centre crop 120 -> 112 (ViT-S/16 needs a multiple of 16). Cropping, not resizing.
  - Divided by 10,000 like every other input.
One pass over the images feeds all encoders and the pixel-statistics control, so images are not stored.

Outputs
  data/bigearthnet/npy/{train,val,test}_y.npy     (N, 19) uint8 multi-hot labels
  data/bigearthnet/npy/{split}_patch_ids.npy, label_names.json
  data/embeddings/<encoder>/ben_{split}.npy       raw embeddings
  data/compressed/<encoder>/<variant>/ben_{split}.npy   Phase 3 variants (existing PCA/int8 fits, no refit)
  data/quantised/<encoder>/<variant>/ben_{split}.npy    Phase 4b quantile variants (existing codebooks)
  data/compressed/pixelstats/full/ben_{split}.npy

Usage
  python scripts/bigearthnet.py
  python scripts/bigearthnet.py --n 3000 1000 1000      # quick test
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

ROOT = "data/bigearthnet"
OUT = os.path.join(ROOT, "npy")
# encoder band order with the factor each BigEarthNet band must be replicated by (None = zero-filled B10)
BANDS = [("B01", 6), ("B02", 1), ("B03", 1), ("B04", 1), ("B05", 2), ("B06", 2), ("B07", 2), ("B08", 1),
         ("B8A", 2), ("B09", 6), ("B10", None), ("B11", 2), ("B12", 2)]
CROP = 112


def load_patch(patch_id):
    import rasterio

    tile = patch_id.rsplit("_", 2)[0]
    d = os.path.join(ROOT, "BigEarthNet-S2", tile, patch_id)
    img = np.zeros((len(BANDS), 120, 120), np.uint16)
    for i, (b, f) in enumerate(BANDS):
        if f is None:
            continue
        with rasterio.open(os.path.join(d, f"{patch_id}_{b}.tif")) as r:
            a = r.read(1)
        if f > 1:
            a = np.repeat(np.repeat(a, f, axis=0), f, axis=1)
        assert a.shape == (120, 120), (patch_id, b, a.shape)
        img[i] = a
    o = (120 - CROP) // 2
    return img[:, o:o + CROP, o:o + CROP]


def stack_batch(b):  # module-level so DataLoader workers can pickle it (macOS uses spawn)
    return np.stack(b)


class BenDataset:
    def __init__(self, ids):
        self.ids = ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return load_patch(self.ids[i])


def select_subset(n, seed):
    import pandas as pd

    df = pd.read_parquet(os.path.join(ROOT, "metadata.parquet"))
    df = df[~df["contains_seasonal_snow"] & ~df["contains_cloud_or_shadow"]]
    names = sorted({l for ls in df["labels"] for l in ls})
    assert len(names) == 19, f"expected 19 BigEarthNet labels, got {len(names)}"
    rng = np.random.default_rng(seed)
    out = {}
    for split, k in zip(("train", "val", "test"), n):
        sub = df[df["split"] == ("validation" if split == "val" else split)]
        sub = sub.iloc[np.sort(rng.choice(len(sub), min(k, len(sub)), replace=False))]
        y = np.zeros((len(sub), 19), np.uint8)
        for r, ls in enumerate(sub["labels"]):
            y[r, [names.index(l) for l in ls]] = 1
        out[split] = (sub["patch_id"].to_numpy(), y)
    return names, out


def compress_like_phase3(enc, emb, split):
    """Apply every existing Phase 3 and Phase 4b fit of this encoder to new rows."""
    for var in sorted(os.listdir(os.path.join(C.COMP, enc))):
        d = os.path.join(C.COMP, enc, var)
        if not os.path.isdir(d):
            continue
        base = var.replace("_int8", "")
        if base == "full":
            z = emb
        else:
            p = np.load(os.path.join(C.COMP, enc, f"{base}.npz"))
            z = ((emb - p["mean"]) @ p["components"].T).astype(np.float32)
        if var.endswith("_int8"):
            s = np.load(os.path.join(d, "scale.npy"))
            z = np.clip(np.rint(z / s), -127, 127).astype(np.int8)
        np.save(os.path.join(d, f"ben_{split}.npy"), z)
        qroot = os.path.join(C.QUANT, enc)
        if var.endswith("_int8") or not os.path.isdir(qroot):
            continue
        for qv in sorted(os.listdir(qroot)):
            if qv.startswith(base + "_q"):
                edges = np.load(os.path.join(qroot, qv, "edges.npy"))
                codes = np.empty(z.shape, np.uint8)
                for j in range(z.shape[1]):
                    codes[:, j] = np.searchsorted(edges[j], z[:, j], side="right")
                np.save(os.path.join(qroot, qv, f"ben_{split}.npy"), codes)


def main():
    import torch
    from embed import ENCODERS, build_encoder
    from probes import tile_stats

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", nargs=3, type=int, default=[30000, 10000, 10000], help="train val test")
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compress-only", action="store_true",
                    help="apply an encoder's Phase 3/4b fits to existing data/embeddings/<enc>/ben_*.npy "
                         "(for Phase 7's own encoders, whose BigEarthNet embeddings come from own_encoder.py)")
    args = ap.parse_args()

    if args.compress_only:
        for e in args.encoders:
            for split in ("train", "val", "test"):
                compress_like_phase3(e, np.load(os.path.join(C.EMB, e, f"ben_{split}.npy")), split)
            print(f"compressed BigEarthNet embeddings for {e}")
        return

    os.makedirs(OUT, exist_ok=True)
    names, subset = select_subset(args.n, args.seed)
    with open(os.path.join(OUT, "label_names.json"), "w") as fh:
        json.dump(names, fh, indent=2)
    models = {e: build_encoder(e, args.device, args.seed) for e in args.encoders}

    for split, (ids, y) in subset.items():
        np.save(os.path.join(OUT, f"{split}_y.npy"), y)
        np.save(os.path.join(OUT, f"{split}_patch_ids.npy"), ids)
        print(f"BigEarthNet {split}: {len(ids)} patches, mean {y.sum(1).mean():.2f} labels per patch", flush=True)
        dl = torch.utils.data.DataLoader(BenDataset(ids), batch_size=args.batch_size, num_workers=args.workers,
                                         collate_fn=stack_batch)
        embs = {e: [] for e in models}
        stats = []
        t0 = time.time()
        with torch.inference_mode():
            for i, batch in enumerate(dl):
                stats.append(tile_stats(batch))
                t = torch.from_numpy(batch.astype(np.float32)).div_(10000.0).to(args.device)
                for e, m in models.items():
                    embs[e].append(m(t).float().cpu().numpy())
                if i % 20 == 0:
                    print(f"  {min((i + 1) * args.batch_size, len(ids))}/{len(ids)} ({time.time() - t0:.0f}s)",
                          flush=True)
        np.save(os.path.join(C.COMP, "pixelstats", "full", f"ben_{split}.npy"), np.concatenate(stats))
        for e in models:
            emb = np.concatenate(embs[e])
            np.save(os.path.join(C.EMB, e, f"ben_{split}.npy"), emb)
            compress_like_phase3(e, emb, split)
        print(f"  done {split} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
