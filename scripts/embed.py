"""Phase 2: embed EuroSAT and GloSoFarID with frozen encoders, at native resolution.

No resizing anywhere. Images go in at 64x64, 10 m/pixel:
  - EuroSAT patches are already 64x64.
  - Each 256x256 GloSoFarID patch is cut into 16 non-overlapping 64x64 tiles,
    each labelled 1 if its mask has any solar-panel pixel.
Only the weights' normalisation (divide by 10,000) is applied. The torchgeo weight
`transforms` are NOT used, because they resize to 256 and centre-crop to 224.

Outputs (under --out, default data/embeddings):
  splits_glosofarid.npz       per tile: y, patch_id, tile_idx (0-15, row-major), split (0 train, 1 val, 2 test)
  <encoder>/eurosat_<split>.npy   (N, D) float32, rows aligned with data/eurosat/npy/<split>_y.npy
  <encoder>/glosofarid.npy        (N_tiles, D) float32, rows aligned with splits_glosofarid.npz
  <encoder>/meta.json

EuroSAT is first cached as uint16 arrays in data/eurosat/npy/ (official torchgeo splits),
reordered to the encoders' band order.

Usage:
  python scripts/embed.py                         # all encoders, both datasets
  python scripts/embed.py --encoders resnet18_moco --sanity
"""
import argparse
import json
import os
import time

import numpy as np
import torch

# Band order every encoder expects (SSL4EO-S12, Sentinel-2 L1C). GloSoFarID is already in it.
ENCODER_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
TILE = 64
TILES_PER_SIDE = 256 // TILE  # 4 -> 16 tiles per GloSoFarID patch
SCALE = 10000.0

ENCODERS = {
    # name: (constructor, weights enum name, weights member)
    "resnet18_moco": ("resnet18", "ResNet18_Weights", "SENTINEL2_ALL_MOCO"),
    "vits16_moco": ("vit_small_patch16_224", "ViTSmall16_Weights", "SENTINEL2_ALL_MOCO"),
    "resnet50_dino": ("resnet50", "ResNet50_Weights", "SENTINEL2_ALL_DINO"),
    "vits16_dino": ("vit_small_patch16_224", "ViTSmall16_Weights", "SENTINEL2_ALL_DINO"),
    # Phase 4 control: same architecture as resnet18_moco, random weights
    "resnet18_random": ("resnet18", None, None),
}


def build_encoder(name, device, seed):
    import torchgeo.models as tgm

    ctor_name, weights_enum, member = ENCODERS[name]
    ctor = getattr(tgm, ctor_name)
    kwargs = {"num_classes": 0}  # return pooled features, no classification head
    if ctor_name.startswith("vit"):
        # Keep the pretrained 14x14 positional grid; timm resamples it to 4x4 for 64x64 input.
        kwargs["dynamic_img_size"] = True
    if weights_enum is None:
        torch.manual_seed(seed)
        model = ctor(weights=None, in_chans=len(ENCODER_BANDS), **kwargs)
    else:
        weights = getattr(getattr(tgm, weights_enum), member)
        assert weights.meta["in_chans"] == len(ENCODER_BANDS)
        model = ctor(weights=weights, **kwargs)
    return model.eval().to(device)


@torch.inference_mode()
def embed_array(model, x, device, batch_size, to_tiles=False):
    """x: uint16 array (N, 13, H, W), may be a memmap. Returns float32 (N or N*16, D)."""
    out = []
    t0 = time.time()
    for i in range(0, len(x), batch_size):
        batch = np.asarray(x[i:i + batch_size])
        if to_tiles:
            batch = patches_to_tiles(batch)
        t = torch.from_numpy(batch.astype(np.float32)).div_(SCALE).to(device)
        out.append(model(t).float().cpu().numpy())
        done = min(i + batch_size, len(x))
        if done == len(x) or (i // batch_size) % 20 == 0:
            print(f"    {done}/{len(x)}  ({time.time() - t0:.0f}s)", flush=True)
    return np.concatenate(out)


def patches_to_tiles(a):
    """(B, C, 256, 256) -> (B*16, C, 64, 64), tiles row-major within each patch."""
    b, c = a.shape[:2]
    a = a.reshape(b, c, TILES_PER_SIDE, TILE, TILES_PER_SIDE, TILE)
    return a.transpose(0, 2, 4, 1, 3, 5).reshape(b * TILES_PER_SIDE ** 2, c, TILE, TILE)


def cache_eurosat(root, cache_dir):
    """Read EuroSAT once via torchgeo (official splits) into uint16 arrays in encoder band order."""
    from torchgeo.datasets import EuroSAT

    os.makedirs(cache_dir, exist_ok=True)
    for split in ("train", "val", "test"):
        xf, yf = os.path.join(cache_dir, f"{split}_x.npy"), os.path.join(cache_dir, f"{split}_y.npy")
        if os.path.exists(xf) and os.path.exists(yf):
            continue
        # download=True only fetches the small split files if missing; the images are already extracted.
        ds = EuroSAT(root=root, split=split, bands=ENCODER_BANDS, download=True)
        x = np.empty((len(ds), len(ENCODER_BANDS), TILE, TILE), dtype=np.uint16)
        y = np.empty(len(ds), dtype=np.int64)
        for i in range(len(ds)):
            s = ds[i]
            img = s["image"].numpy()
            assert img.shape == x.shape[1:], f"unexpected EuroSAT shape {img.shape}"
            x[i] = np.clip(np.rint(img), 0, 65535)
            y[i] = int(s["label"])
        np.save(xf, x)
        np.save(yf, y)
        print(f"  cached EuroSAT {split}: {x.shape}, {len(np.unique(y))} classes")


def make_glosofarid_splits(npy_dir, out_file, seed, frac=(0.7, 0.15, 0.15)):
    """Tile labels + a train/val/test split by ORIGINAL PATCH, so neighbouring tiles never straddle splits."""
    if os.path.exists(out_file):
        return dict(np.load(out_file))
    mask_file = os.path.join(npy_dir, "solar_2023_global_mask.npy")
    if not os.path.exists(mask_file):
        raise SystemExit(f"Missing {mask_file}: rerun scripts/load_glosofarid.py with --masks first.")
    masks = np.load(mask_file, mmap_mode="r")
    n = len(masks)
    y = np.empty(n * TILES_PER_SIDE ** 2, dtype=np.uint8)
    for i in range(0, n, 256):
        m = np.asarray(masks[i:i + 256]).astype(bool)
        b = len(m)
        tiles = m.reshape(b, TILES_PER_SIDE, TILE, TILES_PER_SIDE, TILE).any(axis=(2, 4))
        y[i * 16:(i + b) * 16] = tiles.reshape(-1)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)
    patch_split = np.empty(n, dtype=np.uint8)
    patch_split[order[:n_tr]] = 0
    patch_split[order[n_tr:n_tr + n_va]] = 1
    patch_split[order[n_tr + n_va:]] = 2

    patch_id = np.repeat(np.arange(n), 16)
    splits = {"y": y, "patch_id": patch_id, "tile_idx": np.tile(np.arange(16), n),
              "split": patch_split[patch_id]}
    np.savez(out_file, **splits)
    for k, name in enumerate(("train", "val", "test")):
        sel = splits["split"] == k
        print(f"  GloSoFarID {name}: {sel.sum()} tiles, {int(y[sel].sum())} with solar ({y[sel].mean():.1%})")
    return splits


def sanity_probe(enc_dir, eurosat_dir):
    """Quick EuroSAT linear probe (train -> val) to check an encoder gives sensible features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    xtr, xva = (np.load(os.path.join(enc_dir, f"eurosat_{s}.npy")) for s in ("train", "val"))
    ytr, yva = (np.load(os.path.join(eurosat_dir, f"{s}_y.npy")) for s in ("train", "val"))
    sc = StandardScaler().fit(xtr)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(xtr), ytr)
    return clf.score(sc.transform(xva), yva)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(ENCODERS), choices=list(ENCODERS))
    ap.add_argument("--datasets", nargs="+", default=["eurosat", "glosofarid"], choices=["eurosat", "glosofarid"])
    ap.add_argument("--out", default="data/embeddings")
    ap.add_argument("--eurosat-root", default="data/eurosat")
    ap.add_argument("--glosofarid-npy", default="data/glosofarid/npy")
    ap.add_argument("--batch-size", type=int, default=256, help="images per batch (GloSoFarID: patches, x16 tiles)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sanity", action="store_true", help="EuroSAT linear-probe check after each encoder")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    eurosat_cache = os.path.join(args.eurosat_root, "npy")
    if "eurosat" in args.datasets:
        print("EuroSAT cache")
        cache_eurosat(args.eurosat_root, eurosat_cache)
    if "glosofarid" in args.datasets:
        print("GloSoFarID tiles and splits")
        make_glosofarid_splits(args.glosofarid_npy, os.path.join(args.out, "splits_glosofarid.npz"), args.seed)

    for name in args.encoders:
        enc_dir = os.path.join(args.out, name)
        os.makedirs(enc_dir, exist_ok=True)
        print(f"\n== {name} on {args.device}")
        model = build_encoder(name, args.device, args.seed)
        dim = None

        if "eurosat" in args.datasets:
            for split in ("train", "val", "test"):
                f = os.path.join(enc_dir, f"eurosat_{split}.npy")
                if os.path.exists(f) and not args.overwrite:
                    continue
                print(f"  EuroSAT {split}")
                x = np.load(os.path.join(eurosat_cache, f"{split}_x.npy"), mmap_mode="r")
                e = embed_array(model, x, args.device, args.batch_size)
                np.save(f, e)
                dim = e.shape[1]

        if "glosofarid" in args.datasets:
            f = os.path.join(enc_dir, "glosofarid.npy")
            if not os.path.exists(f) or args.overwrite:
                print("  GloSoFarID (16 tiles per patch)")
                x = np.load(os.path.join(args.glosofarid_npy, "solar_2023_global_x.npy"), mmap_mode="r")
                e = embed_array(model, x, args.device, max(1, args.batch_size // 16), to_tiles=True)
                np.save(f, e)
                dim = e.shape[1]

        meta = {"encoder": name, "spec": ENCODERS[name], "dim": dim, "bands": ENCODER_BANDS,
                "input": f"{TILE}x{TILE}, native 10 m, divided by {SCALE:g}, no resizing",
                "seed": args.seed, "device": args.device}
        if args.sanity and "eurosat" in args.datasets:
            meta["sanity_eurosat_val_acc"] = acc = sanity_probe(enc_dir, eurosat_cache)
            print(f"  sanity: EuroSAT val accuracy {acc:.1%}")
        with open(os.path.join(enc_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
        del model
        if args.device == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
