"""Phase 8b, step 1: features at every split point inside the encoders (pre-registered in
results/preregistration.md, section "Phase 8b split-point settings").

For each encoder and split (scripts/splitpoints.py) two payload forms, written as Phase 3-style variants of a
pseudo-encoder "<encoder>@<split>", so attack4b.py and localise.py read them unchanged:

  data/compressed/<encoder>@<split>/gap/{eurosat_train,eurosat_val,eurosat_test,glosofarid,ben_train,ben_val,ben_test}.npy
        pooled vector, float32 (EuroSAT, GloSoFarID, BigEarthNet)
  data/compressed/<encoder>@<split>/map4/glosofarid.npy
        spatial map (4x4 pooled; ViT patch tokens), float16, GloSoFarID only. A map wider than --pca-above dims
        (ResNet50 layer3, 16,384) is reduced by PCA to --pca-dims, fitted on split-seed-0 train rows only.
  data/compressed/<encoder>@<split>/meta.json

Inputs exactly as Phase 2 / 4b: native 64x64 tiles (EuroSAT, GloSoFarID tile cache), the Phase 4b BigEarthNet
subset (data/bigearthnet/npy/*_patch_ids.npy, same preprocessing via scripts/bigearthnet.py), divided by 10,000.
One pass per dataset feeds every encoder.

Checks (the script stops if one fails):
  - the final split's `gap` equals the encoder's own output on the same batch (max abs diff <= 1e-4 relative)
  - it matches the stored Phase 2 embedding for those rows (warning above 1e-2 relative: MPS numerics)

Resumable: a dataset already finished for an encoder (marker file) is skipped.

Usage
  python scripts/splits8b.py                                   # all 5 encoders, all datasets
  python scripts/splits8b.py --encoders resnet18_moco --datasets eurosat --limit 512   # quick test
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import splitpoints as S  # noqa: E402

ENCODERS = C.AUDITED + ["resnet18_random"]
BEN = "data/bigearthnet/npy"
ROOT = C.COMP  # --out overrides it (tests write elsewhere)


def pseudo(enc, split):
    return f"{enc}@{split}"


def sources(datasets, limit):
    """name -> (n_rows, batch iterator factory, forms). Each iterator yields uint16 arrays (B, 13, H, W)."""
    out = {}
    if "eurosat" in datasets:
        for s in ("train", "val", "test"):
            x = np.load(f"data/eurosat/npy/{s}_x.npy", mmap_mode="r")
            n = len(x) if limit is None else min(limit, len(x))
            out[f"eurosat_{s}"] = (n, ("array", x), ("gap",))
    if "glosofarid" in datasets:
        C.build_tile_cache()
        x = np.load(C.TILES_X, mmap_mode="r")
        n = len(x) if limit is None else min(limit, len(x))
        out["glosofarid"] = (n, ("array", x), ("gap", "map4"))
    if "ben" in datasets:
        for s in ("train", "val", "test"):
            f = os.path.join(BEN, f"{s}_patch_ids.npy")
            if not os.path.exists(f):
                raise SystemExit(f"Missing {f}: the Phase 4b BigEarthNet subset (scripts/bigearthnet.py) is needed")
            ids = np.load(f, allow_pickle=True)
            ids = ids if limit is None else ids[:limit]
            out[f"ben_{s}"] = (len(ids), ("ben", ids), ("gap",))
    return out


def batches(src, n, bs, workers):
    kind, obj = src
    if kind == "array":
        for i in range(0, n, bs):
            yield np.asarray(obj[i:min(i + bs, n)])
        return
    from bigearthnet import BenDataset, stack_batch
    dl = torch.utils.data.DataLoader(BenDataset(list(obj)), batch_size=bs, num_workers=workers,
                                     collate_fn=stack_batch)
    yield from dl


def done_marker(enc, name):
    return os.path.join(ROOT, pseudo(enc, "layer1" if S.ARCH[enc].startswith("resnet") else "block3"),
                        f".done_{name}")


def check_final(model, arch, t, feats_final_gap, enc, name, rows):
    with torch.inference_mode():
        ref = model(t).float()
    d = float((feats_final_gap - ref).abs().max())
    scale = float(ref.abs().max())
    assert d <= 1e-4 * max(1.0, scale), f"{enc}/{name}: final-split gap differs from the encoder by {d:.2e}"
    stored = os.path.join(C.EMB, enc, f"{name}.npy")
    if os.path.exists(stored):
        e = np.load(stored, mmap_mode="r")[rows]
        ds = float(np.abs(e - ref.cpu().numpy()).max())
        msg = f"  check {enc}/{name}: final split = encoder output (diff {d:.1e}); vs stored Phase 2 rows {ds:.1e}"
        if ds > 1e-2 * max(1.0, scale):
            msg += "  WARNING: larger than expected"
        print(msg, flush=True)


def reduce_map(enc, split, pca_dims, seed=0):
    """PCA a wide map4 to pca_dims, fitted on train rows of split seed 0 (at most 20,000, drawn at random)."""
    from sklearn.decomposition import PCA

    d = os.path.join(ROOT, pseudo(enc, split), "map4")
    raw = os.path.join(d, "glosofarid_raw.npy")
    x = np.load(raw, mmap_mode="r")
    tr = np.where(C.load_splits(0)["split"] == 0)[0]
    pick = np.sort(np.random.default_rng(seed).choice(tr, min(20000, len(tr)), replace=False))
    t0 = time.time()
    pca = PCA(n_components=pca_dims, svd_solver="randomized", random_state=seed).fit(
        np.asarray(x[pick], dtype=np.float32))
    out = np.lib.format.open_memmap(os.path.join(d, "glosofarid.npy"), "w+", np.float16, (len(x), pca_dims))
    for i in range(0, len(x), 4096):
        out[i:i + 4096] = pca.transform(np.asarray(x[i:i + 4096], dtype=np.float32)).astype(np.float16)
    out.flush()
    del out
    os.remove(raw)
    ev = float(pca.explained_variance_ratio_.sum())
    print(f"  {enc}@{split} map4: PCA {x.shape[1]} -> {pca_dims} dims on {len(pick)} train rows, "
          f"{ev:.1%} variance kept ({time.time() - t0:.0f}s)", flush=True)
    return {"pca_dims": pca_dims, "orig_dims": int(x.shape[1]), "pca_train_rows": int(len(pick)),
            "pca_variance_kept": round(ev, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--datasets", nargs="+", default=["eurosat", "glosofarid", "ben"],
                    choices=["eurosat", "glosofarid", "ben"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--pca-above", type=int, default=8192)
    ap.add_argument("--pca-dims", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None, help="rows per dataset (testing only)")
    ap.add_argument("--out", default=C.COMP, help="root for the pseudo-encoder folders (default data/compressed)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    global ROOT
    ROOT = args.out
    if args.limit is not None and os.path.abspath(ROOT) == os.path.abspath(C.COMP):
        raise SystemExit("--limit is for tests: also pass --out <scratch folder> so data/compressed stays intact")

    models = {e: S.build(e, args.device) for e in args.encoders}
    for name, (n, src, forms) in sources(args.datasets, args.limit).items():
        todo = [e for e in models if not os.path.exists(done_marker(e, name))]
        if not todo:
            print(f"== {name}: done for every encoder, skipped")
            continue
        print(f"== {name}: {n} rows, encoders {', '.join(todo)}", flush=True)
        outs = {}
        for e in todo:
            arch = S.ARCH[e]
            st = S.stages(models[e], arch)
            with torch.inference_mode():
                probe = torch.zeros((2, 13) + ((64, 64) if not name.startswith("ben") else (112, 112)),
                                    device=args.device)
                h = probe
                for split, f in st:
                    h = f(h)
                    for form in forms:
                        dims = (S.gap if form == "gap" else S.map4)(models[e], arch, h).shape[1]
                        d = os.path.join(ROOT, pseudo(e, split), form)
                        os.makedirs(d, exist_ok=True)
                        wide = form == "map4" and dims > args.pca_above
                        fn = os.path.join(d, f"{name}_raw.npy" if wide else f"{name}.npy")
                        dt = np.float16 if form == "map4" else np.float32
                        outs[(e, split, form)] = np.lib.format.open_memmap(fn, "w+", dt, (n, dims))
        t0, i = time.time(), 0
        for bi, batch in enumerate(batches(src, n, args.batch_size, args.workers)):
            t = torch.from_numpy(batch.astype(np.float32)).div_(10000.0).to(args.device)
            b = len(batch)
            for e in todo:
                m, arch = models[e], S.ARCH[e]
                with torch.inference_mode():
                    h = t
                    for split, f in S.stages(m, arch):
                        h = f(h)
                        for form in forms:
                            v = (S.gap if form == "gap" else S.map4)(m, arch, h).float()
                            outs[(e, split, form)][i:i + b] = v.cpu().numpy()
                    if bi == 0:
                        check_final(m, arch, t, S.gap(m, arch, h).float(), e, name, np.arange(b))
            i += b
            if bi % 20 == 0 or i == n:
                print(f"    {i}/{n} ({time.time() - t0:.0f}s)", flush=True)
        assert i == n, (name, i, n)
        for k in list(outs):
            outs[k].flush()
        outs.clear()
        for e in todo:
            arch = S.ARCH[e]
            for split in S.SPLITS[arch]:
                md = os.path.join(ROOT, pseudo(e, split))
                mf = os.path.join(md, "meta.json")
                meta = json.load(open(mf)) if os.path.exists(mf) else {"encoder": e, "split": split, "arch": arch}
                for form in forms:
                    d = os.path.join(md, form)
                    if os.path.exists(os.path.join(d, f"{name}_raw.npy")) and args.limit is None:
                        meta[f"{form}_reduction"] = reduce_map(e, split, args.pca_dims)
                    elif os.path.exists(os.path.join(d, f"{name}_raw.npy")):
                        os.replace(os.path.join(d, f"{name}_raw.npy"), os.path.join(d, f"{name}.npy"))
                    meta[f"{form}_dims"] = int(np.load(os.path.join(d, f"{name}.npy"), mmap_mode="r").shape[1])
                meta.setdefault("rows", {})[name] = n
                json.dump(meta, open(mf, "w"), indent=2)
            open(done_marker(e, name), "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
        print(f"  done {name} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
