"""Phase 8a: on-board compute and payload size at every split point (descriptive, no training).

For each architecture x split (scripts/splitpoints.py):
  head / tail parameters        parameters of the stages before / after the split
  head / tail / total FLOPs     torch.utils.flop_counter.FlopCounterMode, batch 1, 13x64x64 input
                                (a multiply-add counts as 2 FLOPs)
  peak activation               largest stage output in the head, bytes (fp32, batch 1)
  payload bytes                 full feature map (fp32 / fp16 / int8), pooled vector `gap`, spatial `map4`
  head latency                  median and IQR of the head's forward time, MPS and CPU, batch 1 and 64
                                (20 warm-up + 100 timed runs; MPS synchronised before every timer read)
Raw baseline (rows with arch = raw): the 13x64x64 uint16 tile and its zlib-compressed size (mean over 1,000
GloSoFarID test tiles).

Weights do not change FLOPs, sizes or (to within noise) latency, so each architecture is built once: the
Phase 2 encoders as loaded there, and the Phase 7 own ViT-S/8 with random weights.

Sanity checks (the script stops if one fails):
  - FLOP counter on torchvision ResNet-18, 3x224x224: 3.4-3.9 GFLOPs (published 1.8 GMACs)
  - running the stages reproduces each encoder's own output (max abs diff < 1e-4 relative)
  - head + tail FLOPs = full-model FLOPs, and head + tail parameters = encoder parameters, at every split

Output: results/phase8/compute.csv

Usage
  python scripts/compute8a.py                  # everything (latency ~10-20 min)
  python scripts/compute8a.py --no-latency     # FLOPs, parameters, bytes only (~1 min)
  python scripts/compute8a.py --reps 10        # quick latency test
"""
import argparse
import csv
import os
import sys
import time
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

import splitpoints as S  # noqa: E402

REPRESENTATIVE = {"resnet18": "resnet18_moco", "resnet50": "resnet50_dino", "vits16": "vits16_moco",
                  "own_vits8": "own_vit"}
USED_BY = {"resnet18": "resnet18_moco resnet18_random", "resnet50": "resnet50_dino",
           "vits16": "vits16_moco vits16_dino", "own_vits8": "own_vit_baseline own_vit_defended"}
RAW_BYTES = 13 * 64 * 64 * 2
FIELDS = ["arch", "encoders", "split", "split_index", "is_final", "head_params", "tail_params", "head_flops",
          "tail_flops", "total_flops", "head_flops_share", "peak_activation_bytes", "map_shape", "map_elems",
          "map_fp32_bytes", "map_fp16_bytes", "map_int8_bytes", "map_vs_raw_fp32", "gap_dims", "gap_fp32_bytes",
          "gap_int8_bytes", "map4_dims", "raw_bytes", "raw_zlib_bytes"]
LAT = ["lat_{dev}_b{b}_ms", "lat_{dev}_b{b}_iqr_ms"]


def flops(fn, *args):
    with FlopCounterMode(display=False) as fc:
        with torch.inference_mode():
            fn(*args)
    return int(fc.get_total_flops())


def n_params(modules):
    seen, total = set(), 0
    for m in modules:
        for p in m.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def stage_modules(model, arch):
    """Modules owned by each stage (for parameter counts), and the modules of the tail after the last stage."""
    if arch.startswith("resnet"):
        mods = [[model.conv1, model.bn1, model.layer1], [model.layer2], [model.layer3], [model.layer4]]
        return mods, [model.fc]
    if arch == "vits16":
        emb = [model.patch_embed, model.norm_pre]
        pos = [p for p in (getattr(model, "pos_embed", None), getattr(model, "cls_token", None)) if p is not None]
        mods = [emb + list(model.blocks[0:3]), list(model.blocks[3:6]), list(model.blocks[6:9]),
                list(model.blocks[9:12])]
        return mods, [model.norm, model.fc_norm, model.head], pos
    mods = [[model.patch] + list(model.blocks[0:3]), list(model.blocks[3:6]), list(model.blocks[6:9]),
            list(model.blocks[9:12])]
    return mods, [model.norm]


def encoder_params(model, arch):
    if arch == "own_vits8":  # encoder only (no decoder, utility head or adversary)
        return n_params([model.patch, model.blocks, model.norm])
    return sum(p.numel() for p in model.parameters())


def head_fn(st, k):
    def f(x):
        for _, g in st[:k + 1]:
            x = g(x)
        return x
    return f


def tail_fn(model, arch, st, k):
    def f(h):
        for _, g in st[k + 1:]:
            h = g(h)
        return S.finish(model, arch, h)
    return f


def sync(dev):
    if dev == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def latency(fn, x, dev, reps, warm=20):
    for _ in range(warm):
        fn(x)
    sync(dev)
    ts = []
    for _ in range(reps):
        sync(dev)
        t0 = time.perf_counter()
        fn(x)
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1000)
    q1, med, q3 = np.percentile(ts, [25, 50, 75])
    return float(med), float(q3 - q1)


def sanity_flop_counter():
    import torchvision
    m = torchvision.models.resnet18().eval()
    f = flops(m, torch.zeros(1, 3, 224, 224))
    assert 3.4e9 <= f <= 3.9e9, f"FLOP counter check failed: ResNet-18 at 224 gave {f / 1e9:.2f} GFLOPs"
    print(f"  FLOP counter check: torchvision ResNet-18, 3x224x224 = {f / 1e9:.2f} GFLOPs (expected ~3.6)")


def raw_zlib(n=1000, seed=0):
    C.build_tile_cache()
    sp = C.load_splits(0)
    rows = np.where(sp["split"] == 2)[0]
    rows = np.sort(np.random.default_rng(seed).choice(rows, min(n, len(rows)), replace=False))
    x = np.load(C.TILES_X, mmap_mode="r")
    return float(np.mean([len(zlib.compress(np.ascontiguousarray(x[r]).tobytes(), 6)) for r in rows]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(REPRESENTATIVE))
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 64])
    ap.add_argument("--devices", nargs="+", default=["mps", "cpu"] if torch.backends.mps.is_available() else ["cpu"])
    ap.add_argument("--out", default="results/phase8/compute.csv")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.manual_seed(0)

    sanity_flop_counter()
    zl = raw_zlib()
    print(f"  raw tile {RAW_BYTES} B, zlib-compressed {zl:.0f} B ({zl / RAW_BYTES:.2f}x)")
    lat_fields = [] if args.no_latency else [f.format(dev=d, b=b) for d in args.devices for b in args.batches
                                               for f in LAT]
    rows = [{"arch": "raw", "encoders": "", "split": "raw", "raw_bytes": RAW_BYTES, "raw_zlib_bytes": round(zl),
             "map_fp32_bytes": RAW_BYTES, "map_vs_raw_fp32": 1.0}]

    x1 = torch.rand(1, 13, 64, 64) * 0.3
    for arch in args.archs:
        model = S.build(REPRESENTATIVE[arch], "cpu")
        diff, scale = S.check_stages(model, arch, torch.rand(4, 13, 64, 64) * 0.3)
        assert diff <= 1e-4 * max(1.0, scale), f"{arch}: stages differ from the encoder ({diff:.2e})"
        st = S.stages(model, arch)
        full = (lambda x: model.encode(x)) if arch == "own_vits8" else model
        total = flops(full, x1)
        sm = stage_modules(model, arch)
        mods, tail_mods = sm[0], sm[1]
        extra = sum(p.numel() for p in sm[2]) if len(sm) > 2 else 0  # ViT position embedding + CLS token
        p_stage = [n_params(m) for m in mods]
        p_tail = n_params(tail_mods)
        p_total = encoder_params(model, arch)
        assert sum(p_stage) + p_tail + extra == p_total, f"{arch}: parameter split {sum(p_stage) + p_tail + extra} != {p_total}"

        with torch.inference_mode():
            outs, h = [], x1
            for _, g in st:
                h = g(h)
                outs.append(h)
        for k, (name, _) in enumerate(st):
            hf, tf = flops(head_fn(st, k), x1), flops(tail_fn(model, arch, st, k), outs[k])
            assert abs(hf + tf - total) <= 1e-6 * total + 1e3, f"{arch}/{name}: head+tail {hf + tf} != {total}"
            m = outs[k]
            with torch.inference_mode():
                g_dims = S.gap(model, arch, m).shape[1]
                m4 = S.map4(model, arch, m).shape[1]
            elems = int(m[0].numel())
            r = {"arch": arch, "encoders": USED_BY[arch], "split": name, "split_index": k + 1,
                 "is_final": k == len(st) - 1,
                 "head_params": sum(p_stage[:k + 1]) + extra, "tail_params": sum(p_stage[k + 1:]) + p_tail,
                 "head_flops": hf, "tail_flops": tf, "total_flops": total, "head_flops_share": round(hf / total, 4),
                 "peak_activation_bytes": max(int(o[0].numel()) * 4 for o in outs[:k + 1]),
                 "map_shape": "x".join(map(str, m.shape[1:])), "map_elems": elems,
                 "map_fp32_bytes": elems * 4, "map_fp16_bytes": elems * 2, "map_int8_bytes": elems,
                 "map_vs_raw_fp32": round(elems * 4 / RAW_BYTES, 4), "gap_dims": g_dims,
                 "gap_fp32_bytes": g_dims * 4, "gap_int8_bytes": g_dims, "map4_dims": m4, "raw_bytes": RAW_BYTES}
            if not args.no_latency:
                for dev in args.devices:
                    md = model.to(dev)
                    stg = S.stages(md, arch)
                    for b in args.batches:
                        xb = (torch.rand(b, 13, 64, 64) * 0.3).to(dev)
                        med, iqr = latency(head_fn(stg, k), xb, dev, args.reps)
                        r[f"lat_{dev}_b{b}_ms"], r[f"lat_{dev}_b{b}_iqr_ms"] = round(med, 3), round(iqr, 3)
                    model = md.to("cpu")
                    st = S.stages(model, arch)
            rows.append(r)
            print(f"  {arch:>9} {name:>7}: head {hf / 1e6:8.1f} MFLOPs ({hf / total:5.1%}), "
                  f"map {r['map_shape']:>10} = {elems * 4:>7} B fp32 ({r['map_vs_raw_fp32']:.2f}x raw), "
                  f"gap {g_dims}" + ("" if args.no_latency else
                                     f", MPS/CPU b1 {r.get('lat_mps_b1_ms', '-')}/{r.get('lat_cpu_b1_ms', '-')} ms"),
                  flush=True)

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + lat_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
