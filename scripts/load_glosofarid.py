"""Read GloSoFarID .tfrecord.gz shards straight into numpy (no .tif conversion).

Writes, per shard, into --out:
  <shard>_x.npy     uint16 (N, 13, 256, 256)  Sentinel-2 B1..B12 incl. B8A
  <shard>_y.npy     uint8  (N,)               1 if the patch has any solar-panel pixel
  <shard>_mask.npy  uint8  (N, 256, 256)      per-pixel mask (only with --masks)

Usage:
  python scripts/load_glosofarid.py data/glosofarid/solar_2023_global.tfrecord.gz --out data/glosofarid/npy
"""
import argparse
import os

import numpy as np

BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B10", "B11", "B12"]
RESPONSE = "solar_panel"
SIZE = 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help=".tfrecord.gz shards")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--masks", action="store_true", help="also save per-pixel masks")
    args = ap.parse_args()

    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")  # CPU is fine for parsing
    spec = {k: tf.io.FixedLenFeature([SIZE, SIZE], tf.float32) for k in BANDS + [RESPONSE]}
    os.makedirs(args.out, exist_ok=True)

    for path in args.files:
        name = os.path.basename(path).split(".")[0]
        ds = tf.data.TFRecordDataset(path, compression_type="GZIP")
        n = sum(1 for _ in ds)  # first pass: count records so we can preallocate on disk

        x = np.lib.format.open_memmap(
            os.path.join(args.out, f"{name}_x.npy"), mode="w+",
            dtype=np.uint16, shape=(n, len(BANDS), SIZE, SIZE))
        y = np.zeros(n, dtype=np.uint8)
        m = None
        if args.masks:
            m = np.lib.format.open_memmap(
                os.path.join(args.out, f"{name}_mask.npy"), mode="w+",
                dtype=np.uint8, shape=(n, SIZE, SIZE))

        scale, lo, hi, clipped, mask_vals = None, np.inf, -np.inf, 0, set()
        parsed = ds.map(lambda r: tf.io.parse_single_example(r, spec)).as_numpy_iterator()
        for i, rec in enumerate(parsed):
            bands = np.stack([rec[b] for b in BANDS])
            if scale is None:  # reflectance stored as 0-1 floats -> scale to 0-10000
                scale = 10000.0 if bands.max() <= 1.5 else 1.0
            bands = bands * scale
            lo, hi = min(lo, bands.min()), max(hi, bands.max())
            clipped += int(((bands < 0) | (bands > 65535)).sum())
            x[i] = np.clip(np.rint(bands), 0, 65535)

            raw_mask = rec[RESPONSE]
            if len(mask_vals) < 10:
                mask_vals |= set(np.unique(raw_mask).tolist())
            mask = raw_mask > 0.5
            y[i] = mask.any()
            if m is not None:
                m[i] = mask
            if (i + 1) % 500 == 0:
                print(f"  {name}: {i + 1}/{n}")

        x.flush()
        if m is not None:
            m.flush()
        np.save(os.path.join(args.out, f"{name}_y.npy"), y)
        print(f"{name}: {n} patches, {int(y.sum())} with solar, {n - int(y.sum())} without | "
              f"scale x{scale:g}, band range [{lo:.1f}, {hi:.1f}], {clipped} values clipped | "
              f"raw mask values seen: {sorted(mask_vals)}")


if __name__ == "__main__":
    main()
