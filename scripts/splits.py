"""Phase 4b robustness: extra GloSoFarID train/val/test splits (by original patch, 70/15/15).

Seed 0 is the split every phase has used so far (data/embeddings/splits_glosofarid.npz).
Seeds 1 and 2 are written next to it as splits_glosofarid_seed<k>.npz. Embeddings do not change;
only which patches are train, val and test. Used for the primary comparisons at full size,
which need no refitting (no PCA).

Usage:
  python scripts/splits.py --seeds 1 2
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
from embed import make_glosofarid_splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2])
    args = ap.parse_args()
    for s in args.seeds:
        if s == 0:
            print("seed 0 is the original split; not rewritten")
            continue
        print(f"split seed {s}")
        make_glosofarid_splits("data/glosofarid/npy", C.splits_path(s), seed=s)


if __name__ == "__main__":
    main()
