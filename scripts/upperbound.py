"""Phase 4b: image-level upper bound on leakage.

A ResNet-18 trained from scratch on raw 64x64x13 GloSoFarID tiles ("solar farm present"). It sees
the image itself, not an embedding, so its test AUC is how much leakage the data allows at all.
ResNet-18 matches the audited ResNet18, so "image vs embedding" compares like with like.

Training: BCE with positive-class weight, AdamW (lr 1e-3, wd 0.05), cosine schedule with warm-up,
8 dihedral augmentations (flips, 90-degree rotations: no resizing, no colour changes), model selection
by val AUC with early stopping. Same patch-level split as every other attacker.

Outputs: a row in results/phase4b/runs.csv (encoder image_resnet18, variant raw, task leakage,
attacker cnn_scratch) and test scores in results/preds/phase4b/seed<k>/image_resnet18/raw/leakage_cnn_scratch.npz

Usage
  python scripts/upperbound.py
  python scripts/upperbound.py --epochs 3        # quick test
"""
import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attack4b import FIELDS  # noqa: E402  (imports lightgbm before torch)
import common as C  # noqa: E402
import imagemodels as M  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402


def scores(model, batches):
    model.eval()
    out = []
    with torch.no_grad():
        for _, x, _ in batches.ordered(1024):
            out.append(torch.sigmoid(model(x)[:, 0]).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", default="results/phase4b/runs.csv")
    ap.add_argument("--preds", default="results/preds/phase4b")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sp = C.load_splits(args.split_seed)
    y = sp["y"].astype(np.float32)
    rows = [np.where(sp["split"] == k)[0] for k in range(3)]
    tr, va, te = (M.TileBatches(r, args.device) for r in rows)

    model = M.resnet18_classifier().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)
    steps = args.epochs * math.ceil(len(rows[0]) / args.batch)
    warm = math.ceil(len(rows[0]) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    pos = float(y[rows[0]].mean())
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, dtype=torch.float32,
                                                               device=args.device))
    yt = torch.tensor(y, device=args.device)

    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for r, x, _ in tr.epoch(args.batch, rng):
            opt.zero_grad()
            lossf(model(x)[:, 0], yt[r]).backward()
            opt.step()
            sched.step()
        val = roc_auc_score(y[rows[1]], scores(model, va))
        print(f"epoch {ep + 1:>2}: val AUC {val:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val > best + 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    s = scores(model, te)
    test = roc_auc_score(y[rows[2]], s)
    print(f"image-level upper bound: val AUC {best:.4f}, test AUC {test:.4f}")

    pd = os.path.join(args.preds, f"seed{args.split_seed}", "image_resnet18", "raw")
    os.makedirs(pd, exist_ok=True)
    np.savez_compressed(os.path.join(pd, "leakage_cnn_scratch.npz"), scores=s, y=y[rows[2]].astype(np.int64),
                        rows=rows[2])
    os.makedirs("results/models", exist_ok=True)
    torch.save(best_state, f"results/models/upperbound_resnet18_seed{args.split_seed}.pt")
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"split_seed": args.split_seed, "encoder": "image_resnet18", "variant": "raw", "task": "leakage",
                    "attacker": "cnn_scratch", "metric": "roc_auc", "val_score": round(best, 4),
                    "test_score": round(float(test), 4), "n_train": len(rows[0]), "n_test": len(rows[2]),
                    "dims": 13 * 64 * 64, "params": json.dumps({"epochs_run": ep + 1, "lr": 1e-3, "wd": 0.05}),
                    "seconds": round(time.time() - t0)})


if __name__ == "__main__":
    main()
