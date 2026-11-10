"""Phase 5: panel segmenter (U-Net) trained on REAL GloSoFarID tiles and masks.

It is the measuring instrument for every inverter: run on a reconstruction, it says whether panels
appear (detection) and where (mask IoU). Trained once, frozen, applied identically to all inverters.

Training: Dice + BCE (positive weight 3), AdamW, cosine schedule, 8 dihedral augmentations applied
to image and mask together, selection by val mean IoU on positive tiles. The decision threshold is
chosen on val. Test scores on real images give the ceiling for "panels visible in a reconstruction".

Outputs
  results/models/segmenter_unet.pt
  results/phase5/segmenter.json   threshold, val/test IoU and detection AUC on real tiles,
                                  plus the location-prior baseline on the same test tiles

Usage
  python scripts/segmenter.py
  python scripts/segmenter.py --epochs 2     # quick test
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import imagemodels as M  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

MODEL = "results/models/segmenter_unet.pt"
INFO = "results/phase5/segmenter.json"
THRESHOLDS = np.round(np.arange(0.2, 0.81, 0.05), 2)


def tile_iou(prob, mask, thr):
    """Per-tile IoU of (prob > thr) vs mask; tiles with an empty union count as 1."""
    p = prob > thr
    m = mask > 0.5
    inter = (p & m).sum((-2, -1))
    union = (p | m).sum((-2, -1))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def predict(model, batches, batch=512):
    """Probabilities (N, 64, 64) float16 and true masks, in row order."""
    model.eval()
    probs, masks = [], []
    with torch.no_grad():
        for _, x, m in batches.ordered(batch):
            probs.append(torch.sigmoid(model(x))[:, 0].half().cpu().numpy())
            masks.append(m[:, 0].to(torch.uint8).cpu().numpy())
    return np.concatenate(probs), np.concatenate(masks)


def scores(prob, mask, y, thr):
    """Detection AUC (max probability per tile) on all tiles; mean IoU on positive tiles."""
    det = prob.reshape(len(prob), -1).max(1).astype(np.float32)
    pos = y == 1
    return {"detection_auc": float(roc_auc_score(y, det)),
            "mean_iou_positive": float(tile_iou(prob[pos].astype(np.float32), mask[pos], thr).mean())}


def load(device):
    m = M.UNet().to(device)
    m.load_state_dict(torch.load(MODEL, map_location=device))
    return m.eval(), json.load(open(INFO))["threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sp = C.load_splits(0)
    y = sp["y"]
    rows = [np.where(sp["split"] == k)[0] for k in range(3)]
    tr, va, te = (M.TileBatches(r, args.device, with_mask=True) for r in rows)

    model = M.UNet().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(rows[0]) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    pw = torch.tensor(3.0, device=args.device)

    best, best_state, bad, t0 = -1.0, None, 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for _, x, m in tr.epoch(args.batch, rng):
            opt.zero_grad()
            M.dice_bce(model(x), m, pw).backward()
            opt.step()
            sched.step()
        prob, mask = predict(model, va)
        val = scores(prob, mask, y[rows[1]], 0.5)
        print(f"epoch {ep + 1:>2}: val IoU(positive) {val['mean_iou_positive']:.4f}, "
              f"detection AUC {val['detection_auc']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val["mean_iou_positive"] > best + 1e-4:
            best, bad = val["mean_iou_positive"], 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    prob, mask = predict(model, va)
    pos = y[rows[1]] == 1
    thr = float(max(THRESHOLDS, key=lambda t: tile_iou(prob[pos].astype(np.float32), mask[pos], t).mean()))
    val = scores(prob, mask, y[rows[1]], thr)
    prob, mask = predict(model, te)
    test = scores(prob, mask, y[rows[2]], thr)

    # location prior on the same test tiles: mean train mask for that tile position, threshold chosen on val
    mask_prior, _ = C.position_prior(sp)
    vpos, tpos = rows[1][pos], rows[2][y[rows[2]] == 1]
    vm = np.load(C.TILES_MASK, mmap_mode="r")
    pthr = float(max(THRESHOLDS, key=lambda t: tile_iou(mask_prior[sp["tile_idx"][vpos]],
                                                        np.asarray(vm[np.sort(vpos)]), t).mean()))
    tpos_sorted = np.sort(tpos)
    prior_iou = float(tile_iou(mask_prior[sp["tile_idx"][tpos_sorted]], np.asarray(vm[tpos_sorted]), pthr).mean())

    os.makedirs(os.path.dirname(MODEL), exist_ok=True)
    os.makedirs(os.path.dirname(INFO), exist_ok=True)
    torch.save(best_state, MODEL)
    info = {"threshold": thr, "val_real": val, "test_real": test,
            "test_location_prior": {"mean_iou_positive": prior_iou, "threshold": pthr},
            "epochs_run": ep + 1, "seconds": round(time.time() - t0)}
    with open(INFO, "w") as fh:
        json.dump(info, fh, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
