"""Phase 6: localisation attack, embedding -> 64x64 panel mask.

Model: query-based transformer decoder (DETR / Mask2Former style)
  - memory: the embedding projected to 16 tokens of width 256 (+ learned token embeddings)
  - queries: 64 learned queries, one per cell of an 8x8 grid (+ learned 2D position embeddings)
  - 4 decoder layers: query self-attention, cross-attention to the memory, feed-forward (pre-norm)
  - head: queries reshaped to an 8x8x256 map, 3 upsampling conv stages (8 -> 16 -> 32 -> 64), 1-channel logits
Loss: Dice + focal (gamma 2). Trained on all train tiles, so it also detects (max probability per tile).
Selection: val mean IoU on positive tiles, early stopping; decision threshold chosen on val.

Baselines, on the same test tiles:
  location prior   mean train mask for the tile's position, threshold chosen on val (the attacker never sees
                   position, so this baseline is deliberately strict)
  pixel statistics `--encoder pixelstats`: the same model given only pixel statistics

Outputs
  results/phase6/<encoder>/<variant>/metrics.json, pertile.npz (rows, y, detection_score, iou_positive,
                   prior_iou_positive), examples.npz (16 fixed test tiles: true and predicted masks)
  results/phase6/localisation.csv   one row per run
  results/models/loc_<encoder>_<variant>.pt

Usage
  python scripts/localise.py --encoder resnet50_dino --variant full
  python scripts/localise.py --encoder pixelstats --variant full
"""
import argparse
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

THRESHOLDS = np.round(np.arange(0.2, 0.81, 0.05), 2)
SUMMARY = "results/phase6/localisation.csv"
FIELDS = ["encoder", "variant", "dims", "mean_iou_positive", "prior_mean_iou_positive", "detection_auc",
          "threshold", "epochs_run", "params_m", "seconds"]


class LocalisationDecoder(nn.Module):
    def __init__(self, d_in, d=256, n_mem=16, grid=8, layers=4, heads=8, p=0.1):
        super().__init__()
        self.n_mem, self.d, self.grid = n_mem, d, grid
        self.mem = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, n_mem * d))
        self.mem_pos = nn.Parameter(torch.randn(1, n_mem, d) * 0.02)
        self.queries = nn.Parameter(torch.randn(1, grid * grid, d) * 0.02)
        self.row = nn.Parameter(torch.randn(grid, 1, d) * 0.02)
        self.col = nn.Parameter(torch.randn(1, grid, d) * 0.02)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, p, batch_first=True, norm_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)

        def up(cin, cout):
            return nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1),
                                 nn.GroupNorm(8, cout), nn.GELU(), nn.Conv2d(cout, cout, 3, padding=1),
                                 nn.GroupNorm(8, cout), nn.GELU())
        self.head = nn.Sequential(up(d, 128), up(128, 64), up(64, 32), nn.Conv2d(32, 1, 1))

    def forward(self, z):
        b = z.shape[0]
        mem = self.mem(z).view(b, self.n_mem, self.d) + self.mem_pos
        pos = (self.row + self.col).reshape(1, self.grid * self.grid, self.d)
        q = self.decoder(self.queries.expand(b, -1, -1) + pos, mem)
        x = self.norm(q).transpose(1, 2).reshape(b, self.d, self.grid, self.grid)
        return self.head(x)


def dice_focal(logits, target, gamma=2.0, alpha=0.75):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pt = p * target + (1 - p) * (1 - target)
    at = alpha * target + (1 - alpha) * (1 - target)
    focal = (at * (1 - pt) ** gamma * ce).mean()
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return focal * 10 + dice.mean()


def tile_iou(prob, mask, thr):
    p, m = prob > thr, mask > 0.5
    inter, union = (p & m).sum((-2, -1)), (p | m).sum((-2, -1))
    return np.where(union > 0, inter / np.maximum(union, 1), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=None, help="subsample train tiles (testing only)")
    ap.add_argument("--out", default="results/phase6")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sp = C.load_splits(0)
    tr, va, te = (np.where(sp["split"] == k)[0] for k in range(3))
    if args.max_train:
        tr = np.sort(rng.choice(tr, args.max_train, replace=False))

    x = C.load_variant(args.encoder, args.variant, ["glosofarid"])["glosofarid"]
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    z = torch.as_tensor((x - mu) / sd, dtype=torch.float32, device=args.device)
    C.build_tile_cache()
    masks = np.load(C.TILES_MASK)  # (N, 64, 64) uint8, ~230 MB
    m_t = torch.as_tensor(masks, device=args.device)

    model = LocalisationDecoder(x.shape[1]).to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps = args.epochs * math.ceil(len(tr) / args.batch)
    warm = math.ceil(len(tr) / args.batch)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))

    def predict(rows):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 1024):
                out.append(torch.sigmoid(model(z[rows[i:i + 1024]]))[:, 0].half().cpu().numpy())
        model.train()
        return np.concatenate(out)

    va_pos = va[sp["y"][va] == 1]
    best, best_state, bad = -1.0, None, 0
    print(f"== localisation attack, {args.encoder}/{args.variant} ({x.shape[1]} dims, {n_params:.1f}M parameters)",
          flush=True)
    for ep in range(args.epochs):
        for b in np.array_split(rng.permutation(tr), max(1, len(tr) // args.batch)):
            opt.zero_grad()
            dice_focal(model(z[b]), m_t[b][:, None].float()).backward()
            opt.step()
            sched.step()
        val = float(tile_iou(predict(va_pos).astype(np.float32), masks[va_pos], 0.5).mean())
        print(f"  epoch {ep + 1:>2}: val IoU(positive) {val:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if val > best + 1e-4:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)

    pv = predict(va_pos).astype(np.float32)
    thr = float(max(THRESHOLDS, key=lambda t: tile_iou(pv, masks[va_pos], t).mean()))
    prior, _ = C.position_prior(sp)
    pthr = float(max(THRESHOLDS, key=lambda t: tile_iou(prior[sp["tile_idx"][va_pos]], masks[va_pos], t).mean()))

    pt = predict(te)
    y = sp["y"][te]
    pos = y == 1
    det = pt.reshape(len(te), -1).max(1).astype(np.float32)
    iou = tile_iou(pt[pos].astype(np.float32), masks[te[pos]], thr).astype(np.float32)
    piou = tile_iou(prior[sp["tile_idx"][te[pos]]], masks[te[pos]], pthr).astype(np.float32)
    summary = {"mean_iou_positive": float(iou.mean()), "prior_mean_iou_positive": float(piou.mean()),
               "detection_auc": float(roc_auc_score(y, det)), "threshold": thr, "prior_threshold": pthr}

    d = os.path.join(args.out, args.encoder, args.variant)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "pertile.npz"), rows=te, y=y.astype(np.int8), detection_score=det,
                        iou_positive=iou, prior_iou_positive=piou)
    ex_rng = np.random.default_rng(123)
    ex = np.sort(np.concatenate([ex_rng.choice(te[pos], 8, replace=False), ex_rng.choice(te[~pos], 8, replace=False)]))
    np.savez_compressed(os.path.join(d, "examples.npz"), rows=ex, true=masks[ex], pred=predict(ex))
    info = {"epochs_run": ep + 1, "best_val_iou": best, "params_m": round(n_params, 2), "n_train": len(tr),
            "seconds": round(time.time() - t0)}
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump({"encoder": args.encoder, "variant": args.variant, "dims": x.shape[1], "summary": summary,
                   "info": info}, fh, indent=2)
    os.makedirs("results/models", exist_ok=True)
    torch.save(best_state, f"results/models/loc_{args.encoder}_{args.variant}.pt")
    summary_path = os.path.join(args.out, "localisation.csv")
    new = not os.path.exists(summary_path)
    with open(summary_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"encoder": args.encoder, "variant": args.variant, "dims": x.shape[1],
                    "mean_iou_positive": round(summary["mean_iou_positive"], 4),
                    "prior_mean_iou_positive": round(summary["prior_mean_iou_positive"], 4),
                    "detection_auc": round(summary["detection_auc"], 4), "threshold": thr,
                    "epochs_run": ep + 1, "params_m": round(n_params, 2), "seconds": info["seconds"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
