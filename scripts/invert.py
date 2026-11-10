"""Phase 5: inversion attacks, embedding -> 64x64x13 image, and their evaluation.

Methods
  mean  baseline: every tile reconstructed as the mean training tile
  knn   weak inverter: mean of the k training tiles nearest in embedding space (cosine; k chosen on val)
  gan   strong inverter: Generator (residual blocks, self-attention at 8x8 and 16x16) + PatchGAN
        discriminator; loss = 10 * L1 + 1 * perceptual (frozen ResNet50 MoCo, not audited)
        + 0.1 * hinge adversarial (after a warm-up without it); selection by val L1 + perceptual
Inputs: any embedding variant (data/compressed or data/quantised), or `--encoder pixelstats` for the
control: the same inverter given only pixel statistics. Learned leakage = embedding minus that control.

Trained on GloSoFarID train tiles, selected on val, evaluated on test tiles only (never seen in training).

Evaluation, identical for every method, on all test tiles:
  pixel:   L1 (standardised bands), SSIM (13 bands, reflectance), LPIPS (RGB = B4,B3,B2), spectral angle (deg)
  spatial: the frozen U-Net panel segmenter (scripts/segmenter.py) run on each reconstruction
           -> detection AUC (max probability per tile, all test tiles)
           -> mean mask IoU on positive test tiles, at the segmenter's val threshold
  reference rows in the same file: segmenter on the real test images (ceiling) and the location prior

Outputs
  results/phase5/<method>/<encoder>/<variant>/metrics.json
  results/phase5/<method>/<encoder>/<variant>/pertile.npz   per-test-tile scores, for paired tests
  results/phase5/<method>/<encoder>/<variant>/examples.npz  16 fixed test tiles, real and reconstructed
  results/phase5/inversion.csv                              one summary row per run
  results/models/gan_<encoder>_<variant>.pt

Usage
  python scripts/invert.py --method mean
  python scripts/invert.py --method knn --encoder resnet50_dino --variant full
  python scripts/invert.py --method gan --encoder resnet50_dino --variant full
  python scripts/invert.py --method gan --encoder pixelstats --variant full
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
import imagemodels as M  # noqa: E402
import segmenter as S  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

RGB = [3, 2, 1]  # B4, B3, B2 in encoder band order
SUMMARY = "results/phase5/inversion.csv"
FIELDS = ["method", "encoder", "variant", "dims", "l1", "ssim", "lpips", "sam_deg", "detection_auc",
          "mean_iou_positive", "real_detection_auc", "real_mean_iou_positive", "prior_mean_iou_positive",
          "params", "seconds"]


# ---------------------------------------------------------------- features

def features(enc, var, sp):
    x = C.load_variant(enc, var, ["glosofarid"])["glosofarid"]
    tr = sp["split"] == 0
    mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


# ---------------------------------------------------------------- metrics

class Evaluator:
    def __init__(self, device):
        from torchmetrics.functional.image import structural_similarity_index_measure as ssim
        from torchmetrics.image.lpip import _NoTrainLpips

        self.ssim = ssim
        self.lpips = _NoTrainLpips(net="alex").to(device).eval()
        self.seg, self.thr = S.load(device)
        self.device = device

    @torch.no_grad()
    def batch(self, rec, real, tb):
        """rec, real: standardised (B, 13, 64, 64). Returns per-tile metric arrays and segmenter probs."""
        ra, rb = tb.to_reflectance(rec), tb.to_reflectance(real)
        l1 = (rec - real).abs().mean((1, 2, 3))
        ss = self.ssim(ra, rb, data_range=1.0, reduction="none")
        # LPIPS on RGB, scaled like the GloSoFarID preview (reflectance / 0.3), mapped to [-1, 1]
        rgb = lambda t: (t[:, RGB] / 0.3).clamp(0, 1) * 2 - 1
        lp = self.lpips(rgb(ra), rgb(rb)).flatten()
        cos = (ra * rb).sum(1) / (ra.norm(dim=1) * rb.norm(dim=1) + 1e-8)
        sam = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean((1, 2))
        prob = torch.sigmoid(self.seg(rec))[:, 0]
        return [t.float().cpu().numpy() for t in (l1, ss, lp, sam)], prob.half().cpu().numpy()


def evaluate(recon_batches, te_rows, sp, tb, ev):
    """recon_batches yields (rows, standardised reconstructions) covering te_rows in order."""
    out = {k: [] for k in ("l1", "ssim", "lpips", "sam")}
    probs, masks = [], []
    m_all = np.load(C.TILES_MASK, mmap_mode="r")
    for rows, rec in recon_batches:
        real, m = tb.load(rows)
        vals, prob = ev.batch(rec, real, tb)
        for k, v in zip(out, vals):
            out[k].append(v)
        probs.append(prob)
        masks.append(np.asarray(m_all[rows]))
    per = {k: np.concatenate(v) for k, v in out.items()}
    prob, mask = np.concatenate(probs), np.concatenate(masks)
    y = sp["y"][te_rows]
    det = prob.reshape(len(prob), -1).max(1).astype(np.float32)
    pos = y == 1
    iou = S.tile_iou(prob[pos].astype(np.float32), mask[pos], ev.thr)
    per.update({"detection_score": det, "iou_positive": iou.astype(np.float32), "rows": te_rows,
                "y": y.astype(np.int8)})
    summary = {"l1": float(per["l1"].mean()), "ssim": float(per["ssim"].mean()),
               "lpips": float(per["lpips"].mean()), "sam_deg": float(per["sam"].mean()),
               "detection_auc": float(roc_auc_score(y, det)), "mean_iou_positive": float(iou.mean())}
    return summary, per


# ---------------------------------------------------------------- inverters

def mean_inverter(sp, tb_tr, device):
    s = None
    n = 0
    for _, x, _ in tb_tr.ordered(2048):
        s = x.sum(0) if s is None else s + x.sum(0)
        n += len(x)
    mean_tile = (s / n)[None]

    def run(rows, batch=256):
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            yield r, mean_tile.expand(len(r), -1, -1, -1)
    return run, {}


def knn_inverter(feat, sp, tb_tr, device, ks=(1, 5, 10, 20)):
    tr = np.where(sp["split"] == 0)[0]
    bank = F.normalize(torch.as_tensor(feat[tr], device=device), dim=1)

    def neighbours(rows, k):
        q = F.normalize(torch.as_tensor(feat[rows], device=device), dim=1)
        return tr[(q @ bank.T).topk(k, dim=1).indices.cpu().numpy()]

    def reconstruct(rows, k):
        nb = neighbours(rows, k)
        x, _ = tb_tr.load(nb.reshape(-1))
        return x.view(len(rows), k, 13, C.TILE, C.TILE).mean(1)

    va = np.where(sp["split"] == 1)[0]
    va = va[np.random.default_rng(0).choice(len(va), min(1024, len(va)), replace=False)]
    real, _ = tb_tr.load(va)
    val_l1 = {k: float(sum((reconstruct(va[i:i + 128], k) - real[i:i + 128]).abs().sum().item()
                           for i in range(0, len(va), 128)) / real.numel()) for k in ks}
    k = min(val_l1, key=val_l1.get)
    print(f"  kNN val L1 by k: {val_l1} -> k = {k}", flush=True)

    def run(rows, batch=128):
        for i in range(0, len(rows), batch):
            r = rows[i:i + batch]
            yield r, reconstruct(r, k)
    return run, {"k": k, "val_l1_by_k": val_l1}


def gan_inverter(feat, sp, tb_tr, device, args, tag):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = (np.where(sp["split"] == k)[0] for k in (0, 1))
    va = va[np.random.default_rng(0).choice(len(va), min(1024, len(va)), replace=False)]
    G = M.Generator(feat.shape[1], base=args.base).to(device)
    D = M.PatchDiscriminator().to(device)
    P = M.PerceptualNet(device)
    og = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    fz = torch.as_tensor(feat, device=device)

    def val_loss():
        G.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(va), 256):
                r = va[i:i + 256]
                real, _ = tb_tr.load(r)
                fake = G(fz[r])
                tot += (10 * (fake - real).abs().mean() + P(tb_tr.to_reflectance(fake),
                                                            tb_tr.to_reflectance(real))).item() * len(r)
        G.train()
        return tot / len(va)

    best, best_state, bad, t0, hist = math.inf, None, 0, time.time(), []
    for ep in range(args.epochs):
        adv = ep >= args.warmup
        perm = rng.permutation(tr)
        for i in range(0, len(perm), args.batch):
            r = perm[i:i + args.batch]
            real, _ = tb_tr.load(r)
            fake = G(fz[r])
            if adv:
                od.zero_grad()
                ld = F.relu(1 - D(real)).mean() + F.relu(1 + D(fake.detach())).mean()
                ld.backward()
                od.step()
            og.zero_grad()
            lg = 10 * (fake - real).abs().mean() + P(tb_tr.to_reflectance(fake), tb_tr.to_reflectance(real))
            if adv:
                lg = lg - 0.1 * D(fake).mean()
            lg.backward()
            og.step()
        v = val_loss()
        hist.append(v)
        print(f"  epoch {ep + 1:>2}{' (adv)' if adv else ''}: val L1x10+perceptual {v:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in G.state_dict().items()}
        elif adv:  # early stopping only once the adversarial phase has started
            bad += 1
            if bad >= args.patience:
                break
    G.load_state_dict(best_state)
    G.eval()
    os.makedirs("results/models", exist_ok=True)
    torch.save(best_state, f"results/models/gan_{tag}.pt")

    def run(rows, batch=256):
        with torch.no_grad():
            for i in range(0, len(rows), batch):
                r = rows[i:i + batch]
                yield r, G(fz[r])
    return run, {"epochs_run": ep + 1, "best_val": best, "val_history": hist, "base": args.base,
                 "loss": "10*L1 + perceptual(ResNet50 MoCo) + 0.1*hinge after warm-up"}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["mean", "knn", "gan"])
    ap.add_argument("--encoder", default="resnet50_dino")
    ap.add_argument("--variant", default="full")
    # measured on an M-series Mac: ~3.7 min per warm-up epoch, ~5.1 min per adversarial epoch (2048-dim input)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3, help="epochs before the adversarial loss starts")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--base", type=int, default=512, help="generator width at 4x4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-test", type=int, default=None, help="evaluate on the first N test tiles (testing)")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    if not os.path.exists(S.MODEL):
        raise SystemExit("Train the panel segmenter first: python scripts/segmenter.py")
    t0 = time.time()
    sp = C.load_splits(0)
    tb_tr = M.TileBatches(np.where(sp["split"] == 0)[0], args.device)
    te = np.where(sp["split"] == 2)[0][:args.max_test]
    enc, var = ("baseline", "mean") if args.method == "mean" else (args.encoder, args.variant)
    tag = f"{enc}_{var}"
    print(f"== {args.method} inverter, {tag}", flush=True)

    if args.method == "mean":
        run, info = mean_inverter(sp, tb_tr, args.device)
        dims = 0
    else:
        feat = features(args.encoder, args.variant, sp)
        dims = feat.shape[1]
        run, info = (knn_inverter(feat, sp, tb_tr, args.device) if args.method == "knn"
                     else gan_inverter(feat, sp, tb_tr, args.device, args, tag))

    ev = Evaluator(args.device)
    summary, per = evaluate(run(te), te, sp, tb_tr, ev)
    ref = json.load(open(S.INFO))
    summary.update({"real_detection_auc": ref["test_real"]["detection_auc"],
                    "real_mean_iou_positive": ref["test_real"]["mean_iou_positive"],
                    "prior_mean_iou_positive": ref["test_location_prior"]["mean_iou_positive"]})

    # 16 fixed test tiles for figures: 8 positive, 8 negative
    rng = np.random.default_rng(123)
    ex = np.sort(np.concatenate([rng.choice(te[sp["y"][te] == 1], 8, replace=False),
                                 rng.choice(te[sp["y"][te] == 0], 8, replace=False)]))
    rec = torch.cat([r for _, r in run(ex)])
    real, _ = tb_tr.load(ex)

    d = os.path.join("results/phase5", args.method, enc, var)
    os.makedirs(d, exist_ok=True)
    np.savez_compressed(os.path.join(d, "pertile.npz"), **per)
    np.savez_compressed(os.path.join(d, "examples.npz"), rows=ex,
                        real=tb_tr.to_reflectance(real).half().cpu().numpy(),
                        recon=tb_tr.to_reflectance(rec).half().cpu().numpy())
    info.update({"n_test": len(te), "seconds": round(time.time() - t0)})
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump({"method": args.method, "encoder": enc, "variant": var, "dims": dims,
                   "summary": summary, "info": info}, fh, indent=2)
    new = not os.path.exists(SUMMARY)
    with open(SUMMARY, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"method": args.method, "encoder": enc, "variant": var, "dims": dims,
                    **{k: round(v, 4) for k, v in summary.items()},
                    "params": json.dumps({k: v for k, v in info.items() if k not in ("val_history", "val_l1_by_k")}),
                    "seconds": info["seconds"]})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
