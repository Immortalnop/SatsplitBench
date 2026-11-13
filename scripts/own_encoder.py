"""Phase 7: own encoder (ViT-Small/8, masked autoencoder) - baseline and defended, from ONE training script.

Encoder: ViT-Small (384 dims, depth 12, 6 heads), 8-pixel patches on 64x64x13 tiles (64 tokens),
fixed 2D sin-cos position embeddings (so any multiple-of-8 input works, e.g. BigEarthNet's 112x112),
embedding = mean of the final patch tokens. 384 dims, the same as the audited ViT-S/16 encoders.

Every training step, for BOTH baseline and defended:
  1. MAE: 75% of patches masked; a light decoder (4 blocks, 256 dims) reconstructs the masked patches
     (per-patch normalised targets). Batch = 128 EuroSAT train tiles + 128 GloSoFarID train tiles.
  2. Utility head: land-cover classifier on the EuroSAT half (cross-entropy), weight --utility-weight.
  3. Adversary: MLP predicting "solar farm present" from the GloSoFarID half, behind a gradient-reversal
     layer with strength --adv-weight (lambda).
     baseline : lambda = 0  -> the adversary trains (and monitors leakage) but never changes the encoder
     defended : lambda > 0  -> the encoder is pushed to make solar farms unpredictable
     lambda is ramped linearly from 0 to its final value over the first --adv-ramp-epochs (default 4) epochs.
The two runs are identical except for lambda, so any change in leakage is caused by the defence.
Only train-split tiles are used (GloSoFarID split seed 0, EuroSAT official train). Augmentation:
the 8 flips / 90-degree rotations only (no resizing, no colour changes).

The defended run's lambda (0.5) and the decision not to add a spatial adversary are recorded, with the
reasoning, in results/preregistration.md ("Phase 7 defence settings") before either encoder is trained.

After training, embeddings are exported in the same layout as the audited encoders:
  data/embeddings/<name>/eurosat_{train,val,test}.npy, glosofarid.npy, [ben_{split}.npy with --ben]
so scripts/compress.py, quantise.py, attack4b.py, localise.py and invert.py work on it unchanged.

Usage
  python scripts/own_encoder.py --name own_vit_baseline --adv-weight 0
  python scripts/own_encoder.py --name own_vit_defended --adv-weight <lambda>     # after Phase 4b
  python scripts/own_encoder.py --name own_vit_baseline --export-only --ben        # re-export embeddings
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
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

PATCH = 8


# ---------------------------------------------------------------- model

def sincos_2d(gh, gw, dim):
    """Fixed 2D sin-cos position embedding (MAE), (gh*gw, dim)."""
    def one(pos, d):
        omega = 1.0 / 10000 ** (torch.arange(d // 2, dtype=torch.float32) / (d / 2))
        out = pos.flatten()[:, None] * omega[None]
        return torch.cat([out.sin(), out.cos()], 1)
    yy, xx = torch.meshgrid(torch.arange(gh, dtype=torch.float32), torch.arange(gw, dtype=torch.float32),
                            indexing="ij")
    return torch.cat([one(yy, dim // 2), one(xx, dim // 2)], 1)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class MAEViT(nn.Module):
    def __init__(self, cin=13, dim=384, depth=12, heads=6, dec_dim=256, dec_depth=4, dec_heads=8, n_classes=10):
        super().__init__()
        from timm.models.vision_transformer import Block
        self.dim, self.dec_dim, self.cin = dim, dec_dim, cin
        self.patch = nn.Conv2d(cin, dim, PATCH, stride=PATCH)
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio=4.0, qkv_bias=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.dec_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.dec_blocks = nn.ModuleList([Block(dec_dim, dec_heads, mlp_ratio=4.0, qkv_bias=True)
                                         for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.dec_pred = nn.Linear(dec_dim, PATCH * PATCH * cin)
        self.utility = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))
        self.adversary = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, 256),
                                       nn.GELU(), nn.Linear(256, 1))
        nn.init.normal_(self.mask_token, std=0.02)

    def tokens(self, x):
        t = self.patch(x)
        gh, gw = t.shape[-2:]
        return t.flatten(2).transpose(1, 2) + sincos_2d(gh, gw, self.dim).to(x.device)[None]

    def encode(self, x):
        """Embedding used downstream: mean of final patch tokens, all patches visible."""
        t = self.tokens(x)
        for b in self.blocks:
            t = b(t)
        return self.norm(t).mean(1)

    def mae_loss(self, x, ratio=0.75):
        b = x.shape[0]
        t = self.tokens(x)
        n = t.shape[1]
        keep = int(n * (1 - ratio))
        order = torch.rand(b, n, device=x.device).argsort(1)
        restore = order.argsort(1)
        vis = torch.gather(t, 1, order[:, :keep, None].expand(-1, -1, self.dim))
        for blk in self.blocks:
            vis = blk(vis)
        vis = self.dec_embed(self.norm(vis))
        full = torch.cat([vis, self.mask_token.expand(b, n - keep, -1)], 1)
        full = torch.gather(full, 1, restore[:, :, None].expand(-1, -1, self.dec_dim))
        g = int(math.isqrt(n))
        full = full + sincos_2d(g, g, self.dec_dim).to(x.device)[None]
        for blk in self.dec_blocks:
            full = blk(full)
        pred = self.dec_pred(self.dec_norm(full))
        target = F.unfold(x, PATCH, stride=PATCH).transpose(1, 2)            # (B, n, cin*P*P)
        target = (target - target.mean(-1, keepdim=True)) / (target.var(-1, keepdim=True) + 1e-6).sqrt()
        # unfold orders channels-major per patch; prediction is learned to that layout
        masked = torch.ones(b, n, device=x.device)
        masked[:, :keep] = 0
        masked = torch.gather(masked, 1, restore)
        return (((pred - target) ** 2).mean(-1) * masked).sum() / masked.sum()


# ---------------------------------------------------------------- data

class Tiles:
    """Standardised 13-band tiles from EuroSAT (data/eurosat/npy) or GloSoFarID (tile cache)."""

    def __init__(self, device):
        C.build_tile_cache()
        mean, std = C.band_stats()
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]
        self.device = device
        self.src = {"glosofarid": np.load(C.TILES_X, mmap_mode="r")}
        for s in ("train", "val", "test"):
            self.src[f"eurosat_{s}"] = np.load(f"data/eurosat/npy/{s}_x.npy", mmap_mode="r")

    def load(self, src, rows):
        rows = np.asarray(rows)
        order = np.argsort(rows)
        a = np.empty((len(rows),) + self.src[src].shape[1:], np.uint16)
        a[order] = self.src[src][rows[order]]
        return self.norm(torch.from_numpy(a.astype(np.float32)).to(self.device))

    def norm(self, x):
        return (x / 10000.0 - self.mean) / self.std


# ---------------------------------------------------------------- export

@torch.no_grad()
def export(model, tiles, name, device, ben=False, batch=512):
    model.eval()
    out = os.path.join(C.EMB, name)
    os.makedirs(out, exist_ok=True)
    jobs = [(f"eurosat_{s}", f"eurosat_{s}") for s in ("train", "val", "test")] + [("glosofarid", "glosofarid")]
    for src, fname in jobs:
        n = len(tiles.src[src])
        emb = np.concatenate([model.encode(tiles.load(src, np.arange(i, min(i + batch, n)))).float().cpu().numpy()
                              for i in range(0, n, batch)])
        np.save(os.path.join(out, f"{fname}.npy"), emb)
        print(f"  exported {fname}: {emb.shape}", flush=True)
    if ben:
        from bigearthnet import BenDataset, stack_batch
        for s in ("train", "val", "test"):
            ids = np.load(f"data/bigearthnet/npy/{s}_patch_ids.npy", allow_pickle=True)
            dl = torch.utils.data.DataLoader(BenDataset(ids), batch_size=128, num_workers=6, collate_fn=stack_batch)
            emb = np.concatenate([model.encode(tiles.norm(torch.from_numpy(b.astype(np.float32)).to(device)))
                                  .float().cpu().numpy() for b in dl])
            np.save(os.path.join(out, f"ben_{s}.npy"), emb)
            print(f"  exported ben_{s}: {emb.shape}", flush=True)
    model.train()


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="e.g. own_vit_baseline, own_vit_defended")
    ap.add_argument("--adv-weight", type=float, required=True, help="gradient-reversal lambda; 0 = baseline")
    ap.add_argument("--utility-weight", type=float, default=0.1)
    # measured ~1.3-2 s per step (303 steps/epoch) on an M-series Mac: 40 epochs ~5-7 h
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--adv-ramp-epochs", type=int, default=4,
                    help="lambda rises linearly from 0 to --adv-weight over this many epochs (standard DANN "
                         "schedule; the same schedule scaled by lambda = 0 is exactly 0 for the baseline)")
    ap.add_argument("--batch-half", type=int, default=128, help="EuroSAT and GloSoFarID tiles per step, each")
    ap.add_argument("--lr", type=float, default=3e-4, help="peak lr for a 256-tile step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps-per-epoch", type=int, default=None, help="testing only")
    ap.add_argument("--export-only", action="store_true")
    ap.add_argument("--ben", action="store_true", help="also export BigEarthNet embeddings")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tiles = Tiles(args.device)
    model = MAEViT().to(args.device)
    ckpt = f"results/models/{args.name}.pt"
    if args.export_only:
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
        export(model, tiles, args.name, args.device, args.ben)
        return

    sp = C.load_splits(0)
    g_tr = np.where(sp["split"] == 0)[0]
    g_y = torch.as_tensor(sp["y"].astype(np.float32), device=args.device)
    e_y_np = C.eurosat_labels()["train"]
    e_y = torch.as_tensor(e_y_np, device=args.device)
    e_tr = np.arange(len(e_y_np))
    steps_per_epoch = min(len(g_tr) // args.batch_half, args.max_steps_per_epoch or 10 ** 9)
    total = args.epochs * steps_per_epoch
    warm = args.warmup * steps_per_epoch
    ramp = max(1, args.adv_ramp_epochs * steps_per_epoch)

    enc_params = [p for n, p in model.named_parameters() if not n.startswith("adversary")]
    opt = torch.optim.AdamW(enc_params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    adv_opt = torch.optim.AdamW(model.adversary.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, warm)) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    pos = float(sp["y"][g_tr].mean())
    pos_w = torch.tensor((1 - pos) / pos, device=args.device)

    log, t0, step = [], time.time(), 0
    print(f"== {args.name}: adv-weight {args.adv_weight}, utility-weight {args.utility_weight}, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters, {steps_per_epoch} steps/epoch",
          flush=True)
    for ep in range(args.epochs):
        sums = np.zeros(4)
        gp = rng.permutation(g_tr)
        for i in range(steps_per_epoch):
            gr = gp[i * args.batch_half:(i + 1) * args.batch_half]
            er = rng.choice(e_tr, args.batch_half, replace=False)
            k = int(rng.integers(8))
            xg, xe = M.dihedral(tiles.load("glosofarid", gr), k), M.dihedral(tiles.load("eurosat_train", er), k)
            mae = model.mae_loss(torch.cat([xe, xg]))
            util = F.cross_entropy(model.utility(model.encode(xe)), e_y[er])
            zg = model.encode(xg)
            lam = args.adv_weight * min(1.0, step / ramp)
            adv_logit = model.adversary(GradReverse.apply(zg, lam))[:, 0]
            adv = F.binary_cross_entropy_with_logits(adv_logit, g_y[gr], pos_weight=pos_w)
            opt.zero_grad()
            adv_opt.zero_grad()
            (mae + args.utility_weight * util + adv).backward()  # GRL flips (and scales) adv's grad into the encoder
            opt.step()
            adv_opt.step()
            sched.step()
            step += 1
            with torch.no_grad():
                acc = ((adv_logit > 0).float() == g_y[gr]).float().mean().item()
            sums += [mae.item(), util.item(), adv.item(), acc]
        m = sums / steps_per_epoch
        log.append({"epoch": ep + 1, "mae": m[0], "utility_ce": m[1], "adversary_bce": m[2], "adversary_acc": m[3]})
        print(f"  epoch {ep + 1:>3}: MAE {m[0]:.4f}  utility CE {m[1]:.4f}  adversary BCE {m[2]:.4f} "
              f"acc {m[3]:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    os.makedirs("results/phase7", exist_ok=True)
    with open(f"results/phase7/{args.name}_train.json", "w") as fh:
        json.dump({"name": args.name, "adv_weight": args.adv_weight, "utility_weight": args.utility_weight,
                   "adv_ramp_epochs": args.adv_ramp_epochs, "epochs": args.epochs, "lr": args.lr, "batch": 2 * args.batch_half, "seed": args.seed,
                   "seconds": round(time.time() - t0), "log": log}, fh, indent=2)
    export(model, tiles, args.name, args.device, args.ben)


if __name__ == "__main__":
    main()
