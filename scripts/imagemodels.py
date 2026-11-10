"""Image models and training helpers for Phase 4b (upper bound) and Phase 5 (segmenter, inverters).

  resnet18_classifier  ResNet-18 from scratch, 13-band 64x64 input (image-level upper bound)
  UNet                 4-level U-Net with skip connections (panel segmenter)
  Generator            embedding -> 64x64x13 decoder, residual blocks, self-attention at 8x8 and 16x16
  PatchDiscriminator   PatchGAN with spectral norm
  PerceptualNet        frozen ResNet50 MoCo (SSL4EO, not an audited encoder) feature distance

All images inside the models are per-band standardised reflectance: (x / 10000 - mean) / std, with
mean/std from GloSoFarID train tiles (common.band_stats).
"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

SN = nn.utils.spectral_norm


# ---------------------------------------------------------------- data helpers

class TileBatches:
    """Batches of standardised tiles (and masks) from the tile cache, with dihedral augmentation."""

    def __init__(self, rows, device, with_mask=False):
        self.x = np.load(C.TILES_X, mmap_mode="r")
        self.m = np.load(C.TILES_MASK, mmap_mode="r") if with_mask else None
        mean, std = C.band_stats()
        self.mean = torch.tensor(mean, device=device)[None, :, None, None]
        self.std = torch.tensor(std, device=device)[None, :, None, None]
        self.rows = np.asarray(rows)
        self.device = device

    def load(self, rows):
        order = np.argsort(rows)  # sorted reads are much faster on a memmap
        x = np.empty((len(rows), 13, C.TILE, C.TILE), np.uint16)
        x[order] = self.x[rows[order]]
        x = (torch.from_numpy(x.astype(np.float32)).to(self.device) / 10000.0 - self.mean) / self.std
        if self.m is None:
            return x, None
        m = np.empty((len(rows), C.TILE, C.TILE), np.uint8)
        m[order] = self.m[rows[order]]
        return x, torch.from_numpy(m.astype(np.float32)).to(self.device)[:, None]

    def epoch(self, batch, rng, augment=True):
        perm = rng.permutation(self.rows)
        for i in range(0, len(perm), batch):
            rows = perm[i:i + batch]
            x, m = self.load(rows)
            if augment:
                k = int(rng.integers(8))
                x = dihedral(x, k)
                m = dihedral(m, k) if m is not None else None
            yield rows, x, m

    def ordered(self, batch):
        for i in range(0, len(self.rows), batch):
            rows = self.rows[i:i + batch]
            x, m = self.load(rows)
            yield rows, x, m

    def to_reflectance(self, z):
        """Standardised -> reflectance (/10,000), clipped to [0, 1]."""
        return (z * self.std + self.mean).clamp(0, 1)


def dihedral(t, k):
    """One of the 8 flips/90-degree rotations, applied to the whole batch (H, W are the last two dims)."""
    if t is None:
        return None
    if k >= 4:
        t = t.flip(-1)
    return torch.rot90(t, k % 4, dims=(-2, -1))


# ---------------------------------------------------------------- ResNet-18 upper bound

def resnet18_classifier():
    import timm
    return timm.create_model("resnet18", pretrained=False, in_chans=13, num_classes=1)


# ---------------------------------------------------------------- U-Net segmenter

def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, cin=13, widths=(32, 64, 128, 256), bottleneck=512):
        super().__init__()
        self.down = nn.ModuleList()
        c = cin
        for w in widths:
            self.down.append(conv_block(c, w))
            c = w
        self.mid = conv_block(c, bottleneck)
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        c = bottleneck
        for w in reversed(widths):
            self.up.append(nn.ConvTranspose2d(c, w, 2, stride=2))
            self.dec.append(conv_block(2 * w, w))
            c = w
        self.head = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        skips = []
        for d in self.down:
            x = d(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.mid(x)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            x = dec(torch.cat([up(x), s], 1))
        return self.head(x)


def dice_bce(logits, target, pos_weight=None):
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


# ---------------------------------------------------------------- GAN inverter

class ResBlock(nn.Module):
    def __init__(self, cin, cout, up=False):
        super().__init__()
        self.up = up
        self.n1, self.c1 = nn.GroupNorm(8, cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = nn.GroupNorm(8, cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = F.silu(self.n1(x))
        if self.up:
            h, x = F.interpolate(h, scale_factor=2, mode="nearest"), F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.c2(F.silu(self.n2(self.c1(h))))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        t = self.norm(x).flatten(2).transpose(1, 2)
        a, _ = self.attn(t, t, t, need_weights=False)
        return x + a.transpose(1, 2).reshape(b, c, h, w)


class Generator(nn.Module):
    """Embedding (D) -> 64x64x13. 4x4 seed, residual upsampling, self-attention at 8x8 and 16x16."""

    def __init__(self, d_in, base=512):
        super().__init__()
        self.fc = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, base * 16))
        self.base = base
        self.blocks = nn.ModuleList([
            ResBlock(base, base), ResBlock(base, 256, up=True), SelfAttention(256),      # 8x8
            ResBlock(256, 256), ResBlock(256, 128, up=True), SelfAttention(128),        # 16x16
            ResBlock(128, 128), ResBlock(128, 64, up=True),                              # 32x32
            ResBlock(64, 64), ResBlock(64, 32, up=True), ResBlock(32, 32),               # 64x64
        ])
        self.out = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 13, 3, padding=1))

    def forward(self, z):
        x = self.fc(z).view(-1, self.base, 4, 4)
        for b in self.blocks:
            x = b(x)
        return self.out(x)


class PatchDiscriminator(nn.Module):
    """PatchGAN: 64x64 -> 7x7 realism logits, spectral norm, LeakyReLU."""

    def __init__(self, cin=13, w=64):
        super().__init__()
        self.net = nn.Sequential(
            SN(nn.Conv2d(cin, w, 4, 2, 1)), nn.LeakyReLU(0.2, True),          # 32
            SN(nn.Conv2d(w, 2 * w, 4, 2, 1)), nn.LeakyReLU(0.2, True),        # 16
            SN(nn.Conv2d(2 * w, 4 * w, 4, 2, 1)), nn.LeakyReLU(0.2, True),    # 8
            SN(nn.Conv2d(4 * w, 8 * w, 4, 1, 1)), nn.LeakyReLU(0.2, True),    # 7
            SN(nn.Conv2d(8 * w, 1, 3, 1, 1)))

    def forward(self, x):
        return self.net(x)


class PerceptualNet(nn.Module):
    """Frozen ResNet50 MoCo (SSL4EO, 13 bands). Distance = mean L1 between unit-normalised features
    of stages 1-3. Not one of the audited encoders, so the loss cannot favour one of them."""

    def __init__(self, device):
        super().__init__()
        import torchgeo.models as tgm
        # full model with its pretrained weights (features_only would drop layer4 and break weight loading);
        # features are read from stages 1-3 directly
        self.net = tgm.resnet50(weights=tgm.ResNet50_Weights.SENTINEL2_ALL_MOCO).eval().to(device)
        for p in self.net.parameters():
            p.requires_grad_(False)

    def features(self, x):
        n = self.net
        x = n.maxpool(n.act1(n.bn1(n.conv1(x))))
        f1 = n.layer1(x)
        f2 = n.layer2(f1)
        return [f1, f2, n.layer3(f2)]

    def forward(self, a, b):
        """a, b: reflectance (/10,000) images, (B, 13, 64, 64)."""
        loss = 0.0
        for fa, fb in zip(self.features(a), self.features(b)):
            fa = fa / (fa.norm(dim=1, keepdim=True) + 1e-6)
            fb = fb / (fb.norm(dim=1, keepdim=True) + 1e-6)
            loss = loss + (fa - fb).abs().mean()
        return loss
