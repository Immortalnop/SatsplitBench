"""Phase 8: split points inside the encoders, shared by compute8a.py and splits8b.py.

A split point cuts an encoder into a HEAD (runs on satellite A) and a TAIL (satellite B or ground).
Each encoder is a list of stages; split k means "after stage k".

  ResNet (timm, from torchgeo):  stem + layer1 | layer2 | layer3 | layer4      -> splits layer1..layer4
  ViT-S/16 (timm, from torchgeo): embed + blocks 1-3 | 4-6 | 7-9 | 10-12      -> splits block3..block12
  Own ViT-S/8 (Phase 7, compute only): tokens + blocks 1-3 | ... | 10-12      -> splits block3..block12

The final split (layer4 / block12) is the encoder output the earlier phases used.

Readouts of a split's output (the payload forms of Phase 8b):
  gap   pooled vector. ResNet: global average pool. timm ViT: final LayerNorm, CLS token.
        Own ViT: final LayerNorm, mean over tokens. At the final split this equals the encoder output.
  map4  spatial map. ResNet: average-pooled to 4x4 (layer1-2), native 4x4 (layer3), native 2x2 (layer4).
        timm ViT: final LayerNorm, the patch tokens (no CLS). Flattened channel-major.
"""
import torch
import torch.nn.functional as F

ARCH = {  # encoder -> architecture (8a measures each architecture once)
    "resnet18_moco": "resnet18", "resnet18_random": "resnet18", "resnet50_dino": "resnet50",
    "vits16_moco": "vits16", "vits16_dino": "vits16", "own_vit": "own_vits8",
}
SPLITS = {
    "resnet18": ["layer1", "layer2", "layer3", "layer4"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    "vits16": ["block3", "block6", "block9", "block12"],
    "own_vits8": ["block3", "block6", "block9", "block12"],
}


def stages(model, arch):
    """List of (split_name, callable), one per stage, in order. Applying them in sequence is the encoder."""
    if arch.startswith("resnet"):
        def s1(x):
            x = model.maxpool(model.act1(model.bn1(model.conv1(x))))
            return model.layer1(x)
        return [("layer1", s1), ("layer2", model.layer2), ("layer3", model.layer3), ("layer4", model.layer4)]
    if arch == "vits16":
        def embed(x):
            x = model.patch_embed(x)
            x = model._pos_embed(x)
            x = model.patch_drop(x)
            return model.norm_pre(x)
        return _vit_stages(model.blocks, embed)
    if arch == "own_vits8":
        return _vit_stages(model.blocks, model.tokens)
    raise ValueError(arch)


def _vit_stages(blocks, embed):
    def run(lo, hi, first):
        def f(x):
            if first:
                x = embed(x)
            for b in blocks[lo:hi]:
                x = b(x)
            return x
        return f
    return [(f"block{hi}", run(lo, hi, lo == 0)) for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 12))]


def finish(model, arch, h):
    """Tail after the last stage: the rest of the encoder's forward (for the head + tail check)."""
    if arch.startswith("resnet"):
        return model.forward_head(h)
    if arch == "vits16":
        return model.forward_head(model.norm(h))
    return model.norm(h).mean(1)


def gap(model, arch, h):
    if arch.startswith("resnet"):
        return h.mean((2, 3))
    if arch == "vits16":
        return model.norm(h)[:, 0]
    return model.norm(h).mean(1)


def map4(model, arch, h):
    if arch.startswith("resnet"):
        if h.shape[-1] > 4:
            h = F.adaptive_avg_pool2d(h, 4)
        return h.flatten(1)
    if arch == "vits16":
        t = model.norm(h)[:, model.num_prefix_tokens:]       # (B, 16, 384) at 64x64
        return t.transpose(1, 2).flatten(1)                  # channel-major, like the ResNet maps
    return model.norm(h).transpose(1, 2).flatten(1)


def raw_map(arch, h):
    """The tensor that would actually be transmitted at this split (per tile)."""
    return h


def build(name, device, seed=0):
    """Encoder by name: the Phase 2 encoders via embed.build_encoder, or the Phase 7 architecture (random init)."""
    if name == "own_vit":
        from own_encoder import MAEViT
        return MAEViT().eval().to(device)
    from embed import build_encoder
    return build_encoder(name, device, seed)


@torch.inference_mode()
def check_stages(model, arch, x):
    """Max abs difference between running the stages + finish and the encoder's own forward."""
    h = x
    for _, f in stages(model, arch):
        h = f(h)
    ref = model.encode(x) if arch == "own_vits8" else model(x)
    return float((finish(model, arch, h) - ref).abs().max()), float(ref.abs().max())
