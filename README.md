# EO Embedding Leakage Study

How much object-level information leaks from compact Earth observation (EO) embeddings, and how does that leakage trade off against downstream utility as the embeddings are compressed?

Full research plan: [EO Embedding Leakage Study — Research Plan](https://claude.ai/artifact/QXSerKGJrwe2oUhe8QX23P)

## Research questions

- **RQ1, attribute leakage:** can a linear probe on an embedding detect objects it was not used for (solar farms), above chance?
- **RQ2, trade-off:** as embeddings are compressed (PCA dims, int8), does leakage fall faster or slower than utility?
- **RQ3, metric mismatch:** do pixel metrics (SSIM, LPIPS) of an inverted image rank encoders the same way as object-level leakage?

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Environment setup | Done |
| 1 | Data acquisition | Done |
| 2 | Embeddings | Done |
| 3 | Compression | Done |
| 4 | Probes and controls (linear + MLP) | Done |
| 4b | Learned-leakage follow-ups (pre-registered) | Built, testing |
| 5 | Inversion: weak and strong inverters | Built, testing |
| 6 | Localisation attack | Planned |
| 7 | Own encoder and defence | Planned |
| 8 | Analysis and paper | Planned |

---

## Done

### Phase 0: Environment setup

Python virtual environment in `eoenv/` (Mac, Apple Silicon, MPS available).

| Package | Version |
|---|---|
| torch | 2.15.0 |
| torchgeo | 0.10.1 |
| scikit-learn | 1.9.1 |
| tensorflow | 2.21.0 (only for reading GloSoFarID TFRecords) |

```bash
source eoenv/bin/activate
python -c "import torch, torchgeo; print(torch.backends.mps.is_available(), torchgeo.__version__)"
```

### Phase 1: Data acquisition

All data lives under `data/` (git-ignored).

| Dataset | Role | Location | Contents | Verified |
|---|---|---|---|---|
| EuroSAT (MS) | Utility task (land cover) | `data/eurosat/` | 27,000 patches, 64×64, 13 bands, 10 classes | 27,000 .tif files extracted |
| GloSoFarID, 2023 global shard | Object leakage (solar farms) | `data/glosofarid/` | 3,463 patches, 256×256, 13 bands, masks | md5 `6c3d50e8728b88e53472ed40d7be29ee` matches Zenodo |
| BigEarthNet v2, S2 (optional) | Utility and leakage on the same images | `data/bigearthnet/` | 549,488 patches, 120×120, 12 bands, 19 labels | 549,488 patches + 549,488 reference maps extracted |

**EuroSAT**

```bash
python -c "from torchgeo.datasets import EuroSAT; EuroSAT(root='data/eurosat', split='train', download=True)"
```

**BigEarthNet v2 (Sentinel-2 only)**

```bash
python -c "from torchgeo.datasets import BigEarthNetV2; BigEarthNetV2(root='data/bigearthnet', split='train', bands='s2', download=True)"
```

BigEarthNet is Level-2A (surface reflectance) and has no B10 cirrus band, so it has 12 bands. EuroSAT and GloSoFarID have all 13.

**GloSoFarID**

The data is on Zenodo ([record 10939100](https://zenodo.org/records/10939100)) as five gzipped TFRecord shards (about 10.3 GB). The [GitHub repo](https://github.com/yzyly1992/GloSoFarID) holds only a converter script, which is not needed here. Only the 2023 global shard is downloaded.

```bash
mkdir -p data/glosofarid
# -C - resumes a partial download; the loop retries if the connection drops
until curl -L -C - --retry 5 --retry-delay 5 -o data/glosofarid/solar_2023_global.tfrecord.gz \
  "https://zenodo.org/records/10939100/files/solar_2023_global.tfrecord.gz?download=1"; do sleep 5; done
md5 data/glosofarid/solar_2023_global.tfrecord.gz   # 6c3d50e8728b88e53472ed40d7be29ee
```

`scripts/load_glosofarid.py` reads the TFRecords straight into numpy, skipping the 24 GB .tif conversion:

```bash
python scripts/load_glosofarid.py data/glosofarid/solar_2023_global.tfrecord.gz --out data/glosofarid/npy
```

| File | Shape, dtype | Meaning |
|---|---|---|
| `solar_2023_global_x.npy` | (3463, 13, 256, 256), uint16 | Bands B1–B12 including B8A, reflectance × 10,000 |
| `solar_2023_global_y.npy` | (3463,), uint8 | 1 if any solar-panel pixel: 1,774 positive, 1,689 negative |

Add `--masks` to also save per-pixel masks.

**Encoder weights** are not downloaded yet. torchgeo fetches them on first use into `~/.cache/torch/hub/checkpoints/`.

### Data notes

- **No resizing, anywhere.** Resizing changes the ground resolution (256→64 makes each pixel 40 m instead of 10 m), blurs small objects and would make leakage look lower than it is. Every image goes into the encoders at native 10 m.
- **Patch size:** GloSoFarID patches are 256×256, EuroSAT 64×64. Each GloSoFarID patch is tiled into 16 native 64×64 tiles (~55k tiles), each labelled from its mask. All tiles from one patch stay in the same train/test split. Tiling needs the masks: rerun the loader with `--masks`.
- **ViT input size (resolved):** ViT-S/14 needs sizes divisible by 14, which 64×64 is not, so the plan uses `ViTSmall16` (16-pixel patches; 64 ÷ 16 = 4) instead.
- **BigEarthNet bands:** the 13-band `SENTINEL2_ALL_*` encoders need B10 filled (for example with zeros between B09 and B11). BigEarthNet is L2A while the encoders were trained on L1C, so report its results separately.
- **Disk:** the BigEarthNet archives (`BigEarthNet-S2.tar.gzaa/ab`, 63.5 GB) and the EuroSAT zip (2 GB) can be deleted after confirming loading works with `download=False`.

### Phase 2: Embeddings

Each frozen encoder ran once over EuroSAT and GloSoFarID at native 64×64, 10 m. GloSoFarID was first reloaded with masks:

```bash
python scripts/load_glosofarid.py data/glosofarid/solar_2023_global.tfrecord.gz --out data/glosofarid/npy --masks
python scripts/embed.py --sanity     # all encoders + random-init control, both datasets
```

Results (`--sanity` = linear probe, EuroSAT train → val; chance 10%):

| Encoder | Dim | EuroSAT val accuracy |
|---|---|---|
| ResNet50 DINO | 2048 | 97.2% |
| ResNet18 MoCo | 512 | 95.5% |
| ViT-S/16 DINO | 384 | 95.3% |
| ViT-S/16 MoCo | 384 | 94.1% |
| ResNet18 random (control) | 512 | 79.7% |

GloSoFarID tiles: 55,408, split by original patch 70/15/15.

| Split | Tiles | With solar |
|---|---|---|
| Train | 38,784 | 8,591 (22.2%) |
| Val | 8,304 | 1,818 (21.9%) |
| Test | 8,320 | 2,023 (24.3%) |

Outputs go to `data/embeddings/`: `splits_glosofarid.npz` (tile labels, patch ids, split by patch, 70/15/15) and per encoder `eurosat_{train,val,test}.npy`, `glosofarid.npy`, `meta.json`. EuroSAT is cached once to `data/eurosat/npy/` (official splits: 16,200 / 5,400 / 5,400), reordered to the encoders' band order (torchgeo's EuroSAT puts B8A last). The torchgeo weight `transforms` are not used because they resize to 256 and crop to 224; only the ÷ 10,000 normalisation is applied.

| Encoder (torchgeo) | Architecture | Pretraining | Embedding dim | Why |
|---|---|---|---|---|
| `ResNet18_Weights.SENTINEL2_ALL_MOCO` | CNN | MoCo | 512 | Baseline SSL4EO model |
| `ViTSmall16_Weights.SENTINEL2_ALL_MOCO` | Transformer | MoCo | 384 | Same method as ResNet18, different architecture |
| `ResNet50_Weights.SENTINEL2_ALL_DINO` | CNN | DINO | 2048 | Different SSL method |
| `ViTSmall16_Weights.SENTINEL2_ALL_DINO` | Transformer | DINO | 384 | Same method as ResNet50, different architecture |

All four take 13 bands. The ViTs use 16-pixel patches, so a 64×64 tile is 4×4 patches at native resolution. timm's `dynamic_img_size=True` resamples their positional embeddings from 14×14 to 4×4 at run time (a model change, not a data change); their EuroSAT accuracy above confirms they work at this size.

### Phase 3: Compression

```bash
python scripts/compress.py      # all encoders in data/embeddings, ~15 s
```

- 8 variants per encoder: `full` (uncompressed reference), `pca16`, `pca64`, `pca256`, each also as `_int8`.
- PCA and int8 scales are fitted on training rows only: EuroSAT train + GloSoFarID train tiles, subsampled to equal numbers (16,200 each). Val/test rows are only transformed.
- int8: symmetric per-dimension scale = max |value| on training rows / 127. Round-trip error is at most half a quantisation step; under 0.006% of values clip.
- Outputs in `data/compressed/<encoder>/<variant>/` (int8 variants store int8 + `scale.npy`; multiply to dequantise), PCA fits in `<encoder>/pca<k>.npz`, a summary in `<encoder>/meta.json`.

| Encoder | Variance kept: 16 / 64 / 256 dims | EuroSAT val acc: pca16 / pca16_int8 / pca64 |
|---|---|---|
| ResNet18 MoCo | 55.2% / 79.3% / 95.4% | 93.0% / 92.9% / 95.1% |
| ResNet50 DINO | 63.1% / 87.0% / 94.9% | 91.7% / 91.7% / 95.4% |
| ViT-S/16 DINO | 35.1% / 68.0% / 96.2% | 90.8% / 90.8% / 94.8% |
| ViT-S/16 MoCo | 49.9% / 81.8% / 98.8% | 87.0% / 87.0% / 93.6% |
| ResNet18 random | 99.2% / 99.8% / 100.0% | 69.3% / 69.3% / 76.3% |

Sizes per embedding: 16 B (`pca16_int8`) up to 8,192 B (ResNet50 `full`). int8 costs almost no utility. The random encoder's variance sits almost entirely in 16 directions, so "variance kept" is not comparable across encoders.

### Phase 4: Probes and controls

```bash
python scripts/probes.py        # all encoders x variants + controls, ~30 min on MPS
```

- **Tasks:** utility = EuroSAT land cover (test accuracy, chance 10%); leakage = GloSoFarID tile "solar farm present" (test ROC-AUC, chance 0.5).
- **Probes:** standardised logistic regression (C chosen on val) and a one-hidden-layer MLP (256 units, early stopping on val, 3 seeds).
- **Uncertainty:** 95% bootstrap CIs, 1,000 resamples; leakage resamples whole patches, not tiles.
- **Controls:** `resnet18_random` and `pixelstats` (per-band mean, std, p10, p50, p90; 65 features, no resizing).
- Results in `results/probes.csv` (164 rows), log in `results/probes.log`.

| Encoder | Utility, full (linear) | Leakage AUC, full: linear / MLP | Leakage AUC, pca16: linear / MLP |
|---|---|---|---|
| ViT-S/16 DINO | 96.3% | 0.937 / 0.947 | 0.889 / 0.919 |
| ResNet50 DINO | 97.4% | 0.934 / 0.934 | 0.855 / 0.907 |
| ViT-S/16 MoCo | 95.3% | 0.922 / 0.932 | 0.861 / 0.892 |
| ResNet18 MoCo | 96.6% | 0.894 / 0.904 | 0.791 / 0.863 |
| Pixel statistics (control) | 89.8% | 0.836 / 0.907 | n/a |
| ResNet18 random (control) | 79.9% | 0.796 / 0.824 | 0.762 / 0.812 |

Findings:
- **Most leakage is spectral.** Pixel statistics reach AUC 0.907 (MLP). The best embedding (ViT-S/16 DINO, 0.947) adds about 4 points, with non-overlapping CIs; ResNet18 MoCo only matches the pixel baseline.
- **H1 supported, with a qualifier:** leakage stays far above chance at 16–64 dims, but compressed embeddings add little beyond pixel statistics.
- **H2 modestly supported:** ResNet18 MoCo and ViT-S/16 DINO have near-equal utility (96.6% vs 96.3%) but different leakage (0.894 vs 0.937, CIs disjoint). Not yet tested with paired comparisons.
- **RQ2:** for ResNet18, full → pca16 costs 2.9 points of utility and 10.3 points of linear leakage, but only 4.1 points of MLP leakage. Compression makes leakage less linearly accessible rather than removing it.
- **int8 makes no difference** to utility or leakage.

---

## Planned

From Phase 4b on, every attacker and model is a research-grade model, tuned on val and evaluated on test. Linear probes stay only as a reference row, because they are the standard measure in the probing literature and carry the RQ2 finding above.

The central question from here: **what does an embedding reveal beyond the spectrum?** Every leakage number is reported against a control that removes the spectral shortcut (pixel statistics) or the location shortcut (a location prior: the average mask at that tile position, since GloSoFarID patches were likely collected centred on farms).

### Models by phase

Each model is chosen for what it lets the study measure:
- **Vector attackers span three model families** (neural, tree, kernel), so no leakage finding rests on one family.
- **Repeats are kept where they help a comparison:** the deep MLP extends the Phase 4 linear → MLP ladder; the upper-bound CNN is a ResNet-18 so it matches the audited ResNet18; the own encoder is a ViT so it can sit beside the audited ViT-S/16s.
- **Controls** use the same architecture as the model they control (the pixel-statistics inverter mirrors the strong inverter).
- **Measurement tools** stay identical across every encoder, compression level and task they compare (the 4b attackers score attribute, spatial and utility tasks alike; the panel segmenter scores every inverter).
- **Frozen loss network:** the strong inverter's perceptual loss uses ResNet50 MoCo, deliberately not an audited encoder, so the loss cannot favour one.

| Phase | Component | Family | Model | Training | Scored by |
|---|---|---|---|---|---|
| 4b | Attribute attackers | MLP, tree ensemble, kernel | Deep residual MLP (3 blocks, 512 units, LayerNorm, GELU, dropout); LightGBM; RBF-kernel SVM (Nyström approximation) | Hyperparameter search on val | Test ROC-AUC, paired bootstrap |
| 4b | Spatial probe | Same three | The 4b attackers, one per cell of a 4×4 grid inside a tile (reuse by design) | Same | Grid IoU vs location prior |
| 4b | Utility on BigEarthNet | Same three | The 4b attackers, one per label, 19 labels (reuse by design) | Same | Macro mAP |
| 4b | Image-level upper bound | CNN | ResNet-18 trained from scratch on raw 64×64×13 tiles (matches the audited ResNet18) | Supervised; flips and 90° rotations only | Leakage available in the image itself |
| 5 | Weak inverter | Non-parametric | k-nearest-neighbour retrieval: reconstruct a tile as the mean of the k training tiles closest in embedding space | k chosen on val | SSIM, LPIPS (RGB bands), spectral angle |
| 5 | Strong inverter | GAN (CNN) | Generator: decoder with residual blocks and self-attention at 8×8 and 16×16; discriminator: PatchGAN | L1 + multispectral perceptual loss (frozen ResNet50 MoCo, not audited) + adversarial loss | Same, plus mask IoU |
| 5 | Pixel-statistics inverter (control) | GAN (CNN) | Same GAN, input = pixel statistics (reuse by design) | Same | Learned leakage = embedding inverter minus this |
| 5 | Panel segmenter | CNN | U-Net on real GloSoFarID tiles and masks | Dice + BCE | Run on reconstructions: detection and mask IoU |
| 6 | Localisation attack | Transformer + CNN head | Query-based decoder (DETR / Mask2Former style): memory tokens from the embedding, 64 learnable queries (8×8), 4 cross-attention layers, conv upsampling head | Dice + focal loss | Mask IoU vs location prior and pixel-statistics control |
| 7 | Own encoder, baseline | Transformer | ViT-Tiny/Small, 8-pixel patches on 64×64×13 (64 tokens) | Masked autoencoder (MAE), 75% masking, train-split tiles only | Phase 4b–6 attackers |
| 7 | Own encoder, defended | Transformer + MLP heads | Same ViT and budget, plus a solar-farm adversary behind a gradient-reversal layer | MAE + land-cover head − adversary | Phase 4b–6 attackers, retrained fresh |
| 7 | Defence filter (optional) | MLP-based | Variational information bottleneck (stochastic encoder with a KL penalty) on frozen pretrained embeddings, plus the gradient-reversal adversary | Utility head + KL + adversary | Phase 4b–6 attackers |

LightGBM is not installed yet (`uv pip install lightgbm`).

Out of scope: diffusion-based inversion (too heavy for the timeline on a Mac). Named as a limitation: stronger attacks can only reveal more.

### Phase 4b: Learned-leakage follow-ups (pre-registered)

Written down before running; every one is reported whatever it shows.

- **Harder utility task (BigEarthNet):** EuroSAT accuracy is 95–97% for every encoder, too close to its ceiling to separate them, so "matched utility" for H2 is measured on BigEarthNet's 19-label land-cover task instead (macro mAP).
  - Subset of the official splits (237,871 / 122,342 / 119,825 patches available), for example 50k / 10k / 10k.
  - Centre-cropped to 112×112, which is cropping, not resizing: ViT-S/16 needs a multiple of 16, and 120 is not one. Labels cover the full 120×120 patch; the cropped 4-pixel border (40 m) is a small source of label noise.
  - B10 zero-filled between B09 and B11. BigEarthNet is L2A while the encoders were trained on L1C, so this shift affects all encoders and results are reported separately from EuroSAT.
  - Compressed with the existing Phase 3 PCA and int8 fits (transform only, no refit).
- **Incremental leakage:** attacker on pixel statistics alone vs pixel statistics + embedding. The AUC gain is the learned leakage.
- **Per-class leakage:** AUC within each land-cover class, removing the "bright built-up surface" shortcut. GloSoFarID tiles get land-cover pseudo-labels from the best EuroSAT utility attacker.
- **Location prior check:** test from the masks whether panels really concentrate at the patch centre before relying on that baseline.
- **Spatial probe:** 4×4 grid localisation inside a tile, against the location prior.
- **Image-level upper bound:** how much leakage the raw tile itself allows.
- **Paired tests for H2:** paired bootstrap of the leakage difference between encoders on the same test patches; leakage compared at matched utility on each encoder's trade-off curve.
- **Robustness:** 3 train/val/test split seeds, not only probe seeds.
- **Coarser quantisation:** int4, int2 and 1-bit, to find where quantisation starts to cut leakage and utility.

The full pre-registration, with primary and secondary analyses, Holm correction and its disclosure, is in `results/preregistration.md`.

**Status: scripts built and tested end to end on small variants (numbers not used); full run not started.**

Run everything, in pre-registered order (resumes if interrupted; roughly 15–18 h in total, best overnight):

```bash
bash scripts/run_phase4b.sh 2>&1 | tee results/phase4b/run.log
```

The script runs these steps; each can also be run on its own:

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python scripts/probes.py --out results/probes_v2.csv --preds-dir results/preds/phase4` | Phase 4 probes again, saving per-tile predictions and validation scores (needed for paired tests) | ~30 min |
| 2 | `python scripts/quantise.py` then `python scripts/probes.py --comp-root data/quantised --out results/probes_quant.csv --preds-dir results/preds/phase4_quant` | Quantile codes at 8/4/2/1 bits, then linear + MLP probes on them | ~1 h |
| 3 | `python scripts/splits.py --seeds 1 2` | Two extra train/val/test splits by patch | ~2 min |
| 4 | `python scripts/bigearthnet.py` | BigEarthNet subset (30k / 10k / 10k): embed with all encoders, compress with the existing fits, pixel statistics | ~25 min |
| 5 | `python scripts/attack4b.py --tasks leakage leakage_incr utility_eurosat utility_ben spatial` | Deep MLP, LightGBM and RBF SVM on every task, encoder and float variant | ~10–14 h |
| 6 | `python scripts/upperbound.py` | ResNet-18 from scratch on raw tiles | ~45 min |
| 7 | `python scripts/attack4b.py --tasks leakage leakage_incr --variants full --split-seed 1` (and 2) | Primary comparisons on the extra splits | ~2.5 h |
| 8 | `python scripts/analysis4b.py` | All pre-registered tests, into `results/phase4b/analysis/` | ~5 min |

Implementation notes:
- **Attacker selection:** for each cell, the attacker with the best **validation** score is selected and its test score is the leakage estimate. Phase 4's linear and MLP probes are candidates too. (Amended before any 4b run: the first version selected on test, which inflates estimates.)
- **Quantisation scheme:** q8/q4/q2/q1 use per-dimension quantile (equal-frequency) bins with bin-mean reconstruction, fitted on the Phase 3 PCA fit rows. Phase 3's max-scaled int8 would collapse most values to zero at 4 bits or fewer. q8 matches float (bridge to Phase 3's int8).
- **BigEarthNet bands:** 20 m and 60 m bands are put on the 10 m grid by nearest-neighbour copying (each value repeated 2×2 or 6×6; no values invented, ground extent unchanged), matching how EuroSAT and GloSoFarID are delivered. Approved as consistent with the no-resizing rule.
- **Location prior (checked):** panels concentrate at patch centres. P(tile has panels) is 0.42 for the 4 centre tiles and 0.11–0.12 for the corners; tile position alone gives test AUC 0.659. Spatial baselines therefore use the average mask for that tile position, deliberately stricter than the attackers, who never see position.
- **LightGBM settings:** 63 histogram bins, learning rate 0.1, early stopping after 50 rounds: 3.6× faster than the defaults on 2048-dim inputs, test AUC 0.933 vs 0.935.
- **macOS:** LightGBM needs `brew install libomp` and must be imported before torch (both ship an OpenMP runtime). `attackers.py` handles the order.

Measured on the largest input (ResNet50 DINO, 2048 dims, full-size leakage):

| Attacker | Val AUC | Test AUC | Time | Peak memory |
|---|---|---|---|---|
| Deep MLP | 0.931 | 0.926 | 2.7 min | n/a |
| LightGBM | 0.931 | 0.933 | 4.3 min | 2.8 GB |
| RBF SVM | 0.931 | 0.937 | 1.4 min | 4.2 GB |

### Phase 5: Inversion

**Status: scripts built and tested end to end (1–2-epoch runs, numbers not used); full run not started.**

Run after Phase 4b has finished (both use the GPU, and RQ3 reads Phase 4b's results). Roughly 14–20 h:

```bash
mkdir -p results/phase5 && bash scripts/run_phase5.sh 2>&1 | tee results/phase5/run.log
```

The analysis plan is pre-registered in `results/preregistration.md` (Phase 5 section), written before any Phase 5 model was trained: 8 primary tests (learned and spatial leakage through GAN inversion at full size, Holm-corrected), with secondary analyses and RQ3's rank correlations.

Measured times:

| Step | Time |
|---|---|
| Segmenter | ~2 min per epoch, 40 max (~80 min) |
| Mean baseline, kNN | under 5 min per run |
| GAN | ~3.7 min per warm-up epoch, ~5.1 min per adversarial epoch; 20 max with early stopping, ~1–1.7 h per GAN |

Fixes found in testing: the perceptual network now loads the full pretrained ResNet50 MoCo and reads stages 1–3 directly (timm's `features_only` dropped stage 4 and broke weight loading).

| Step | Command | What it does |
|---|---|---|
| 1 | `python scripts/segmenter.py` | U-Net panel segmenter on real tiles; threshold on val; real-image ceiling and location-prior baseline in `results/phase5/segmenter.json` |
| 2 | `python scripts/invert.py --method mean` | Mean-tile baseline |
| 3 | `python scripts/invert.py --method knn --encoder <enc> --variant <var>` | k-NN retrieval inverter (k chosen on val), all 5 encoders at `full` and `pca16`, plus pixel statistics |
| 4 | `python scripts/invert.py --method gan --encoder <enc> --variant <var>` | GAN inverter: 4 audited encoders at `full` and `pca16`, plus the pixel-statistics and random-ResNet18 controls |
| 5 | `python scripts/analysis5.py` | Pre-registered tests into `results/phase5/analysis/`: `summary.csv`, `primary.csv`, `secondary.csv`, `rq3.csv` |

Every run writes `results/phase5/<method>/<encoder>/<variant>/` (`metrics.json`, per-tile scores for paired tests, 16 example reconstructions) and a row in `results/phase5/inversion.csv`.

- Weak (k-nearest-neighbour retrieval) and strong (GAN with a self-attention generator) inverters, plus the pixel-statistics GAN as a control.
- The strong inverter's perceptual loss uses frozen ResNet50 MoCo, which is not one of the audited encoders, so the loss cannot favour any of them.
- Trained on train tiles only; never evaluated on tiles an inverter was trained on.
- Scored on pixel metrics (RQ3) and spatially, by running the panel segmenter on reconstructions (mask IoU vs location prior).

### Phase 6: Localisation attack

- Query-based transformer decoder from embedding to a 64×64 panel mask, on every encoder and compression level.
- Pixel statistics cannot localise, so localisation above the location prior is learned leakage.
- Compared with the inverters on the same mask-IoU score.

### Phase 7: Own encoder and defence

- Train the same small ViT twice with the same data and budget: baseline MAE, and MAE with the adversarial leakage penalty.
- The only difference between the two is the privacy objective, so any change in leakage is caused by the defence.
- Realistic target: remove the learned excess over pixel statistics, not reach AUC 0.5, since spectral leakage is inherent to any useful embedding.
- Evaluated only against attackers trained fresh on its outputs (Phase 4b–6), never only against its own adversary.

### Phase 8: Analysis and paper

- Main figure: utility (x) vs leakage (y), one line per encoder, one point per compression level, with the pixel-statistics level marked. Utility on BigEarthNet mAP for H2; EuroSAT kept as a second panel.
- Report mean ± std over seeds and 95% CIs; paired tests for encoder comparisons.
- Test hypotheses H1–H3 and the Phase 4b–7 questions.
- Paper, about 6 pages: introduction, related work, setup, results, discussion, conclusion.

**Scope risk:** Phases 5–7 add four models trained from scratch (strong inverter, localisation attack, own encoder twice). If time runs short, cut Phase 7 first; Phases 4b–6 make a complete paper.

## Repository layout

```
data/                      # downloaded datasets (git-ignored)
  eurosat/
  glosofarid/
    solar_2023_global.tfrecord.gz
    npy/
  bigearthnet/
  embeddings/              # Phase 2 outputs (+ BigEarthNet ben_*.npy, split seeds 1-2)
  compressed/              # Phase 3 outputs (+ pixelstats control)
  quantised/               # Phase 4b q8/q4/q2/q1 variants
eoenv/                     # Python virtual environment
scripts/
  load_glosofarid.py       # GloSoFarID TFRecords -> numpy (+ masks)
  embed.py                 # Phase 2: tiling, splits, encoders
  compress.py              # Phase 3: PCA + int8
  probes.py                # Phase 4: linear + MLP probes, controls
  common.py                # shared loading, tile cache, location prior, bootstrap, paired tests, Holm
  attackers.py             # Phase 4b: deep MLP, LightGBM, RBF SVM
  quantise.py              # Phase 4b: quantile codes at 8/4/2/1 bits
  splits.py                # Phase 4b: extra split seeds
  bigearthnet.py           # Phase 4b: BigEarthNet utility task
  attack4b.py              # Phase 4b: attackers on every task
  upperbound.py            # Phase 4b: ResNet-18 image-level upper bound
  analysis4b.py            # Phase 4b: pre-registered tests
  imagemodels.py           # Phase 4b/5: ResNet-18, U-Net, GAN generator and discriminator, perceptual net
  segmenter.py             # Phase 5: panel segmenter
  invert.py                # Phase 5: mean, kNN and GAN inverters + evaluation
  analysis5.py             # Phase 5: pre-registered tests
  run_phase4b.sh           # runs all of Phase 4b
  run_phase5.sh            # runs all of Phase 5
results/
  preregistration.md       # Phase 4b pre-registration
  probes.csv               # Phase 4 results
  probes.log
  phase4b/                 # runs.csv, analysis/
  phase5/                  # inversion.csv, per-run folders
  preds/                   # per-tile test predictions
  models/                  # trained image models
```
