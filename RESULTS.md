# Results, phase by phase

Running record of what each phase found. The methods are in README.md and the pre-registered tests in
results/preregistration.md. Phases 0–8 are complete.

**Conventions**
- Leakage means "does this 64×64 GloSoFarID tile contain solar panels?", scored as test ROC-AUC (chance 0.5).
- Utility means EuroSAT land cover (accuracy, chance 10%) or BigEarthNet 19 labels (macro AP).
- CIs are 95% cluster-bootstrap intervals over GloSoFarID patches (1,000 resamples).
- From Phase 4b on, each number is the test score of the attacker that scored best on **validation**.

| Short name | Encoder |
|---|---|
| R18-MoCo | ResNet18, MoCo |
| R50-DINO | ResNet50, DINO |
| ViT-MoCo / ViT-DINO | ViT-S/16, MoCo / DINO |
| random | ResNet18 with random weights (control) |
| pixelstats | 65 per-band statistics (control) |

---

## Phase 0: Environment
Python venv `eoenv` on the MacBook (Apple Silicon, MPS): torch 2.15.0, torchgeo 0.10.1, scikit-learn 1.9.1,
timm 1.0.31, torchmetrics 1.9.0, lightgbm 4.7.0. All runs use this one machine.

## Phase 1: Data

| Dataset | Use | Size | Check |
|---|---|---|---|
| EuroSAT MS (L1C, 13 bands) | Utility | 27,000 patches, 64×64 | 27,000 files |
| GloSoFarID 2023 global (L1C, 13 bands) | Leakage | 3,463 patches, 256×256, with masks; 1,774 contain solar | md5 matches Zenodo |
| BigEarthNet v2 S2 (L2A, 12 bands) | Second utility task | 549,488 patches, 120×120 | all patches extracted |

No resizing anywhere:
- GloSoFarID patches are cut into 16 native 64×64 tiles, giving 55,408 tiles, split by patch 70/15/15.
- BigEarthNet's 20/60 m bands are put on the 10 m grid by nearest-neighbour copying, B10 is zero-filled and patches
  are centre-cropped to 112.

## Phase 2: Embeddings
Frozen encoders at native 64×64 input. Sanity check: linear probe on EuroSAT, validation accuracy.

| Encoder | Dim | EuroSAT val acc |
|---|---|---|
| R50-DINO | 2048 | 97.2% |
| R18-MoCo | 512 | 95.5% |
| ViT-DINO | 384 | 95.3% |
| ViT-MoCo | 384 | 94.1% |
| random | 512 | 79.7% |

The pretrained encoders work at 64×64 without resizing. The random encoder sits well below them, as a control should.
Share of tiles with solar panels: train 22.2%, val 21.9%, test 24.3%.

## Phase 3: Compression (PCA 16/64/256, ± int8)

| Encoder | Variance kept, 16 / 64 / 256 dims | EuroSAT val acc, pca16 / pca16_int8 / pca64 |
|---|---|---|
| R18-MoCo | 55% / 79% / 95% | 93.0 / 92.9 / 95.1 |
| R50-DINO | 63% / 87% / 95% | 91.7 / 91.7 / 95.4 |
| ViT-DINO | 35% / 68% / 96% | 90.8 / 90.8 / 94.8 |
| ViT-MoCo | 50% / 82% / 99% | 87.0 / 87.0 / 93.6 |

- int8 costs at most 0.1 points.
- Even 16 dimensions (16 bytes) keep 87–93% EuroSAT accuracy.

## Phase 4: Linear and MLP probes

| Encoder | EuroSAT acc (linear) | Leakage AUC, full: linear / MLP | Leakage AUC, pca16: linear / MLP |
|---|---|---|---|
| ViT-DINO | 96.3% | 0.937 / 0.947 | 0.889 / 0.919 |
| R50-DINO | 97.4% | 0.934 / 0.934 | 0.855 / 0.907 |
| ViT-MoCo | 95.3% | 0.922 / 0.932 | 0.861 / 0.892 |
| R18-MoCo | 96.6% | 0.894 / 0.904 | 0.791 / 0.863 |
| pixelstats | 89.8% | 0.836 / 0.907 | n/a |
| random | 79.9% | 0.796 / 0.824 | 0.762 / 0.812 |

Takeaways:
- **Most of the leakage is spectral.** Pixel statistics alone reach AUC 0.907.
- **H1 holds with a qualifier:** leakage stays far above chance even at 16 dimensions, but compressed embeddings
  add little beyond pixel statistics.
- **H2 is modestly supported** (no paired test yet).
- **RQ2 is nuanced:** compression makes leakage less *linearly* accessible. The MLP still recovers most of it.
- **int8 changes nothing.**

---

## Phase 4b: Stronger attackers, pre-registered tests

**Setup**
- Attackers: deep residual MLP, LightGBM and RBF SVM, with hyperparameters chosen on validation. The final model is
  whichever scored best on validation, with Phase 4's linear probe and MLP also among the candidates.
- New tasks: incremental leakage (embedding + pixelstats), spatial leakage (which of the 16 cells contain panels),
  and BigEarthNet utility (30k / 10k / 10k patches).
- Also: an image-level upper bound (a ResNet-18 trained from scratch on raw pixels), quantile quantisation down to
  1 bit, and split seeds 1 and 2 as replications.

### 4b.1 Leakage with the stronger attackers (seed 0)

| Encoder | Phase 4 MLP | 4b best | Attacker chosen | pca256 | pca64 | pca16 |
|---|---|---|---|---|---|---|
| ViT-DINO | 0.947 | **0.948** | SVM | 0.939 | 0.941 | 0.918 |
| R50-DINO | 0.934 | 0.933 | LightGBM | 0.932 | 0.925 | 0.905 |
| ViT-MoCo | 0.932 | 0.932 | SVM | 0.922 | 0.917 | 0.891 |
| R18-MoCo | 0.904 | 0.904 | deep MLP | 0.900 | 0.895 | 0.865 |
| pixelstats | 0.907 | 0.910 | deep MLP | | | |
| random | 0.824 | 0.829 | deep MLP | 0.823 | 0.826 | 0.813 |

**The stronger attackers add at most +0.005 AUC.** The Phase 4 MLP had already reached close to the ceiling of what
these embeddings reveal, so the leakage estimates hold up whichever attacker is used. Thanks to the extra attackers,
this is now shown rather than assumed.

### 4b.2 Image-level upper bound
A CNN trained on the raw tiles reaches **AUC 0.970** (CI 0.963–0.977).

| Representation | AUC | Gap to raw pixels [95% CI] |
|---|---|---|
| ViT-DINO | 0.948 | 0.022 [0.015, 0.030] |
| R50-DINO | 0.933 | 0.038 [0.029, 0.046] |
| ViT-MoCo | 0.932 | 0.039 [0.031, 0.047] |
| pixelstats | 0.910 | 0.060 [0.050, 0.072] |
| R18-MoCo | 0.905 | 0.065 [0.054, 0.077] |

All the gaps are significant (p = 0.001). The best embedding keeps about **95%** of the above-chance detection signal
in the raw image: (0.948 − 0.5) / (0.970 − 0.5). R18-MoCo retains *less* than plain pixel statistics.

### 4b.3 Pre-registered primary tests (seed 0, Holm-corrected over 6 tests)

| Test | A vs B | AUC A / B | Difference [95% CI] | p (Holm) |
|---|---|---|---|---|
| Incremental leakage | R18-MoCo + px vs px | 0.926 / 0.910 | +0.016 [0.007, 0.027] | 0.006 ✓ |
| Incremental leakage | ViT-MoCo + px vs px | 0.936 / 0.910 | +0.026 [0.016, 0.036] | 0.006 ✓ |
| Incremental leakage | R50-DINO + px vs px | 0.941 / 0.910 | +0.031 [0.022, 0.042] | 0.006 ✓ |
| Incremental leakage | ViT-DINO + px vs px | 0.949 / 0.910 | +0.039 [0.030, 0.049] | 0.006 ✓ |
| H2 architecture | ViT-MoCo vs R18-MoCo | 0.932 / 0.905 | +0.026 [0.019, 0.035] | 0.006 ✓ |
| H2 architecture | ViT-DINO vs R50-DINO | 0.948 / 0.933 | +0.016 [0.010, 0.022] | 0.006 ✓ |

(px = pixelstats.) **All 6 primary tests are significant.**

Replications on split seeds 1 and 2 (not part of the Holm family):

| Test | Seed 1 | Seed 2 |
|---|---|---|
| Incr. R18-MoCo | +0.017 ✓ | **+0.002, p = 0.31 ✗** |
| Incr. ViT-MoCo | +0.023 ✓ | +0.015 ✓ |
| Incr. R50-DINO | +0.029 ✓ | +0.017 ✓ |
| Incr. ViT-DINO | +0.034 ✓ | +0.026 ✓ |
| H2 ViT-MoCo vs R18-MoCo | +0.032 ✓ | +0.034 ✓ |
| H2 ViT-DINO vs R50-DINO | **+0.007, p = 0.07 ✗** | +0.017 ✓ |

Interpretation:
- **Every audited embedding leaks object information beyond what pixel statistics carry.** The effect is robust for
  the ViTs and R50-DINO. For R18-MoCo it is small (+0.002 to +0.017) and fails to replicate on one split.
- The size ranking is stable across all three seeds: ViT-DINO > R50-DINO > ViT-MoCo > R18-MoCo.
- **H2 is now strongly supported for the MoCo pair** (3/3 seeds, about +0.03). It is supported for the DINO pair on
  2/3 seeds, with a smaller effect (+0.007 to +0.017).
- Sanity check: adding random-encoder features to pixelstats *lowers* AUC (0.891 vs 0.910). The gains above come from
  real information, not from extra dimensions.

### 4b.4 H2 at matched utility
Leakage is interpolated along each encoder's compression curve at the same BigEarthNet macro AP:

| Pair | Leakage gap (ViT − ResNet) across matched-utility points |
|---|---|
| ViT-MoCo vs R18-MoCo | +0.012 to +0.030 |
| ViT-DINO vs R50-DINO | +0.010 to +0.019 |

**At equal utility, the ViTs always leak more.** The architecture effect isn't just a by-product of the ViTs being
better encoders.

### 4b.5 Utility (seed 0, best attacker = deep MLP in almost every case)

| Encoder | BigEarthNet macro AP: full / pca256 / pca64 / pca16 | EuroSAT acc: full / pca16 |
|---|---|---|
| ViT-DINO | **0.680** / 0.675 / 0.646 / 0.606 | 96.7 / 93.3 |
| R50-DINO | 0.666 / 0.663 / 0.636 / 0.617 | **97.4** / 93.6 |
| ViT-MoCo | 0.659 / 0.650 / 0.636 / 0.558 | 96.4 / 90.4 |
| R18-MoCo | 0.617 / 0.623 / 0.613 / 0.570 | 96.5 / 94.8 |
| pixelstats | 0.636 | 95.1 |
| random | 0.538 / 0.549 / 0.545 / 0.521 | 82.2 / 77.7 |

- **BigEarthNet separates the encoders where EuroSAT is near its ceiling.** All four score 96–97% on EuroSAT, but
  0.62–0.68 on BigEarthNet. This fixes the Phase 4 flaw that utility was near its ceiling.
- R18-MoCo is *below* pixelstats on BigEarthNet as well.
- **Compression costs utility faster than it removes leakage.** Going from full size to pca16 changes:

  | Encoder | BigEarthNet AP | Leakage AUC |
  |---|---|---|
  | ViT-MoCo | −0.10 | −0.04 |
  | ViT-DINO | −0.07 | −0.03 |
  | R18-MoCo | −0.05 | −0.04 |

  As a defence, plain PCA has a poor trade-off.

### 4b.6 Quantile quantisation (MLP probe, full-size embedding)

| Encoder | Leakage AUC: 32 / 4 / 2 / 1 bit | EuroSAT acc: 32 / 4 / 2 / 1 bit |
|---|---|---|
| ViT-DINO | 0.947 / 0.946 / 0.944 / 0.936 | 96.6 / 96.6 / 96.2 / 95.1 |
| R50-DINO | 0.934 / 0.933 / 0.930 / 0.928 | 97.2 / 97.1 / 97.0 / 96.4 |
| ViT-MoCo | 0.932 / 0.932 / 0.924 / 0.916 | 95.9 / 95.8 / 95.3 / 93.8 |
| R18-MoCo | 0.904 / 0.900 / 0.895 / 0.808 | 96.7 / 96.5 / 95.3 / 82.1 |

- 4 bits per dimension is lossless for both leakage and utility. At 1 bit the large encoders still leak 0.92–0.94.
- **Quantisation is not a defence either.** Leakage and utility only collapse together (R18-MoCo at 1 bit).
- pca16 at 1 bit (2 bytes per tile) still gives 0.80–0.86 leakage AUC.

### 4b.7 Spatial leakage: *where* the panels are (16 cells per patch)
Location prior (cell position only): grid IoU 0.497.

| Encoder | Macro AP: full / pca256 / pca64 / pca16 | Grid-IoU gain over the prior, full → pca16 |
|---|---|---|
| ViT-MoCo | **0.897** / 0.888 / 0.873 / 0.830 | +0.056 → +0.006 |
| R50-DINO | 0.890 / 0.867 / 0.842 / 0.814 | +0.055 → +0.005 |
| R18-MoCo | 0.873 / 0.869 / 0.843 / 0.814 | +0.039 → +0.003 (n.s. at α = 0.01) |
| ViT-DINO | 0.859 / 0.862 / 0.855 / 0.837 | +0.010 → +0.007 |
| pixelstats | 0.851 | +0.009 |
| random | 0.850 / 0.852 / 0.848 / 0.823 | +0.024 → +0.015 |

- **Location leaks in a different way from presence.** ViT-DINO, the strongest encoder on presence, is barely better
  than pixelstats at locating panels. ViT-MoCo and R50-DINO locate them best.
- **Compression removes location information far more than presence information.** At pca16, every audited encoder
  is within about 0.006 IoU of the position-only prior, while presence AUC stays at 0.87–0.92.

### 4b.8 Leakage by land-cover type (EuroSAT pseudo-labels on GloSoFarID tiles)
- Detection is easiest on agricultural and natural backgrounds: AnnualCrop, Forest and HerbaceousVegetation reach
  0.94–0.97 for the DINO encoders.
- It is hardest where panels resemble the surroundings: Highway, Industrial and Residential reach 0.79–0.90.
- The encoder ranking (ViT-DINO > R50-DINO ≈ ViT-MoCo > R18-MoCo) holds in almost every class.

### Phase 4b status of the hypotheses

| | Status after 4b |
|---|---|
| **RQ1 / H1**, leakage above chance and beyond pixels | **Supported.** All 4 incremental tests pass (Holm). R18-MoCo's margin is small and fragile. Most of the signal is still spectral (pixelstats 0.910). |
| **H2**, ViT leaks more than CNN | **Supported.** Significant for both pairs, holds at matched utility, and replicates fully for MoCo (3/3 seeds) and partly for DINO (2/3). |
| **RQ2**, compression trade-off | **Clearer now.** PCA and quantisation cut utility at least as fast as presence leakage, so neither is a defence. PCA *does* strip location information. |
| Attacker strength | Saturated. The stronger attackers add ≤ 0.005 AUC, and the embeddings sit 0.02–0.065 below the raw-image upper bound. |

**What this means for Phase 7 (to be pre-registered before the defended run):**
- Presence leakage is mostly spectral and closely tied to land-cover utility, so a presence-only adversary is unlikely
  to push it below the pixelstats level (≈ 0.91) without hurting utility.
- Location information is what the embedding adds and what compression removes. The defended encoder should
  therefore target spatial leakage as well, or instead.

---

## Phase 5: Inversion (rebuild the tile from its embedding)

**Setup**
- A U-Net **segmenter** trained on real tiles finds panels in any image: on real test tiles it reaches detection AUC
  **0.939** and mask IoU **0.608**. These are the ceiling for what the inverted images can show.
- Three inverters:
  - **Mean tile:** the same average tile for everyone (a floor).
  - **kNN:** returns the training tile whose embedding is closest.
  - **GAN:** a self-attention generator with a PatchGAN discriminator, trained on 10·L1 + perceptual loss (a frozen
    ResNet50-MoCo, which is not one of the audited encoders) + 0.1·hinge.
- The segmenter is run on every reconstruction: **detection AUC** (are panels visible in the rebuilt tile?) and
  **mask IoU** (are they in the right place?). Pixel metrics: L1, SSIM, LPIPS, spectral angle.
- The location prior (mean mask for that tile position) scores mask IoU 0.335.
- 1 mean + 11 kNN + 10 GAN runs; GANs on the 4 audited encoders at full size and 16 dims, plus pixelstats and the random encoder.

### 5.1 Reconstructions from full-size embeddings (test set, 8,320 tiles)

| Inverter | Input | Detection AUC | Mask IoU | L1 ↓ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|---|
| Mean tile | none | 0.500 | 0.000 | 0.643 | 0.853 | 0.624 |
| GAN | ViT-MoCo | **0.788** | **0.253** | **0.206** | 0.892 | **0.300** |
| GAN | ViT-DINO | 0.774 | 0.230 | 0.244 | 0.889 | 0.345 |
| GAN | R18-MoCo | 0.762 | 0.239 | 0.274 | 0.886 | 0.313 |
| GAN | R50-DINO | 0.760 | 0.236 | 0.260 | 0.891 | 0.375 |
| GAN | random (control) | 0.766 | 0.254 | 0.209 | 0.890 | 0.296 |
| GAN | pixelstats (control) | 0.711 | 0.160 | 0.219 | 0.892 | 0.365 |
| kNN | ViT-MoCo | 0.776 | 0.236 | 0.306 | 0.891 | 0.346 |
| kNN | ViT-DINO | 0.773 | 0.204 | 0.289 | 0.892 | 0.346 |
| kNN | R50-DINO | 0.726 | 0.185 | 0.307 | 0.890 | 0.347 |
| kNN | R18-MoCo | 0.714 | 0.192 | 0.326 | 0.889 | 0.348 |

- **Panels are detectable in rebuilt tiles,** at AUC 0.76–0.79 against 0.500 for the mean tile. That is well below
  the 0.93–0.95 that a classifier reads straight off the embedding, so reconstruction throws away a lot of the signal.
- **Pixel quality barely separates the encoders.** SSIM is 0.88–0.89 for every input, including pixelstats.

### 5.2 Pre-registered primary tests (GAN, full size, Holm over 8)

| Test | Encoder | Difference [95% CI] | Holm p |
|---|---|---|---|
| Detection AUC, embedding minus pixelstats GAN | ViT-MoCo | **+0.076** [0.064, 0.090] | 0.008 ✓ |
| | ViT-DINO | **+0.063** [0.049, 0.078] | 0.008 ✓ |
| | R18-MoCo | **+0.051** [0.035, 0.066] | 0.008 ✓ |
| | R50-DINO | **+0.049** [0.034, 0.065] | 0.008 ✓ |
| Mask IoU, embedding minus location prior | ViT-MoCo | −0.082 [−0.104, −0.062] | 1.0 ✗ |
| | ViT-DINO | −0.105 [−0.122, −0.090] | 1.0 ✗ |
| | R18-MoCo | −0.096 [−0.116, −0.078] | 1.0 ✗ |
| | R50-DINO | −0.099 [−0.117, −0.081] | 1.0 ✗ |

- **Panels are recoverable as "present" from every audited embedding, beyond what pixel statistics give** (4/4).
- **Panels are not recoverable as "where".** Every GAN's mask IoU (0.23–0.25) is *below* the position-only prior (0.335),
  so no reconstruction beats simply guessing from tile position.

### 5.3 Secondary results (exploratory)

- **GAN vs kNN:**
  - At full size the GAN is ahead of the kNN on detection for all 4 encoders (+0.001 to +0.048; clearly so for R18-MoCo
    and R50-DINO) and on mask IoU for all 4 (+0.016 to +0.052).
  - At 16 dimensions the kNN is as good or better for 3 of 4 encoders (e.g. R50-DINO detection 0.704 vs 0.671).
  - A GAN trained on 16 numbers has little to work with, and retrieving a real tile is then the safer attack.
- **Compression hurts inversion more than it hurts classification.** GAN detection AUC at 16 dims is 0.74 for the ViTs
  and 0.67–0.68 for the ResNets (full size: 0.76–0.79). For R18-MoCo and R50-DINO this falls below the pixelstats GAN (0.711).
- **The random encoder is as good as the trained ones** (GAN detection 0.766, vs 0.76–0.79). Much of what the GAN uses
  is generic spectral information that any encoder passes through. The pre-registered contrast against pixelstats is what
  isolates the learned part, and all four audited encoders clear it.
- **RQ3 (do pixel metrics rank encoders like object measures?), 9 encoder/variant pairs, Kendall τ:**

  | Pixel metric | vs GAN detection AUC | vs GAN mask IoU | vs Phase 4b probe AUC |
  |---|---|---|---|
  | SSIM | **0.67** (p = 0.013) | **0.67** (p = 0.013) | 0.39 (p = 0.18) |
  | LPIPS | 0.28 (p = 0.36) | 0.50 (p = 0.075) | −0.22 (p = 0.48) |

  **H3 is only partly supported.** LPIPS does not track object-level leakage (none of its correlations is significant),
  and neither pixel metric tracks the probe leakage from Phase 4b. SSIM does track the detection and IoU measured on the
  *same reconstructions*, but with SSIM values spanning only 0.88–0.89 and n = 9, this is a weak result.

---

## Phase 6: Localisation attack (draw the panel mask from the embedding)

**Setup:** a query-based transformer decoder (DETR/Mask2Former style; 5–13M parameters) maps an embedding directly to a
64×64 mask, trained with Dice + focal loss, early-stopped on validation IoU. 18 runs: pixelstats; 4 audited encoders ×
{full, pca256, pca64, pca16}; random. Baselines: the location prior and the same model on pixelstats.

### 6.1 Primary tests (full size, mask IoU on tiles that contain panels, Holm over 8)

| Encoder | IoU | vs location prior [95% CI] | vs pixelstats [95% CI] | Holm p |
|---|---|---|---|---|
| R18-MoCo | 0.344 | **+0.0085** [0.004, 0.013] | **+0.0068** [0.004, 0.010] | 0.008 ✓ ✓ |
| ViT-MoCo | 0.342 | **+0.0069** [0.003, 0.011] | **+0.0051** [0.004, 0.007] | 0.008 ✓ ✓ |
| R50-DINO | 0.342 | +0.0072 [−0.002, 0.015] | +0.0054 [−0.003, 0.012] | 0.21, 0.23 ✗ |
| ViT-DINO | 0.335 | −0.0002 [−0.005, 0.005] | −0.0020 [−0.006, 0.001] | 1.0 ✗ |

References: location prior 0.335, pixelstats 0.337, random encoder 0.332, **real-image segmenter 0.608**.

- **A dedicated localisation attack finds the panels' outline essentially no better than the position-only guess.**
  Two of four encoders show a statistically significant gain, but of under 0.01 IoU, which is practically negligible.
- **ViT-DINO**, the strongest encoder for presence, gives no location information at all, matching Phase 4b.
- The same model still *detects* panels well (AUC 0.87–0.93; Phase 4b's direct classifiers get 0.90–0.95).

### 6.2 Secondary results (exploratory)

- **Compression changes nothing for location.** IoU is 0.324–0.347 at every level (pca256, pca64, pca16), within noise of the
  prior. Detection AUC falls from 0.88–0.93 at full size to 0.84–0.88 at 16 dims.
- **A dedicated attack beats general reconstruction.** Mask IoU of the localiser minus the Phase 5 GAN is **+0.09 to +0.11**
  at full size and **+0.14 to +0.21** at 16 dims (all p = 0.001). Training an attack for the target is far better than
  rebuilding the image and segmenting it.
- The random encoder scores 0.332, with no gain over the prior (p = 0.71).

---

## What Phases 4b–6 add up to

1. **Presence leaks** from every audited embedding, beyond what pixel statistics carry, and ViTs leak more than CNNs
   (Phase 4b, replicated on 3 splits for the MoCo pair).
2. **Location does not leak** from any audited encoder to any attack we built (Phase 7's own encoder is an exception, 7.3): the grid classifiers (4b), the GAN (5) and the transformer localiser (6)
   all land at or below the position-only prior.
3. **Attack strength is not the bottleneck.** Stronger attackers add ≤ 0.005 AUC, and a CNN on the raw pixels reaches 0.970
   against 0.948 for the best embedding. Where the signal exists the attacks find it; where it doesn't, none can.
4. **Neither PCA nor quantisation is a defence:** utility drops at least as fast as presence leakage.
5. **Pixel metrics are not a reliable measure of leakage** (RQ3): LPIPS does not track it, and SSIM only tracks the
   reconstruction-based measures.

**Limits to state in the paper:** all numbers are lower bounds (a better, unknown attack could extract more); one dataset pair
and one object class; the R18-MoCo incremental effect and the DINO architecture gap are fragile across seeds; Phase 5 and 6 use
a single split seed and a single training run per input.

---

## Phase 7: Own encoder, baseline and defended

**Setup** (pre-registered 2026-11-12, before either encoder was trained; results/preregistration.md, "Phase 7 defence
settings"): a ViT-S/8 MAE (25.3M parameters) trained 40 epochs on EuroSAT + GloSoFarID train tiles (split seed 0), with a
land-cover head (weight 0.1) and a solar-presence adversary behind a gradient-reversal layer. **Baseline λ = 0; defended
λ = 0.5**, ramped linearly over 4 epochs; no spatial adversary; one untuned λ. Both encoders then went through the same
pipeline as the audited ones: compression → Phase 4b attackers on 5 tasks → Phase 6 localiser → Phase 5 GAN inverter. Every
attack is **fresh**: trained after the encoder was frozen, never seen during encoder training.

### 7.1 Training (sanity check only, not a result)

| Encoder | MAE loss (first → last) | Utility CE (train) | In-training adversary: BCE / accuracy (last epoch) | Time |
|---|---|---|---|---|
| Baseline (λ = 0) | 0.722 → 0.174 | 0.0001 | 0.157 / 0.956 | 5.1 h |
| Defended (λ = 0.5) | 0.728 → 0.216 | 0.015 | 0.875 / 0.691 | 5.7 h |

The defence fooled its own adversary (accuracy ≈ 0.50 from epoch 4 to about epoch 35, recovering to 0.69 at the end) at the
price of worse reconstruction (MAE 0.216 vs 0.174).

### 7.2 Pre-registered primary tests (full size, Holm over 5)

| Test | Baseline | Defended | Difference [95% CI] | Holm p | Result |
|---|---|---|---|---|---|
| P1 leakage AUC, Phase 4b best attacker (deep MLP) | 0.946 | 0.928 | **−0.018** [−0.028, −0.010] | 0.005 | ✓ reduced |
| P2 detection AUC, Phase 6 localiser | 0.926 | 0.901 | **−0.025** [−0.036, −0.014] | 0.005 | ✓ reduced |
| P3 detection AUC, Phase 5 GAN reconstructions | 0.824 | 0.776 | **−0.048** [−0.058, −0.039] | 0.005 | ✓ reduced |
| P4 BigEarthNet macro AP, non-inferior within 0.02 | 0.629 | 0.601 | −0.028 [−0.047, −0.016] | 0.91 | ✗ |
| P5 EuroSAT accuracy, non-inferior within 0.01 | 0.978 | 0.968 | −0.010 [−0.014, −0.005] | 0.89 | ✗ |

**Verdict (pre-registered rule): "leakage cut at a cost".** All three leakage measures fall significantly, but neither
utility measure passes non-inferiority. The BigEarthNet loss is clearly beyond the 0.02 margin; the EuroSAT loss sits right at
the 0.01 margin, with a CI that crosses it.

### 7.3 Secondary results (exploratory)

**The defence is small at full size but large after compression** (leakage AUC, best attacker; utility point estimates):

| Size | Leakage: baseline → defended | Change [95% CI] | BigEarthNet AP change | EuroSAT acc. change |
|---|---|---|---|---|
| full (384) | 0.946 → 0.928 | −0.018 [−0.028, −0.010] | −0.028 | −0.010 |
| pca256 | 0.943 → 0.922 | −0.021 [−0.034, −0.009] | +0.006 | −0.015 |
| pca64 | 0.930 → 0.892 | −0.038 [−0.049, −0.029] | −0.004 | −0.010 |
| pca16 | 0.884 → 0.760 | **−0.124** [−0.142, −0.105] | −0.032 | −0.007 |

- **The defence and compression work together.** Neither does much alone: PCA was not a defence in Phase 4b, and the defence
  cuts only 0.018 at full size. Combined at 16 dims, leakage falls by 0.12 AUC while EuroSAT accuracy loses 0.007. The
  defended encoder seems to push solar information out of the leading principal components, where PCA-16 discards it.
- **But a stronger attacker gets it back.** If the attacker has the raw pixel statistics as well as the embedding (the
  incremental task), defended pca16 leakage is 0.921 against 0.917 for the baseline. The information removed from the
  embedding is still trivially available from the pixels: defended pca16 (0.760) leaks *less* than pixelstats alone (0.910).
- **The defended embedding still leaks beyond pixel statistics at full size:** +0.018 AUC over pixelstats [0.008, 0.027], down
  from +0.036 [0.028, 0.045] for the baseline.
- **The in-training adversary is fooled; fresh attackers are not.** The adversary is at chance for most of training, yet a
  fresh deep MLP still reaches 0.928 AUC on the full embedding. This is the known weakness of adversarial removal (fresh
  probes recover "removed" attributes), here confirmed for EO embeddings.
- **Location:** the baseline localiser beats the location prior by **+0.054 IoU** [0.046, 0.063] (0.390 vs 0.335), unlike any
  audited encoder (≤ +0.009, Phase 6). The defence removes this gain: defended 0.343, not distinguishable from the prior
  (+0.008 [−0.006, 0.019], p = 0.22), although it had no spatial adversary. Spatial-grid macro AP also falls (0.924 → 0.892).
- **GAN inversion:** defended reconstructions are worse on every metric (L1 0.174 → 0.196, LPIPS 0.291 → 0.331, spectral angle
  3.57° → 3.93°, mask IoU 0.331 → 0.255; all p = 0.001). SSIM drops only by 0.002.

**Own baseline compared with the audited encoders** (full size; for context, not a pre-registered test):

| | Own baseline | ViT-MoCo | ViT-DINO | pixelstats |
|---|---|---|---|---|
| Leakage AUC | 0.946 | 0.932 | 0.948 | 0.910 |
| BigEarthNet macro AP | 0.629 | 0.659 | 0.680 | 0.636 |
| EuroSAT accuracy | 0.978* | 0.964 | 0.967 | 0.951 |
| Localiser IoU (prior 0.335) | 0.390 | 0.342 | 0.335 | 0.337 |

\* The own encoders were trained with EuroSAT labels, so their EuroSAT accuracy is not comparable with the audited encoders.
BigEarthNet is the fair utility comparison: there the own baseline is below both ViTs and slightly below pixelstats.
The own encoders were also trained on GloSoFarID train tiles with 8-pixel patches (audited ViTs: 16), which probably explains
why the own baseline is the only encoder whose embedding carries some location.

### What Phase 7 adds

1. **A standard adversarial defence significantly reduces leakage against fresh attackers, but only slightly** at full size
   (−0.018 AUC), and it costs utility beyond the pre-registered margins.
2. **Defence + compression is the useful combination:** −0.12 AUC at 16 dims for −0.007 EuroSAT accuracy (−0.032 BigEarthNet AP).
   This is exploratory, so it should be called a hypothesis for future work, not a finding.
3. **Fooling the in-training adversary is not evidence of a working defence.** Evaluation must use fresh attackers.

**Phase 7 limits:** one λ (untuned), one seed, one split, one object class and one dataset pair; the own encoders saw
GloSoFarID and EuroSAT train tiles, so they are not general-purpose foundation models; the compression and location results are
exploratory.

---

## Phase 8: SatSplit extensions (compute, split points, ISLs)

Run 18 Nov 2026: 8a 18:14–18:24, 8b 18:38–21:24, 8c 21:24 (`results/phase8/`). 8b was pre-registered the same day
(`results/preregistration.md`, "Phase 8b split-point settings"). Splits are layer1–4 for ResNets, and blocks 3/6/9/12 for ViTs.
`gap` is the pooled vector at the split. `map4` is the spatial map (4×4-pooled for ResNets, patch tokens for ViTs).
All 8b numbers come from deep-MLP attackers whose hyperparameters were picked on validation; the scores are test scores.

### 8a. Compute and payload size per split (descriptive; `results/phase8/compute.csv`)

| Arch | Split | Head share of FLOPs | Full map, int8 | vs raw tile (106,496 B, fp32 map) | Head latency, MPS batch 1 |
|---|---|---|---|---|---|
| ResNet18 (0.36 GFLOPs) | layer1 → layer4 | 44% → 100% | 16,384 → 2,048 B | 0.62× → 0.08× | 1.0 → 3.0 ms |
| ResNet50 (0.73 GFLOPs) | layer1 → layer4 | 26% → 100% | 65,536 → 8,192 B | **2.46×** → 0.31× | 1.4 → 7.0 ms |
| ViT-S/16 (0.77 GFLOPs) | block3 → block12 | 29% → 100% | 6,528 B at every split | 0.25× | 8.3 → 10.2 ms |
| own ViT-S/8 (2.83 GFLOPs) | block3 → block12 | 26% → 100% | 24,576 B at every split | 0.92× | 1.4 → 3.7 ms |

- zlib already compresses the raw tile to 36,013 B (0.34×). So early ResNet maps in fp32 are **larger** than the compressed image.
- ResNet maps shrink with depth. ViT token maps stay the same size at every split, so splitting a ViT early saves compute but no bytes.
- Every check passed:
  - The FLOP counter reproduces ResNet-18 at 224 (3.63 GFLOPs).
  - head + tail = total FLOPs and parameters at every split.
  - The stages reproduce each encoder exactly.

### 8b.1 Pre-registered primary tests (Holm over 8, 1,000 resamples, one-sided)

| Test | Encoder | Measure | Final | Earliest | Diff [95% CI] | p_holm |
|---|---|---|---|---|---|---|
| P8b.1 | R18-MoCo | presence AUC, `gap`, final − earliest | 0.904 | 0.902 | +0.002 [−0.009, +0.014] | 0.586 |
| P8b.2 | ViT-MoCo | 〃 | 0.929 | 0.924 | +0.005 [−0.003, +0.013] | 0.385 |
| P8b.3 | R50-DINO | 〃 | 0.926 | 0.924 | +0.002 [−0.006, +0.010] | 0.586 |
| P8b.4 | ViT-DINO | 〃 | 0.944 | 0.929 | **+0.016** [+0.009, +0.022] | **0.008** |
| P8b.5 | R18-MoCo | spatial macro AP, `map4`, earliest − final | 0.896 | 0.947 | **+0.052** [+0.042, +0.063] | **0.008** |
| P8b.6 | ViT-MoCo | 〃 | 0.933 | 0.936 | +0.003 [−0.001, +0.007] | 0.385 |
| P8b.7 | R50-DINO | 〃 | 0.929 | 0.952 | **+0.023** [+0.017, +0.031] | **0.008** |
| P8b.8 | ViT-DINO | 〃 | 0.943 | 0.946 | +0.003 [−0.001, +0.007] | 0.385 |

**Verdicts (pre-registered rule):**
- "Depth adds presence leakage": **not supported (1/4)**.
- "Early splits leak location": **mixed (2/4)**. It holds for both ResNets and for neither ViT.

### 8b.2 Secondary results (exploratory, 300 resamples; `results/phase8/analysis/`)

Test scores by split, from earliest to final:

| Encoder | Presence AUC (`gap`) | BigEarthNet AP (`gap`) | EuroSAT acc (`gap`) | Spatial AP (`map4`) |
|---|---|---|---|---|
| R18-MoCo | 0.902 / 0.922 / 0.918 / 0.904 | 0.627 / 0.649 / 0.628 / 0.617 | 0.962 / 0.972 / 0.977 / 0.965 | 0.947 / 0.936 / 0.933 / 0.896 |
| ViT-MoCo | 0.924 / 0.930 / 0.926 / 0.929 | 0.626 / 0.643 / 0.657 / 0.658 | 0.953 / 0.961 / 0.966 / 0.964 | 0.936 / 0.934 / 0.935 / 0.933 |
| R50-DINO | 0.924 / 0.938 / 0.939 / 0.926 | 0.644 / 0.656 / 0.664 / 0.666 | 0.980 / 0.983 / 0.982 / 0.974 | 0.952 / 0.946 / 0.934 / 0.929 |
| ViT-DINO | 0.929 / 0.941 / 0.939 / 0.944 | 0.638 / 0.658 / 0.665 / 0.680 | 0.960 / 0.965 / 0.966 / 0.967 | 0.946 / 0.948 / 0.947 / 0.943 |
| random R18 | 0.884 / 0.893 / 0.868 / 0.829 | 0.606 / 0.604 / 0.576 / 0.538 | 0.948 / 0.939 / 0.910 / 0.822 | 0.936 / 0.923 / 0.911 / 0.851 |

Findings:
- **Presence leakage is set by the first stage.**
  - In the audited encoders it varies by ≤ 0.02 AUC across splits, and the middle splits are often the highest.
  - It already matches pixelstats at the earliest split; ViT-MoCo, R50-DINO and ViT-DINO are significantly above it (+0.014 to +0.019).
  - Splitting earlier never brought presence leakage below pixelstats in an audited encoder. Only the random encoder falls below it, and only at deep splits.
- **Utility mostly rises with depth.** BigEarthNet AP rises with depth for ViT-MoCo, R50-DINO and ViT-DINO (up to +0.042 for ViT-DINO); R18-MoCo peaks at layer2. So an early split gives no privacy gain and loses utility.
- **Spatial maps leak location; pooled vectors do not.**
  - Grid IoU from `map4` beats the position-only prior (0.497) at every split of every encoder:
    - earliest split: +0.15 to +0.19;
    - final split: +0.07 to +0.17.
  - The pooled final embeddings in Phase 4b scored +0.01 to +0.06.
  - The random encoder's layer1 map is almost as good (+0.149). This means the location signal is the map's spatial layout, not learned features.
  - The Phase 4b–6 statement "location does not leak" therefore holds for **pooled** payloads only.
- **The joint frontier does not narrow the choice.** Pareto over int8 bytes, head FLOPs and BigEarthNet AP leaves all 20 splits non-dominated, because bytes and FLOPs pull against each other in ResNets.

### 8c. Orbit-realistic ISLs (`results/phase8/isl_*.csv`)

Sources and settings:
- Real orbits come from CelesTrak TLEs (18 Nov 2026) propagated with skyfield over 24 h in 10 s steps.
- The line of sight must stay ≥ 100 km above the Earth.
- Workload is a 290 km swath at 6.63 km/s, giving 4,693 tiles/s, or 4.0 Gbit/s raw.

| Scenario | Link | Line of sight | Windows / day | Distance (km) |
|---|---|---|---|---|
| S1 LEO→GEO relay | Sentinel-2A/2B/2C → EDRS-C or Eutelsat 9B | 70–71% | 13–14 (≈ 78 min each) | 35,000–49,300 |
| S2 same plane (real) | Sentinel-2A ↔ 2B | 100% | continuous | 4,454–4,465 |
| S2 same plane (real) | 2A ↔ 2C, 2B ↔ 2C | 0% | none | 13,600–14,300 |
| S2 leader–follower (synthetic) | 1° / 5° / 20° apart | 100% | continuous | 125 / 625 / 2,489 |
| S3 Walker 24/3/1 (synthetic) | in-plane neighbour | 100% | continuous | 5,486 |
| S3 Walker 24/3/1 (synthetic) | adjacent plane | 33.5% | 29 (≈ 17 min each) | 898–12,308 |

What fits live, as bytes per tile at 4,693 tiles/s:

| Link class | Max bytes per tile | Raw / zlib-raw live share | Best feasible payload (BigEarthNet AP; its presence leakage) | Best feasible (EuroSAT acc; leakage) |
|---|---|---|---|---|
| S-band 2 Mbit/s | 53 B | 0.05% / 0.15% | none fits | `resnet50_dino/pca64_q4`, 32 B: 0.963; 0.925 |
| Ka-band 500 Mbit/s | 13.3 kB | 12.5% / 37% | `vits16_dino@block12` int8 map, 6,528 B: 0.680; 0.923 | R50-DINO layer2 `gap`, 2,048 B: 0.983; 0.938 |
| Optical 1.8 Gbit/s | 47.9 kB | 45% / **100%** | same, fp32 map: 0.680; 0.923 | R50-DINO layer2 int8 map: 0.983; 0.926 |
| Optical ≥ 10 Gbit/s | ≥ 266 kB | 100% / 100% | 〃 | 〃 |

Findings:
- **Embeddings are needed for bandwidth only below optical rates.**
  - At 1.8 Gbit/s (the EDRS optical class), the zlib-compressed raw tile already flows live.
  - At Ka-band, embeddings and token maps fit where raw images do not.
  - At S-band, only the 4-bit quantised PCA codes fit.
- **Every feasible high-utility payload leaks presence at AUC 0.92–0.94.** The lowest-leakage payloads are:
  - the random R18 layer4 vector (0.829), whose BigEarthNet AP is 0.538;
  - R18-MoCo pca16_q1 (0.798), whose EuroSAT accuracy is 0.892.
  
  Bandwidth limits never push the choice toward a private payload at no cost.
- **Sentinel-2's real geometry matters.** 2A–2B are about 4,460 km apart, with continuous line of sight. 2C sits opposite them in the orbit and can never link to either. A LEO→GEO relay is available about 70% of the day but adds about 120 ms one-way latency (≈ 136–146 ms against ≈ 15–25 ms for 2A↔2B).

Assumptions:
- Nominal link rates, with no link budget, pointing or acquisition time.
- Circular two-body synthetic orbits.
- No queueing or on-board storage.
- 13 bands on the 10 m grid, which overstates Sentinel-2's raw volume.

### 8d. Follow-ups (exploratory; `results/phase8/deepen/`)

Run 18 Nov 2026, 23:15–23:56. The 8d plan was written **after** the 8b results were seen
(`results/preregistration.md`, "Phase 8d follow-ups"), so 8d does not change any 8b verdict.
Every comparison is paired on the same test tiles: cluster bootstrap over patches, 1,000 resamples, Holm within each family.

**1. Replication of the 8 primary contrasts on split seeds 0, 1 and 2** (Holm over 8 within each seed):

| Contrast | R18-MoCo | ViT-MoCo | R50-DINO | ViT-DINO |
|---|---|---|---|---|
| Presence, final − earliest (`gap`) | +0.002 / −0.000 / +0.003 (0/3) | +0.005 / +0.004 / +0.013 (1/3) | +0.002 / −0.002 / +0.002 (0/3) | **+0.016 / +0.026 / +0.014 (3/3)** |
| Location, earliest − final (`map4`) | **+0.052 / +0.066 / +0.048 (3/3)** | +0.003 / +0.001 / +0.003 (0/3) | **+0.023 / +0.032 / +0.023 (3/3)** | +0.003 / +0.006 / +0.005 (0/3) |

Each effect that was significant in 8b replicates in all three seeds. Each effect that was null stays null, except ViT-MoCo
presence, which is significant once. So the 8b verdicts are robust: "not supported" and "mixed" respectively.

**2. Payload form: `map4` vs `gap` on the same tiles** (spatial task; Holm over 8):

| Encoder | Earliest split, grid IoU map / gap | Final split, grid IoU map / gap |
|---|---|---|
| R18-MoCo | 0.664 / 0.549 (+0.116) | 0.563 / 0.536 (+0.027) |
| ViT-MoCo | 0.647 / 0.562 (+0.085) | 0.650 / 0.553 (+0.098) |
| R50-DINO | 0.684 / 0.588 (+0.095) | 0.628 / 0.552 (+0.076) |
| ViT-DINO | 0.675 / 0.526 (+0.150) | 0.666 / 0.507 (+0.159) |

The pooled vectors sit at 0.51–0.59, close to the position-only prior (0.497), as in Phase 4b. The map is better in all 8
comparisons, each with p_holm 0.008, so the leak comes from the payload's **form**, not its depth. ViT-DINO shows the gap
most clearly: its pooled vector has the least location of the four encoders, while its map has among the most.

**3. Pixel control: the raw tile average-pooled to the same 4×4 grid** (13 × 16 = 208 dims; Holm over 16 audited cells per task):

The pixel map alone reaches spatial macro AP 0.932 and grid IoU 0.631, which is +0.134 over the prior. Measured against it:

| Encoder `map4` minus pixel map | Grid IoU, earliest → final split | Presence AUC, earliest → final split |
|---|---|---|
| R18-MoCo | +0.034 / +0.017 / −0.000 / **−0.068** | +0.012 / +0.008 / +0.013 / −0.001 (only layer1 significant) |
| ViT-MoCo | +0.016 / +0.025 / +0.015 / +0.020 | −0.004 / +0.009 / +0.007 / +0.002 (none significant) |
| R50-DINO | +0.053 / +0.045 / +0.025 / −0.003 | **+0.024 / +0.028 / +0.029 / +0.033** (all significant) |
| ViT-DINO | +0.044 / +0.051 / +0.046 / +0.035 | **+0.026 / +0.030 / +0.030 / +0.025** (all significant) |

Every positive IoU difference is significant (p_holm ≤ 0.036). The two near-zero cells and the R18-MoCo layer4 deficit are not
gains over the pixels. Reading the table:
- **About three-quarters of the location signal in a spatial payload is already in the pooled pixels.** Pixels give
  +0.134 IoU over the prior; the best encoder map gives +0.186.
- The encoders add at most about +0.05, mostly at early ResNet splits and at every ViT-DINO split. At the final ResNet splits
  they add nothing, or remove some.
- For presence, the DINO maps carry learned signal beyond the pixel map (+0.024 to +0.033) at every split, while the MoCo maps
  carry almost none. The pixel map's presence AUC (0.898) is just below pixelstats (0.91).
- Controls: the random encoder's map falls from +0.015 to −0.104 IoU relative to the pixels as depth grows. That is the expected
  decay for untrained features.

### What Phase 8 adds

1. **RQ4, the split point does not control presence leakage.** It is near its maximum after the first stage, so moving the
   split earlier saves bytes only in ResNets, saves compute, and loses utility. It gives no privacy.
2. **Spatial payloads leak location; pooled ones do not** (8d: +0.03 to +0.16 IoU on the same tiles, 8/8). Most of that
   signal is already in the pixels pooled to the same grid, so a spatial payload leaks location mainly because it keeps the
   image layout. The encoders add at most about +0.05. The payload *form* matters more for location than the split point.
3. **The significant 8b effects replicate in 3 of 3 split seeds** (8d), and the null effects stay null in at least 2 of 3 seeds.
4. **RQ5: real ISLs are fast enough for embeddings, and at optical rates even for compressed raw tiles.** None of the feasible
   high-utility payloads is private.

**Phase 8 limits:**
- One attacker family (deep MLP). The primary contrasts were replicated on 3 split seeds in 8d; the other cells use seed 0 only.
- The 8d follow-ups were planned after 8b, so they are exploratory.
- `map4` is pooled, so the full maps could leak more.
- The ResNet50 layer3 map was reduced by PCA to 2,048 dims.
- Latency was measured on an M-series laptop, not flight hardware.
- 8c is a rate-and-geometry model, not a link budget.
