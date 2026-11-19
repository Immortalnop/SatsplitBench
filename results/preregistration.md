# Phase 4b pre-registration

Written and approved on 2026-11-10, before any Phase 4b analysis was run. Every analysis below is reported whatever it shows.

## Disclosure

- Phase 4 results (`results/probes.csv`) were seen before this document was written. They are exploratory.
- In particular, the ResNet18 MoCo vs ViT-S/16 DINO leakage gap had already been noticed. The H2 pairs below were chosen by design (same pretraining method, different architecture), not that observed pair, but they were fixed after Phase 4.
- Known before writing: panels concentrate at patch centres (P(tile has panels) 0.42 for the 4 centre tiles vs 0.11–0.12 for the corners); tile position alone predicts "has panels" with test AUC 0.659.

## Leakage estimate

For each encoder and variant, the leakage estimate is the test score of the attacker with the best **validation** score. Every attacker's test score is reported.

> Amendment, 2026-11-10, before any Phase 4b analysis ran: the first version said "the best test score among the attackers". Selecting on test inflates the estimate, so selection is on validation. Attackers: logistic regression and one-hidden-layer MLP (Phase 4), deep residual MLP, LightGBM, RBF-kernel SVM with the Nyström approximation (Phase 4b). Hyperparameters are chosen on val only.

## Primary comparisons

Corrected for multiple testing with Holm's method across all primary tests, at α = 0.05.

1. **Incremental (learned) leakage**, per audited encoder at full size (4 tests): test AUC of the best attacker on pixel statistics + embedding, minus test AUC of the best attacker on pixel statistics alone. Paired bootstrap over test patches, 1,000 resamples, 95% CI. Hypothesis: the difference is above 0.
2. **H2 architecture pairs**, at full size (2 tests): ResNet18 MoCo vs ViT-S/16 MoCo, and ResNet50 DINO vs ViT-S/16 DINO. Leakage difference with a paired bootstrap over test patches. Utility for each pair is reported on BigEarthNet (19 labels, macro mAP) and EuroSAT.

## Secondary analyses

Reported in full, uncorrected, labelled exploratory.

- Leakage within each land-cover class (GloSoFarID tiles pseudo-labelled by the best EuroSAT utility attacker).
- Spatial probe: 4×4 grid localisation inside a tile, scored by grid IoU against the average mask for that tile position (a deliberately strict baseline, since attackers do not see tile position).
- Image-level upper bound: ResNet-18 trained from scratch on raw tiles.
- Coarser quantisation: int4, int2 and 1-bit, on full and PCA variants.
- Robustness: the primary comparisons repeated on 2 further train/val/test split seeds (3 in total).
- Matched-utility comparison for H2 on each encoder's trade-off curve.

## Scope

- New attackers run on float variants: full, pca256, pca64, pca16. int8 matched float everywhere in Phase 4; quantisation is covered by the coarser-quantisation analysis.
- Phase 4 probes are re-run with per-tile test predictions saved, so they can enter paired tests. Same seeds and splits, so their numbers should reproduce.

## Build order

1. Save Phase 4 predictions (re-run `probes.py`).
2. Coarser quantisation (extend `compress.py`).
3. Shared attacker module (deep MLP, LightGBM, SVM, hyperparameter search).
4. Incremental and per-class leakage, paired tests.
5. Spatial probe.
6. BigEarthNet subset: embed, compress, utility.
7. ResNet-18 image-level upper bound.
8. Split seeds.

---

# Phase 5 pre-registration (inversion)

Written on 2026-11-10, before any Phase 5 model was trained beyond 1–2-epoch code tests, whose numbers are not results and are not used.

## Measures

The frozen U-Net panel segmenter, trained on real tiles with its threshold chosen on val, is run on every reconstruction:
- **Detection AUC:** maximum segmenter probability per tile vs "has panels", on all test tiles.
- **Mask IoU:** on positive test tiles, at the segmenter's val threshold.

Pixel metrics: L1, SSIM (13 bands), LPIPS (RGB), spectral angle.

## Primary comparisons

Holm-corrected across these 8 tests, at α = 0.05, one-sided (greater), paired cluster bootstrap over test patches, 1,000 resamples:

1. **Learned leakage through inversion** (4 tests): for each audited encoder at full size, detection AUC on GAN reconstructions from the embedding, minus detection AUC on GAN reconstructions from pixel statistics.
2. **Spatial leakage through inversion** (4 tests): for each audited encoder at full size, mask IoU on GAN reconstructions, minus mask IoU of the location prior (mean train mask for that tile position, threshold chosen on val).

## Secondary analyses

Uncorrected, labelled exploratory:
- The same two comparisons at 16 dims, for the k-NN inverter, and for the random-ResNet18 control.
- Strong vs weak inverter: GAN minus k-NN, per encoder and variant, on detection AUC and mask IoU.
- **RQ3:** across the inverted encoder/variant pairs, Kendall's τ between each pixel metric (SSIM, LPIPS) and each object measure (detection AUC, mask IoU, and the Phase 4b leakage AUC for the same encoder and variant). Exact permutation p-value. H3 predicts that pixel metrics rank encoders differently from object measures.
- Each measure against the mean-tile baseline.

---

# Phase 6 pre-registration (localisation attack)

Written on 2026-11-10, before any Phase 6 model was trained. Code tests use a few epochs on small subsets; their numbers are not results and are not used.

## Model and measures

A query-based transformer decoder maps an embedding to a 64×64 panel mask. It is trained on all train tiles (Dice + focal loss) and selected on val mean IoU over positive tiles; the decision threshold is chosen on val.
- **Mask IoU:** on positive test tiles.
- **Detection AUC:** maximum predicted probability per tile vs "has panels", on all test tiles, comparable with Phase 5.

Baselines:
- **Location prior:** the mean train mask for the tile's position, threshold chosen on val. Strict, because the attacker never sees tile position.
- **Pixel-statistics control:** the same model given pixel statistics instead of an embedding.

## Primary comparisons

Holm-corrected across these 8 tests, at α = 0.05, one-sided (greater), paired cluster bootstrap over test patches, 1,000 resamples, full-size embeddings:

1. **Spatial leakage** (4 tests): mask IoU of each audited encoder minus mask IoU of the location prior.
2. **Learned spatial leakage** (4 tests): mask IoU of each audited encoder minus mask IoU of the pixel-statistics control.

## Secondary analyses

Uncorrected, labelled exploratory:
- The same comparisons at pca256, pca64 and pca16, and for the random-ResNet18 control: spatial leakage along the compression curve.
- Localisation attack vs the Phase 5 GAN inverter, mask IoU on the same positive test tiles (does a dedicated attacker beat general reconstruction?).
- Detection AUC vs the Phase 4b best attacker for the same encoder and variant.

---

# Phase 7 defence settings (pre-registration)

Written on 2026-11-12, after the Phase 4b, 5 and 6 results were seen and **before any Phase 7 encoder was trained**. The only Phase 7 runs so far are 2-epoch code tests (a few training steps; their numbers are not results and are not used). The settings below apply to both the baseline and the defended encoder, so the two are identical except for λ.

## What the earlier phases say about the design

- **Presence leaks mostly through spectral content.** Pixel statistics alone reach AUC 0.910 (Phase 4b). The audited embeddings add +0.016 to +0.039 beyond that (4 incremental tests, Holm-significant). A defence cannot realistically go much below the pixel-statistics level without destroying land-cover utility, so the realistic target is to remove the *learned excess*, not to reach AUC 0.5.
- **Location barely leaks.** The Phase 6 localisation attack reaches mask IoU 0.335–0.344 at full size against 0.335 for the position-only prior (significant for 2 of 4 encoders, with gains under 0.01 IoU), and every Phase 5 GAN reconstruction scores below the prior. There is almost no location information to remove, so **no spatial adversary is added**. (This changes the earlier tentative plan, written after Phase 4b only, to consider one; Phase 6 removed its motivation.)
- **Compression and quantisation are not defences** (Phase 4b): utility falls at least as fast as leakage, so the defence is trained into the encoder rather than applied afterwards.

## Settings

| Setting | Baseline | Defended |
|---|---|---|
| Encoder, data, optimiser, epochs, seed | ViT-S/8 MAE, 40 epochs, seed 0, as in `scripts/own_encoder.py` | identical |
| Adversary | solar-farm presence MLP, trained every step, **zero gradient into the encoder** (λ = 0) | same MLP behind a gradient-reversal layer |
| **λ (gradient-reversal strength)** | **0** | **0.5** |
| λ schedule | 0 | linear ramp from 0 to 0.5 over the first 4 epochs (standard DANN schedule); identical code path, so λ = 0 stays exactly 0 |
| Utility head weight | 0.1 | 0.1 |
| Spatial adversary | none | **none** |

- **λ = 0.5 is a single, untuned choice.** The deadline allows one defended training run (about 5–7 h of training plus 6–10 h of evaluation), so there is no sweep. 0.5 is the middle of the usual gradient-reversal range (0.1–1). In the early code tests the adversary's loss (about 1.07) and the MAE loss (about 1.1–1.2) were of the same size, so a weight of 0.5 pushes the encoder hard enough to matter without swamping the reconstruction objective. If λ = 0.5 turns out too weak or too strong, that is reported as such; λ is **not** re-chosen after seeing results, and any additional λ is labelled exploratory.
- **Known limitation.** Reversing the adversary's gradient maximises its loss and does not force it to chance. The encoder can make its own adversary confidently *wrong* without removing information. This is why the defence is judged only against attackers trained fresh on its embeddings (Phases 4b, 5, 6), never by its own adversary's accuracy. The adversary's training accuracy is reported as a diagnostic only.

## Evaluation

Baseline and defended encoders go through exactly the audited encoders' pipeline (compress, quantise, BigEarthNet, all Phase 4b attackers on every task, the Phase 6 localisation attack and the Phase 5 GAN inverter at full size). The analysis is `scripts/analysis7.py`. Selection of the attacker is on validation among the three Phase 4b attackers only, for both encoders (Phase 4's probes are not run on the own encoders).

### Primary tests

Holm-corrected across these 5 tests, at α = 0.05, paired cluster bootstrap over test patches (leakage) or images (utility), 1,000 resamples, full-size embeddings, split seed 0:

| # | Test | Hypothesis |
|---|---|---|
| P1 | Leakage AUC, Phase 4b best attacker | baseline − defended > 0 |
| P2 | Detection AUC of the Phase 6 localisation attack | baseline − defended > 0 |
| P3 | Detection AUC of the Phase 5 GAN reconstructions | baseline − defended > 0 |
| P4 | BigEarthNet macro AP (non-inferiority) | defended − baseline > −0.02 |
| P5 | EuroSAT accuracy (non-inferiority) | defended − baseline > −0.01 |

The margins (0.02 macro AP, 0.01 accuracy) were fixed here, before any result. They are small against the differences between the audited encoders (0.04–0.07 macro AP).

### Verdict rule

- **Works:** P1 is significant and P4 and P5 are significant (leakage cut, utility within the margins).
- **Leakage cut at a cost:** P1 is significant but P4 or P5 is not.
- **No reduction:** P1 is not significant.

All three outcomes are reported as findings. A "no reduction" result is a legitimate answer to whether an adversarial penalty on a single presence label removes leakage that fresh attackers can still recover.

### Secondary analyses

Uncorrected, labelled exploratory:
- Leakage and incremental leakage of each encoder against pixel statistics (is the learned excess removed?).
- The compression curve (pca256, pca64, pca16): leakage with paired CIs, utility and spatial macro AP as point estimates.
- Localisation mask IoU and GAN reconstruction metrics (L1, SSIM, LPIPS, spectral angle, mask IoU), baseline vs defended and each against the location prior.
- The own baseline against the audited ViT-S/16 encoders at full size (is the own encoder a reasonable encoder?).
- Training diagnostics: last-epoch losses and the adversary's accuracy for both runs.

### Scope

- A single seed and split for each encoder (cost). Differences are paired over test patches, so they carry the sampling uncertainty of the test set but not the variation between training runs; the paper says so.
- No claim is made beyond the one defence setting, the one label (solar-farm presence) and the one dataset pair.

# Phase 8b split-point settings (pre-registration)

Written on 2026-11-18, **before any Phase 8b attacker was run on real data**, after Phases 4b–7 were complete. What was seen before writing: the plan in README.md (Phase 8, written earlier the same day, with these two hypotheses); code tests on 300-row subsets, which check that the feature extraction runs and that the final split reproduces the Phase 2 embeddings exactly; and **one pipeline test on full data**, run to check that `attack4b.py` reads a split-point variant: `resnet18_moco@layer1`, `map4`, `spatial`, deep MLP, val macro AP 0.938, test 0.947. That number is disclosed here. It was produced after the hypotheses were written and did not change them. The same cell is computed again in the real run.

## Setting

A split point cuts an encoder into a head (satellite A) and a tail (satellite B or ground). The payload sent across the inter-satellite link is the head's output. Splits (`scripts/splitpoints.py`):

| Encoder | Splits (earliest → final) |
|---|---|
| resnet18_moco, resnet50_dino, resnet18_random (control) | layer1, layer2, layer3, layer4 |
| vits16_moco, vits16_dino | block3, block6, block9, block12 |

The final split (layer4 / block12) is the Phase 2–7 encoder output.

Two payload forms per split (`scripts/splits8b.py`):
- **`gap`**: global average pool (ResNets); CLS token after the final LayerNorm (ViTs). On EuroSAT, GloSoFarID and the Phase 4b BigEarthNet subset. At the final split it equals the stored Phase 2 embedding (checked).
- **`map4`**: the feature map average-pooled to 4 × 4 (ResNet layer1–2), native 4 × 4 (layer3), native 2 × 2 (layer4), or the 16 patch tokens after the final LayerNorm (ViTs), flattened. GloSoFarID only; stored as float16. **Any `map4` wider than 8,192 dims (only resnet50_dino@layer3, 16,384) is reduced by PCA to 2,048 dims, fitted on at most 20,000 randomly drawn train rows of split seed 0.** The variance kept is recorded in `meta.json`. This is a lower bound on what the full transmitted map reveals.

Attacker: the Phase 4b deep residual MLP only (same architecture and grid: lr {1e-3, 3e-4} × weight decay {1e-4, 1e-2} × dropout {0.1, 0.3}, chosen on validation; inputs standardised on train). In Phase 4b it was the chosen attacker, or within 0.005 AUC of it, in almost every cell, and it is about 3 times cheaper than running all three. Split seed 0 only. Tasks as in Phase 4b: `gap` → `leakage`, `leakage_incr`, `utility_eurosat`, `utility_ben`; `map4` → `spatial`, `leakage`.

## Primary tests

Eight tests, Holm-corrected together (α = 0.05). Paired cluster bootstrap over GloSoFarID test patches, 1,000 resamples, one-sided, using the same test rows for both splits. Earliest split = layer1 / block3; final = layer4 / block12.

| Test | Hypothesis | Measure |
|---|---|---|
| P8b.1–4 | **Depth adds presence leakage:** AUC(final) − AUC(earliest) > 0, one test per audited encoder | `leakage`, `gap` |
| P8b.5–8 | **Early splits leak location:** macro AP(earliest) − macro AP(final) > 0, one test per audited encoder | `spatial`, `map4` |

Reasoning: Phase 4b found that the leakage beyond pixel statistics is learned, so it should grow with depth. Phases 4b–6 found almost no location information in the final embeddings, so if location leaks anywhere it should be at early splits, where the maps still have spatial resolution.

### Verdict rule

For each family of four: **supported** if at least 3 are significant after Holm, **not supported** if at most 1 is, **mixed** if 2. Every test is reported whatever it shows. A test whose inputs are missing counts as p = 1 and is reported as not run.

## Secondary analyses (exploratory, no multiplicity correction; 300 bootstrap resamples)

- Leakage, spatial AP and utility curves over all four splits, per encoder, with 95% CIs.
- `gap` leakage and `leakage_incr` minus the pixel-statistics control, per split (two-sided).
- Spatial grid IoU (`map4`) minus the location prior, per split.
- The random-encoder control at every split.
- Payload bytes and head FLOPs per split (Phase 8a), the Pareto frontier over (int8 map bytes, head FLOPs, BigEarthNet AP) with the leakage at each frontier point, and the ISL feasibility of each payload (Phase 8c).
- Tier 3, if run: the Phase 6 localiser on `map4` at the earliest split of each audited encoder, against the location prior.

## Fallbacks and scope

- Run order: tier 1 (`gap` leakage + BigEarthNet utility, `map4` spatial, audited encoders), then tier 2 (the rest, plus the random control), then tier 3 (localiser, optional). If tier 1 has not finished by **2026-11-19 14:00**, the encoders that finished are reported and the rest are listed as not run.
- One attacker family, one split seed, one training run per cell. The own encoders (Phase 7) are not included. No claim is made beyond these four encoders, this object class and these datasets.

# Phase 8d follow-ups (exploratory; written 2026-11-18, after the 8b results were seen)

These analyses were chosen **after** the 8b primary results were seen, so they are exploratory, not confirmatory.
They do not change any 8b verdict.

1. **Pixel control for location.** Pseudo-encoder `pixels`/`map4` is the raw tile divided by 10,000 and average-pooled to 4×4 (13 bands × 16 = 208 dims).
   - Tasks: spatial and leakage.
   - Comparison: encoder `map4` minus the pixel map on the same test tiles (macro AP or AUC, and grid IoU).
   - Holm within each task over the 16 audited encoder-by-split cells.
2. **Payload form.** The spatial task on `gap` at the earliest and final split of each audited encoder.
   - Comparison: `map4` minus `gap` grid IoU.
   - Holm over 8.
3. **Replication.** The 8 primary 8b contrasts on split seeds 1 and 2: same attacker, same grid, one-sided, Holm over 8 within each seed. A contrast is called replicated if it is significant in at least 2 of the 3 seeds.

All tests use a paired cluster bootstrap over patches with 1,000 resamples. Script: `scripts/deepen8.py`. Run with `scripts/run_phase8.sh 8d`.
