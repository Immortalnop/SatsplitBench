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
