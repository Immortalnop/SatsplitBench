# Context handover: EO Embedding Leakage Study

Written 11 Nov 2026, 10:40 IST. A new chat should read this file first, then README.md (full phase docs)
and results/preregistration.md. **Submission deadline: 21 Nov 2026, 11:59 pm** (moved from 15 Nov).

## What the project is
University-course research project. It measures how much object-level information (solar farms) leaks from
compact Earth-observation embeddings, and how that trades off against utility under compression. The study is
framed as an **information-leakage audit, not a privacy attack** (Sentinel-2 is public, so there is no victim).
Output: a ~6-page paper. The plan doc is a Claude Docs artifact, "EO Embedding Leakage Study — Research Plan"
(doc id `be824fae-1baa-4cc1-99c3-6baae19fa06e`; the progress table still says 4b+ are "Planned").

Everything runs on the user's **MacBook (Apple Silicon, 16 GB, MPS)**, in `~/Desktop/projects/Satsplitbench`,
with the venv `eoenv` (torch 2.15, torchgeo 0.10.1, sklearn 1.9.1, timm 1.0.31, torchmetrics 1.9.0, lightgbm 4.7.0).

## Decisions made (do not reopen these)
- **No resizing anywhere.** The only exception is nearest-neighbour band copying for BigEarthNet's 20/60 m bands
  (approved). BigEarthNet B10 is zero-filled and centre-cropped 120→112.
- Torchgeo's weight transforms (Resize/CenterCrop) are not used; the only preprocessing is ÷10000. The ViTs use
  `dynamic_img_size=True` at 64×64.
- **Keep ViT encoders.** Audited encoders: `resnet18_moco`, `vits16_moco`, `resnet50_dino`, `vits16_dino`. Controls:
  `resnet18_random`, `pixelstats` (65 per-band stats) and a location prior (tile position).
- GloSoFarID is cut into 16 native 64×64 tiles per patch (55,408 tiles) and split by patch 70/15/15 (seeds 0, 1, 2).
- **"Mixed" model plan:** research-grade models in every phase, with linear probes kept as a reference row.
  Technology repeats are allowed only where they help a comparison.
- Fixed flaws: the perceptual loss uses **ResNet50 MoCo, which is not audited** (this avoids circularity), and
  utility near the ceiling was addressed by adding the **BigEarthNet 19-label** task.
- Phase order: **4b → 5 (inversion) → 6 (localisation) → 7 (own encoder: baseline, then defended)**.
- The **Phase 7 defence settings (λ, maybe a spatial adversary) are chosen only after Phase 4b's primary results**,
  and they must be written into `results/preregistration.md` under the heading "Phase 7 defence settings" before the
  defended run. `run_phase7.sh` refuses to start without that section.
- Leakage estimate = **the test score of the attacker that scored best on validation** (amended pre-registration;
  never the best test score).
- **Mac only, full plan, no Colab, no scope cuts** (user: "simplet to use mac only. I have time"). The fallback, only if
  the defended run would miss the deadline on 14 Nov: report the baseline and describe the defence as future work.
- Statistics: cluster bootstrap over patches (1,000 resamples), paired tests, Holm correction, Kendall τ with an exact
  permutation test.
- macOS specifics: `brew install libomp` is done; **lightgbm must be imported before torch** (already handled in
  attackers.py); run long jobs under `caffeinate -i`, with the lid open and on AC power.

## Status (12 Nov, evening)
| Phase | State |
|---|---|
| 0–7 | **All done (13 Nov).** Results in `RESULTS.md` and `results/phase{4b,5,6,7}/analysis/` |
| 8 | **Done (18 Nov).** Results in `RESULTS.md` and `results/phase8/` |

Phase 7 verdict: **"leakage cut at a cost"**. P1–P3 leakage reductions significant (−0.018 / −0.025 / −0.048 AUC);
BigEarthNet (−0.028) and EuroSAT (−0.010) fail non-inferiority. Exploratory: at pca16 leakage falls −0.124. Remaining work is
writing only (paper, plan doc, git commit).

Phase 8 (added 18 Nov, see README "Phase 8"): 8a compute per split, 8b split points (pre-registered 18 Nov in
`results/preregistration.md`, "Phase 8b split-point settings"; 8 primary tests), 8c orbit-realistic ISLs (CelesTrak TLEs,
skyfield). Scripts: splitpoints.py, compute8a.py, splits8b.py, analysis8b.py, isl8c.py, run_phase8.sh (all tested on
subsets). **Ran in full 18 Nov, 18:38–21:24** (log `results/phase8/run.log`; results in RESULTS.md "Phase 8").
Verdicts: depth adds presence leakage **not supported (1/4)**, with only ViT-DINO significant (+0.016); early splits leak location
**mixed (2/4)**, true for both ResNets (+0.052, +0.023) and for neither ViT. Spatial `map4` payloads beat the location prior at every split
(+0.07 to +0.19 IoU), so "location does not leak" applies to pooled payloads only. 8c: S-band fits only quantised codes;
1.8 Gbit/s optical carries even zlib raw live; every feasible high-utility payload leaks AUC 0.92–0.94.
Disclosed in the pre-registration: one pipeline test result was seen (resnet18_moco@layer1 map4 spatial, test AP 0.947).
Phase 9 = paper. Deadline 21 Nov 2026, 23:59.

Phase 7 settings (pre-registered 2026-11-12 in `results/preregistration.md`, section "Phase 7 defence settings"):
**λ = 0.5** (ramped linearly from 0 over 4 epochs; baseline λ = 0), **no spatial adversary** (Phase 6 shows almost no
location information to remove), one untuned λ because the deadline allows one defended run. Five primary tests in
`scripts/analysis7.py` (Holm over 5): leakage via Phase 4b best attacker, Phase 6 detection AUC and Phase 5 GAN detection
AUC must fall; BigEarthNet macro AP (margin 0.02) and EuroSAT accuracy (margin 0.01) must be non-inferior. Verdict rule:
works / leakage cut at a cost / no reduction. `run_phase7.sh defended` refuses any λ other than 0.5 and runs the analysis
at the end. `analysis7.py` was tested on stand-in encoders (`--baseline vits16_moco --defended vits16_dino --boot 100`).

Lessons: Phase 6 stalled for hours because of memory pressure (swap full; Pylance and other apps). Start long jobs with
`nohup ... &` from macOS Terminal, keep other apps closed. Restarting VS Code extensions killed a run once.

## Run commands (from the project root)
```bash
mkdir -p results/phase4b && caffeinate -i bash scripts/run_phase4b.sh 2>&1 | tee results/phase4b/run.log   # running
mkdir -p results/phase5 && caffeinate -i bash scripts/run_phase5.sh 2>&1 | tee results/phase5/run.log
mkdir -p results/phase6 && caffeinate -i bash scripts/run_phase6.sh 2>&1 | tee results/phase6/run.log
mkdir -p results/phase7 && caffeinate -i bash scripts/run_phase7.sh baseline 2>&1 | tee results/phase7/baseline.log
LAMBDA=<value> caffeinate -i bash scripts/run_phase7.sh defended 2>&1 | tee results/phase7/defended.log
```
All of these scripts are resumable: completed runs are skipped.

## Phase 4 key results (already in the README)
Leakage AUC on full-size embeddings (linear / MLP): ViT-DINO 0.937/0.947, R50-DINO 0.934/0.934, ViT-MoCo 0.922/0.932,
R18-MoCo 0.894/0.904, pixelstats 0.836/0.907, random 0.796/0.824. Location prior: position-only AUC 0.659.
Findings: most of the leakage is spectral. H1 holds with a qualifier; H2 is modestly supported. RQ2 is nuanced:
compression reduces linear accessibility rather than removing the leakage. int8 makes no difference.

## Phases and models
- **4b** (`attack4b.py`, `attackers.py`, `upperbound.py`, `analysis4b.py`, `quantise.py`, `splits.py`, `bigearthnet.py`):
  - Attackers: deep residual MLP, LightGBM and RBF SVM (Nyström), with hyperparameters chosen on validation.
  - Tasks: leakage, incremental leakage, spatial (16 cells), EuroSAT utility, BigEarthNet utility.
  - Also: quantile quantisation q8/q4/q2/q1, a ResNet-18 upper bound trained from scratch, and seeds 1 and 2 as replications.
  - Outputs go to `results/phase4b/analysis/`: best, primary (Holm), per_class, spatial, quantisation,
    matched_utility and upper_bound.
- **5** (`segmenter.py`, `invert.py`, `imagemodels.py`, `analysis5.py`):
  - Inverters: mean, kNN and GAN (a self-attention generator with a PatchGAN, trained on 10·L1 + perceptual + 0.1·hinge).
  - Metrics: L1, SSIM, LPIPS, spectral angle; detection AUC and mask IoU, both measured with a U-Net segmenter.
  - Tests: 8 pre-registered primary tests.
- **6** (`localise.py`, `analysis6.py`):
  - Model: a query-based transformer decoder (DETR/Mask2Former style) trained with Dice + focal loss.
  - Tests: 8 primary tests, comparing IoU against the location prior and against pixelstats.
- **7** (`own_encoder.py`, `run_phase7.sh`):
  - Model: a ViT-S/8 MAE with fixed 2D sin-cos positions, a land-cover head and an adversary behind a gradient-reversal layer.
  - Runs: λ=0 for the baseline, λ>0 for the defended encoder. Each one goes through the same compress → 4b → 6 → 5 GAN pipeline.

## Next steps for the new chat
1. Start the Phase 7 baseline (detached, see README "How to run"), then the defended run after it (both done 13 Nov).
2. While they run: help draft the paper from `RESULTS.md`; update the plan doc's progress table (Claude Docs artifact, still shows 4b+ as Planned).
3. After the defended run: read `results/phase7/analysis/` (primary.csv, README.json verdict), add a Phase 7 section to `RESULTS.md`.
4. If time runs short: report the baseline only and describe the defence as future work (Phases 4b–6 are a complete paper).
5. Optional: fix the LightGBM `eval_set` warning (harmless).

Git: branch `main`. Almost everything since "till phase 5 I believe" is uncommitted (Phase 4b–7 scripts, results, README, RESULTS.md, HANDOVER.md). `data/` and `eoenv/` are gitignored.
