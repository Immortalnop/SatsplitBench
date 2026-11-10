#!/usr/bin/env bash
# Phase 5: inversion attacks. Run from the project root, after Phase 4b has finished
# (both use the GPU, and RQ3 reads Phase 4b results):
#   mkdir -p results/phase5 && bash scripts/run_phase5.sh 2>&1 | tee results/phase5/run.log
# Measured: segmenter ~2 min/epoch (40 max); kNN a few min per run; GAN ~3.7 min per warm-up epoch and
# ~5.1 min per adversarial epoch (20 max, early stopping) = ~1-1.7 h per GAN, 10 GANs. Total ~14-20 h.
# Completed runs are skipped when re-run (checked by their metrics.json).
#
# Scope: the weak inverter (kNN) runs on every encoder at full size and 16 dims. The strong inverter (GAN)
# trains one model per input, so it runs on the 4 audited encoders at full size and 16 dims (the two ends
# of the compression curve), plus two controls: pixel statistics (learned leakage = embedding minus this)
# and the random-weight ResNet18.
set -euo pipefail
cd "$(dirname "$0")/.."
source eoenv/bin/activate
mkdir -p results/phase5

done_run() { [ -f "results/phase5/$1/$2/$3/metrics.json" ]; }

echo "== 1. Panel segmenter on real tiles"
[ -f results/models/segmenter_unet.pt ] || python scripts/segmenter.py

echo "== 2. Mean-tile baseline"
done_run mean baseline mean || python scripts/invert.py --method mean

echo "== 3. kNN retrieval inverter"
for enc in resnet18_moco vits16_moco resnet50_dino vits16_dino resnet18_random; do
  for var in full pca16; do
    done_run knn "$enc" "$var" || python scripts/invert.py --method knn --encoder "$enc" --variant "$var"
  done
done
done_run knn pixelstats full || python scripts/invert.py --method knn --encoder pixelstats --variant full

echo "== 4. GAN inverter"
done_run gan pixelstats full || python scripts/invert.py --method gan --encoder pixelstats --variant full
for enc in resnet18_moco vits16_moco resnet50_dino vits16_dino; do
  for var in full pca16; do
    done_run gan "$enc" "$var" || python scripts/invert.py --method gan --encoder "$enc" --variant "$var"
  done
done
done_run gan resnet18_random full || python scripts/invert.py --method gan --encoder resnet18_random --variant full

echo "== 5. Analysis: pre-registered Phase 5 tests"
python scripts/analysis5.py
echo "== Phase 5 done. Results in results/phase5/analysis/"
