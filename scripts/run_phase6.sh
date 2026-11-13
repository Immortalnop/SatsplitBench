#!/usr/bin/env bash
# Phase 6: localisation attack. Run from the project root, after Phase 5 (both use the GPU, and the
# secondary comparison reads the Phase 5 GAN results):
#   mkdir -p results/phase6 && caffeinate -i bash scripts/run_phase6.sh 2>&1 | tee results/phase6/run.log
# Completed runs are skipped when re-run. Measured on a 2,000-tile test: ~6 s per epoch, so roughly
# 1-2 min per epoch on the full training set and ~30-70 min per run with early stopping; 18 runs, ~12-18 h.
set -euo pipefail
cd "$(dirname "$0")/.."
source eoenv/bin/activate
mkdir -p results/phase6

run() { [ -f "results/phase6/$1/$2/metrics.json" ] || python scripts/localise.py --encoder "$1" --variant "$2"; }

echo "== 1. Pixel-statistics control"
run pixelstats full
echo "== 2. Audited encoders, full size first (primary tests), then the compression curve"
for var in full pca256 pca64 pca16; do
  for enc in resnet18_moco vits16_moco resnet50_dino vits16_dino; do
    run "$enc" "$var"
  done
done
echo "== 3. Random-weight control"
run resnet18_random full
echo "== 4. Analysis"
python scripts/analysis6.py
echo "== Phase 6 done. Results in results/phase6/analysis/"
