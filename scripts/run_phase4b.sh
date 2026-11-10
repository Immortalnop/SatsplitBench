#!/usr/bin/env bash
# Phase 4b, in pre-registered build order. Every step resumes where it stopped if re-run.
# Run from the project root:   bash scripts/run_phase4b.sh 2>&1 | tee results/phase4b/run.log
# Rough timings on an Apple-silicon Mac with 16 GB (measured on the largest embedding; smaller ones are faster):
#   step 1 ~30 min, 2 ~1 h, 3 ~2 min, 4 ~25 min, 5 ~10-14 h, 6 ~45 min, 7 ~2.5 h, 8 ~5 min
set -euo pipefail
cd "$(dirname "$0")/.."
source eoenv/bin/activate
mkdir -p results/phase4b

echo "== 1. Phase 4 probes again, saving per-tile predictions and validation scores"
python scripts/probes.py --out results/probes_v2.csv --preds-dir results/preds/phase4

echo "== 2. Coarser quantisation: probes on q8/q4/q2/q1 (variants built by scripts/quantise.py)"
python scripts/quantise.py
python scripts/probes.py --comp-root data/quantised --out results/probes_quant.csv --preds-dir results/preds/phase4_quant

echo "== 3. Extra split seeds"
python scripts/splits.py --seeds 1 2

echo "== 4. BigEarthNet subset: embed, compress, pixel statistics"
if [ ! -f data/embeddings/vits16_dino/ben_test.npy ] || [ "$(python -c 'import numpy as np; print(len(np.load("data/bigearthnet/npy/test_y.npy")))')" -lt 10000 ]; then
  python scripts/bigearthnet.py
fi

echo "== 5. Attackers (deep MLP, LightGBM, RBF SVM) on all tasks, split seed 0"
python scripts/attack4b.py --tasks leakage --encoders pixelstats
python scripts/attack4b.py --tasks leakage leakage_incr
python scripts/attack4b.py --tasks utility_eurosat utility_ben
python scripts/attack4b.py --tasks spatial

echo "== 6. Image-level upper bound (ResNet-18 from scratch)"
if ! grep -q "image_resnet18" results/phase4b/runs.csv 2>/dev/null; then
  python scripts/upperbound.py
fi

echo "== 7. Robustness: primary comparisons on split seeds 1 and 2 (full size only)"
for s in 1 2; do
  python scripts/attack4b.py --tasks leakage --encoders pixelstats --split-seed "$s"
  python scripts/attack4b.py --tasks leakage leakage_incr --encoders resnet18_moco vits16_moco resnet50_dino vits16_dino \
    --variants full --split-seed "$s"
done

echo "== 8. Analysis: pre-registered tests"
python scripts/analysis4b.py
echo "== Phase 4b done. Results in results/phase4b/analysis/"
