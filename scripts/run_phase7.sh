#!/usr/bin/env bash
# Phase 7: own encoder (ViT-Small/8, MAE) - baseline, then defended. Run from the project root, after Phase 6:
#   mkdir -p results/phase7 && caffeinate -i bash scripts/run_phase7.sh baseline 2>&1 | tee results/phase7/baseline.log
#   caffeinate -i bash scripts/run_phase7.sh defended 2>&1 | tee results/phase7/defended.log
# The defended run uses the lambda pre-registered in results/preregistration.md ("Phase 7 defence settings":
# lambda = 0.5, no spatial adversary). Any other LAMBDA is refused, so the pre-registered setting cannot be
# changed by accident; the script also refuses to start if that section is missing. After the defended run it
# writes the Phase 7 analysis (scripts/analysis7.py).
#
# Each encoder then goes through exactly the pipeline the audited encoders went through:
# compress (PCA 16/64/256 + int8) -> quantise -> BigEarthNet compression -> Phase 4b attackers on every task
# -> Phase 6 localisation attack -> Phase 5 GAN inverter. All attackers are trained fresh on its embeddings.
# Measured: ~5-7 h encoder training (40 epochs), then roughly 6-10 h of evaluation per encoder.
set -euo pipefail
cd "$(dirname "$0")/.."
source eoenv/bin/activate
mkdir -p results/phase7

stage="${1:?usage: run_phase7.sh baseline|defended}"
if [ "$stage" = "baseline" ]; then
  name=own_vit_baseline; lam=0
elif [ "$stage" = "defended" ]; then
  name=own_vit_defended; lam="${LAMBDA:-0.5}"
  [ "$lam" = "0.5" ] || { echo "LAMBDA=$lam differs from the pre-registered 0.5; refusing (see results/preregistration.md)."; exit 1; }
  grep -q "Phase 7 defence settings" results/preregistration.md || {
    echo "Record the defence settings in results/preregistration.md (section 'Phase 7 defence settings') first."; exit 1; }
else
  echo "unknown stage $stage"; exit 1
fi

echo "== 1. Train $name (adv-weight $lam) and export embeddings incl. BigEarthNet"
if [ -f "results/models/$name.pt" ]; then
  [ -f "data/embeddings/$name/ben_test.npy" ] || python scripts/own_encoder.py --name "$name" --adv-weight "$lam" --export-only --ben
else
  python scripts/own_encoder.py --name "$name" --adv-weight "$lam" --ben
fi

echo "== 2. Compression exactly as Phase 3 / 4b"
[ -d "data/compressed/$name/pca16" ] || python scripts/compress.py --encoders "$name"
python scripts/quantise.py --encoders "$name"
python scripts/bigearthnet.py --compress-only --encoders "$name"

echo "== 3. Phase 4b attackers (deep MLP, LightGBM, RBF SVM) on every task"
python scripts/attack4b.py --tasks leakage leakage_incr utility_eurosat utility_ben spatial --encoders "$name"

echo "== 4. Phase 6 localisation attack (full size)"
[ -f "results/phase6/$name/full/metrics.json" ] || python scripts/localise.py --encoder "$name" --variant full

echo "== 5. Phase 5 GAN inverter (full size)"
[ -f "results/phase5/gan/$name/full/metrics.json" ] || python scripts/invert.py --method gan --encoder "$name" --variant full

echo "== $name done"
if [ "$stage" = "defended" ]; then
  echo "== 6. Phase 7 analysis: baseline vs defended (pre-registered tests)"
  python scripts/analysis7.py
  echo "== Phase 7 done. Results in results/phase7/analysis/"
fi
