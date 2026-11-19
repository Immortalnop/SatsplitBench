#!/usr/bin/env bash
# Phase 8: SatSplit extensions. Run from the project root, detached:
#   mkdir -p results/phase8 && nohup caffeinate -i bash scripts/run_phase8.sh all > results/phase8/run.log 2>&1 &
# Stages (each resumable; finished work is skipped):
#   8a   compute8a.py: FLOPs, parameters, payload bytes, head latency           (~15-25 min; needs an idle GPU)
#   8b   splits8b.py + deep-MLP attackers in tiers + analysis8b.py             (~6-9 h)
#        refuses to start unless results/preregistration.md has "Phase 8b split-point settings"
#        TIER3=1 also runs the localiser on the earliest split of each audited encoder (+2-4 h)
#   8c   isl8c.py: orbits, contact windows, ISL feasibility                    (~1 min; fetches TLEs once a day)
#   all  8a, then 8b, then 8c (8c last so its frontier includes the 8b payloads)
#   8d   deepen8.py follow-ups (exploratory, after 8b): pixel-map control, gap-vs-map location, split seeds 1-2 (~2-3 h)
set -euo pipefail
cd "$(dirname "$0")/.."
source eoenv/bin/activate
stage="${1:-all}"
mkdir -p results/phase8

AUDITED="resnet18_moco vits16_moco resnet50_dino vits16_dino"
splits_of() { case "$1" in resnet*) echo "layer1 layer2 layer3 layer4";; *) echo "block3 block6 block9 block12";; esac; }
pseudo() { local out=""; for e in "$@"; do for s in $(splits_of "$e"); do out="$out $e@$s"; done; done; echo $out; }
AUD_SPLITS=$(pseudo $AUDITED)
RND_SPLITS=$(pseudo resnet18_random)
ATTACK="python scripts/attack4b.py --attackers deep_mlp --out results/phase8/runs8b.csv --preds results/preds/phase8b"

run_8a() {
  echo "== 8a. Compute per split $(date '+%F %T')"
  if [ -f results/phase8/compute.csv ] && [ -z "${REDO_8A:-}" ]; then
    echo "results/phase8/compute.csv exists, skipped (REDO_8A=1 to redo)"
  else
    python scripts/compute8a.py
  fi
}

run_8b() {
  grep -q "Phase 8b split-point settings" results/preregistration.md || {
    echo "Record the 8b plan in results/preregistration.md (section 'Phase 8b split-point settings') first."; exit 1; }
  echo "== 8b.1 Extract split features $(date '+%F %T')"
  python scripts/splits8b.py
  echo "== 8b.2 Tier 1: presence leakage + BigEarthNet utility (gap), location (map4 spatial), audited encoders $(date '+%F %T')"
  $ATTACK --tasks leakage utility_ben --encoders $AUD_SPLITS --variants gap
  $ATTACK --tasks spatial --encoders $AUD_SPLITS --variants map4
  echo "== 8b.3 Tier 2: leakage_incr, EuroSAT utility, map4 leakage, random-encoder control $(date '+%F %T')"
  $ATTACK --tasks leakage leakage_incr utility_eurosat utility_ben --encoders $AUD_SPLITS $RND_SPLITS --variants gap
  $ATTACK --tasks spatial leakage --encoders $AUD_SPLITS $RND_SPLITS --variants map4
  if [ -n "${TIER3:-}" ]; then
    echo "== 8b.4 Tier 3: localiser on map4 at the earliest split $(date '+%F %T')"
    for e in $AUDITED; do
      first=$(splits_of "$e" | cut -d' ' -f1)
      [ -f "results/phase8/localise/$e@$first/map4/metrics.json" ] || \
        python scripts/localise.py --encoder "$e@$first" --variant map4 --out results/phase8/localise
    done
  fi
  echo "== 8b.5 Analysis $(date '+%F %T')"
  python scripts/analysis8b.py
}

run_8d() {
  [ -f results/phase8/runs8b.csv ] || { echo "Run 8b first."; exit 1; }
  grep -q "Phase 8d follow-ups" results/preregistration.md || {
    echo "Record the 8d note in results/preregistration.md (section 'Phase 8d follow-ups') first."; exit 1; }
  ENDS=""; for e in $AUDITED; do s=$(splits_of "$e"); ENDS="$ENDS $e@$(echo $s | cut -d' ' -f1) $e@$(echo $s | awk '{print $NF}')"; done
  echo "== 8d.1 Pixel 4x4 map control $(date '+%F %T')"
  [ -f data/compressed/pixels/map4/glosofarid.npy ] || python scripts/deepen8.py --build
  $ATTACK --tasks spatial leakage --encoders pixels --variants map4
  echo "== 8d.2 Spatial attack on the pooled vector at the earliest and final split $(date '+%F %T')"
  $ATTACK --tasks spatial --encoders $ENDS --variants gap
  for seed in 1 2; do
    echo "== 8d.3 Replication of the 8 primary contrasts, split seed $seed $(date '+%F %T')"
    $ATTACK --split-seed $seed --tasks leakage --encoders $ENDS --variants gap
    $ATTACK --split-seed $seed --tasks spatial --encoders $ENDS --variants map4
  done
  echo "== 8d.4 Analysis $(date '+%F %T')"
  python scripts/deepen8.py --analyse
}

run_8c() {
  echo "== 8c. Orbit-realistic ISLs $(date '+%F %T')"
  python scripts/isl8c.py
}

case "$stage" in
  8a) run_8a ;;
  8b) run_8b ;;
  8c) run_8c ;;
  8d) run_8d ;;
  all) run_8a; run_8b; run_8c ;;
  *) echo "unknown stage $stage (8a | 8b | 8c | 8d | all)"; exit 1 ;;
esac
echo "== Phase 8 stage '$stage' done $(date '+%F %T'). Results in results/phase8/"
