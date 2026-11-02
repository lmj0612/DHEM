#!/bin/bash
# 6-7 April 2000 (Figures 6-8): DHEM-AP and DHEM-EP in the time-series mode with the seeds of the paper, compared with the paper
# maps. A third argument "independent" adds DHEM-AP with independent noise. Usage: bash scripts/run_example_event.sh [OUT_DIR] [DEVICE] [independent]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); EX=$ROOT/examples/2000-04-06; OUT=${1:-$EX/out}; DEV=${2:-cuda}; EXTRA=${3:-}
PY=${PYTHON:-python}; mkdir -p $OUT; cd $ROOT
C=$EX/conditions; P=$EX/paper_outputs
run() { echo "=== $(date '+%F %T') $*"; $PY scripts/sample.py "$@" --device $DEV --batch-size 256 || echo "FAILED $*"; }
SEED_AP=1000; SEED_EP=0
run --family dhem_ap --conditions $C/conds_dhem_ap_2000-04-06.npz --out $OUT/dhem_ap.npz --seed $SEED_AP
run --family dhem_ep --conditions $C/conds_dhem_ep_2000-04-06.npz --out $OUT/dhem_ep.npz --seed $SEED_EP
$PY scripts/compare_event.py --samples $OUT/dhem_ap.npz --paper $P/dhem_ap_flux_maps_2000-04-06.npz --out $OUT/compare_ap.json --plot $OUT/compare_ap_HP.png
$PY scripts/compare_event.py --samples $OUT/dhem_ep.npz --paper $P/dhem_ep_pot_maps_2000-04-06.npz  --out $OUT/compare_ep.json --plot $OUT/compare_ep_CPCP.png
if [ "$EXTRA" = "independent" ]; then
  run --family dhem_ap --conditions $C/conds_dhem_ap_2000-04-06.npz --out $OUT/dhem_ap_independent.npz --mode independent --seed $SEED_AP
  $PY scripts/compare_event.py --samples $OUT/dhem_ap_independent.npz --paper $P/dhem_ap_flux_maps_2000-04-06.npz --out $OUT/compare_ap_independent.json --plot $OUT/compare_ap_independent_HP.png
fi
echo "=== $(date '+%F %T') ALL DONE"
