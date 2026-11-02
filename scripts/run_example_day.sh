#!/bin/bash
# 2003-11-20, 132 minutes at 10-min cadence: the three families in both modes, a 5-member DHEM-AP ensemble, HP / CPCP vs the
# paper run (and vs AMIE if examples/2003-11-20/amie_npy/ exists). Usage: bash scripts/run_example_day.sh [OUT_DIR] [DEVICE]
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd); EX=$ROOT/examples/2003-11-20; OUT=${1:-$EX/out_dryrun}; DEV=${2:-cuda}
PY=${PYTHON:-python}; mkdir -p $OUT; cd $ROOT
C=$EX/conditions
run() { echo "=== $(date '+%F %T') $*"; $PY scripts/sample.py "$@" --device $DEV --batch-size 128 --seed 0 || echo "FAILED $*"; }
run --family dhem_ep  --conditions $C/conds_pot56_2003-11-20.npz   --out $OUT/dhem_ep.npz
run --family dhem_ap  --conditions $C/conds_dhem_ap_2003-11-20.npz --out $OUT/dhem_ap.npz
run --family dhem_ape --conditions $C/conds_dhem_ap_2003-11-20.npz --out $OUT/dhem_ape.npz
run --family dhem_ap  --conditions $C/conds_dhem_ap_2003-11-20.npz --out $OUT/dhem_ap_independent.npz --mode independent
run --family dhem_ap  --conditions $C/conds_dhem_ap_2003-11-20.npz --out $OUT/dhem_ap_R5.npz --members 5
echo "=== $(date '+%F %T') evaluate"
P=$EX/paper_outputs; AM=""; [ -d $EX/amie_npy ] && AM="--amie-dir $EX/amie_npy"
$PY scripts/evaluate.py --samples $OUT/dhem_ap.npz  $AM --paper $P/dhem_ap_flux_maps_2003-11-20.npz  --out $OUT/metrics_HP.json
$PY scripts/evaluate.py --samples $OUT/dhem_ap_independent.npz $AM --paper $P/dhem_ap_flux_maps_2003-11-20.npz --out $OUT/metrics_HP_independent.json
$PY scripts/evaluate.py --samples $OUT/dhem_ep.npz  $AM --paper $P/dhem_ep_pot_maps_2003-11-20.npz   --out $OUT/metrics_CPCP.json
[ -n "$AM" ] && $PY scripts/evaluate.py --samples $OUT/dhem_ape.npz $AM --paper $P/dhem_ape_energy_maps_2003-11-20.npz --out $OUT/metrics_HME.json
$PY - <<PYEOF
import numpy as np, json, sys
sys.path.insert(0, '$ROOT'); from dhem import metrics; from dhem.spatial_basis import mlat_rows
d = np.load('$OUT/dhem_ap_R5.npz', allow_pickle=True); aw = metrics.area_weights(mlat_rows())
hp = np.array([[metrics.hemispheric_power(m, aw) for m in row] for row in d['physical']])
print(f"R=5 ensemble (DHEM-AP, N={hp.shape[0]}): mean per-minute HP std {hp.std(axis=1, ddof=1).mean():.1f} GW, mean HP {hp.mean():.1f} GW, sampling time {float(d['elapsed_s']):.0f} s")
json.dump(dict(N=int(hp.shape[0]), R=5, hp_std_mean_GW=float(hp.std(axis=1, ddof=1).mean()), hp_mean_GW=float(hp.mean())), open('$OUT/ensemble_spread_ap_R5.json', 'w'), indent=2)
PYEOF
echo "=== $(date '+%F %T') ALL DONE"
