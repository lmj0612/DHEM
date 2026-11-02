#!/usr/bin/env python
# mj_20260918 -- 48-d SML/SMU base condition file for a fixed-cadence 2003 TEST-year evaluation set (paper Fig 2 Bland-Altman,
# both models on the SAME minutes): every cleaned 2003 minute of sampling_npy_joint at a 10-min clock cadence (hh:00,10,..,50)
# that also has valid OMNI (= selected_labels of the pot56 cond file built by pot/scripts/build_pot56_2003_stratified.py with
# MINUTE_FILE=minutes_2003_cad10_request.txt). No random draw, no stratification, no trimming: the whole test year at its
# natural activity / season / UT proportions; the 30-min subset (hh:00, hh:30) is beyond the error decorrelation time (<0.1 at >30 min).
# Same recipe as build_halloween_case.py step 2 (SuperMAG MLT file, z-scored with conditions_joint_9_zscore_stats_SMU_SML.npz, gaps -> 0).
# Output: sampling_npy_joint/conditions_<PREFIX>_minute_joint_9_zscore_SMU_SML.npz (+ minutes_<PREFIX>.txt), then run
#   FOURCAT_PREFIX=<PREFIX> generate_conditions_4cat_sampling.py ; TSH52_IN/TSH52_OUT generate_conditions_tsh52_sampling.py ;
#   TSH52X_PREFIX=<PREFIX> generate_conditions_4cat_tsh52x.py
import os, sys, numpy as np
from datetime import datetime, timedelta
B = '/home/lim2/data/amie_npy_joint_from105/'; S = B + 'sampling_npy_joint/'; T = B + 'training_npy_joint/'
PREFIX = os.environ.get('PREFIX', 'y2003cad10'); POT = os.environ.get('POT_COND', S + 'conditions_y2003cad10_pot56_2003.npz')
labels = [str(l) for l in np.load(POT, allow_pickle=True)['selected_labels']]; N = len(labels); print(f'{POT}: {N} minutes')
open(S + f'minutes_{PREFIX}.txt', 'w').write('\n'.join(labels) + '\n')
def lab2dt(l): d, m = l.split('_'); return datetime.strptime(d, '%Y%m%d') + timedelta(minutes=int(m))
dts = [lab2dt(l) for l in labels]; assert all(os.path.exists(S + l + '.npy') for l in labels)
GAP = 999999; SML_COLS = list(range(6, 30)); SMU_COLS = list(range(30, 54)); mlt = {}
with open('/home/lim2/data/supermag/MLT/2003-supermag.txt') as fh:
    for line in fh:
        s = line.strip()
        if not (s and s[0].isdigit()): continue
        p = s.split()
        if len(p) != 54: continue
        v = [float(x) for x in p]; dt = datetime(int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]))
        a = np.array([v[c] for c in SML_COLS]); b = np.array([v[c] for c in SMU_COLS]); a[a >= GAP] = np.nan; b[b >= GAP] = np.nan
        mlt[dt] = np.concatenate([a, b])
st = np.load(T + 'conditions_joint_9_zscore_stats_SMU_SML.npz'); mean48, std48 = st['mean'], st['std']
c48 = np.zeros((N, 48), np.float32); miss = 0
for i, dt in enumerate(dts):
    if dt in mlt: c48[i] = np.nan_to_num((mlt[dt] - mean48) / (std48 + 1e-6), nan=0.0)
    else: miss += 1
# self-check: the pot56 file carries the same 48-d block in cols 8:56
pc = np.load(POT, allow_pickle=True)['conditions'][:, 8:56]; print(f'  48-d missing minutes {miss}/{N}; max|diff| vs pot56 cols 8:56 = {np.abs(pc - c48).max():.2e}')
paths = np.array([S + l + '.npy' for l in labels]); ts = np.array([d.strftime('%Y-%m-%d %H:%M') for d in dts])
out = S + f'conditions_{PREFIX}_minute_joint_9_zscore_SMU_SML.npz'; np.savez(out, conditions=c48, npy_paths=paths, timestamps=ts); print('saved', out, c48.shape)
