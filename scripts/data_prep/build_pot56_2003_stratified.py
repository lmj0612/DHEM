#!/usr/bin/env python
# mj_20260914 -- build a 2003-TEST-YEAR stratified (epsilon_t x clock angle) condition file for the pot56-MLAT model,
# same recipe as amie_data_process/generate_npz_1min_sampling_climatology.py (8 eps bins x 8 clock sectors, K per bin,
# one minute per 30-min window, seed 42) and same 56-d layout as conditions_climatology_stratified_pot56.npz
# (from105 sw8 [Newell, sin, cos, SYM-H, Pdyn, F107, AE, tilt] z-scored with conditions_joint_9_zscore_stats.npz
#  + 24 SML + 24 SMU z-scored with conditions_joint_9_zscore_stats_SMU_SML.npz, gaps -> 0).
# sw8 cols 0-5 and AE come from keep_dict_joint.npz (old layout [.., AL, AE]); tilt = analytic dipole tilt
# (generate_conditions_tsh52_sampling.dipole_tilt_deg, median |diff| 0.04 deg vs IGRF). Every 2003 minute that also
# appears in the 9 storm pot56 files is used as a self-check of the layout / z-scoring.
import os, sys, zipfile, io, numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, '/home/lim2/code/improved-diffusion_mj_1D_flux_claude/scripts')
from generate_conditions_tsh52_sampling import dipole_tilt_deg  # mj_20260914
S = '/home/lim2/data/amie_npy_joint_from105/sampling_npy_joint/'; T = '/home/lim2/data/amie_npy_joint_from105/training_npy_joint/'
KEEP = '/home/lim2/data/amie_npy_joint/1999_2010_condition/keep_dict_joint.npz'
YEAR = os.environ.get('YEAR', '2003'); K_PER_BIN = int(os.environ.get('K_PER_BIN', '1000')); MIN_SEP = 30; SEED = 42
OUT = os.environ.get('OUT', S + f'conditions_climatology_stratified_pot56_{YEAR}.npz')
# ---- 1. minutes of the year present in the sampling set
files = sorted(f for f in os.listdir(S) if f.endswith('.npy') and f.startswith(YEAR))
labels = [f[:-4] for f in files]; N = len(labels); print(f'{YEAR}: {N} npy minutes')
def lab2dt(l): d, m = l.split('_'); return datetime.strptime(d, '%Y%m%d') + timedelta(minutes=int(m))
dts = [lab2dt(l) for l in labels]
# ---- 2. sw8 raw from keep_dict (zip member read, no full load)
if os.environ.get('KEEP_MODE', 'live') == 'live':  # mj_20260914 EXACT route: the from105 pipeline never saved its keep_dict; it is rebuilt in memory by importing
    # generate_condition_all (hro2s OMNI, 5-min gap interpolation, joint layout [Newell, sin, cos, SYM-H, Pdyn, F107, AE, tilt]) -- same as
    # transfer_amie_parallel_new.py / generate_npz_1min.py did.  A copy with TRAINING_MODES['joint_training']=True is imported (~5 min).
    gdir = os.environ.get('GCA_DIR', os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, gdir); os.chdir('/home/lim2/code/amie_data_process')
    import gca_joint; keep = gca_joint.keep_dict; print(f'  live keep_dict: {len(keep)} entries; sample', next(iter(keep.items())))
    sw8 = np.full((N, 8), np.nan, np.float32); miss = 0
    for i, l in enumerate(labels):
        v = keep.get(l + '.npy', keep.get(l))
        if v is None: miss += 1
        else: sw8[i] = np.asarray(v, np.float32)
    print(f'  keep_dict missing {miss}/{N}; tilt col7 range {np.nanmin(sw8[:, 7]):.1f}..{np.nanmax(sw8[:, 7]):.1f}')
else:  # old on-disk keep_dict_joint.npz (hros OMNI, [.., AL, AE]) + analytic tilt -- NOT identical to from105 (kept for reference)
    zf = zipfile.ZipFile(KEEP); members = set(zf.namelist()); raw8 = np.full((N, 8), np.nan, np.float32); miss = 0
    for i, l in enumerate(labels):
        m = l + '.npy'
        if m in members: raw8[i] = np.load(io.BytesIO(zf.read(m)), allow_pickle=True)
        else: miss += 1
        if i % 50000 == 0: print(f'  keep_dict {i}/{N}', flush=True)
    print(f'  keep_dict missing {miss}/{N}')
    ep = np.array([(d - datetime(1970, 1, 1)).total_seconds() for d in dts]); tilt = dipole_tilt_deg(ep)
    sw8 = np.stack([raw8[:, 0], raw8[:, 1], raw8[:, 2], raw8[:, 3], raw8[:, 4], raw8[:, 5], raw8[:, 7], tilt], 1).astype(np.float32)  # AE=col7 of old layout
st8 = np.load(T + 'conditions_joint_9_zscore_stats.npz'); zi = st8['z_idx']; z8 = sw8.copy(); z8[:, zi] = (sw8[:, zi] - st8['mean']) / (st8['std'] + 1e-6)
# ---- 3. 48-d SML/SMU from SuperMAG MLT file
GAP = 999999; SMLC = list(range(6, 30)); SMUC = list(range(30, 54)); mlt = {}
with open(f'/home/lim2/data/supermag/MLT/{YEAR}-supermag.txt') as fh:
    for line in fh:
        s = line.strip()
        if not (s and s[0].isdigit()): continue
        p = s.split()
        if len(p) != 54: continue
        v = [float(x) for x in p]; dt = datetime(int(v[0]), int(v[1]), int(v[2]), int(v[3]), int(v[4]))
        a = np.array([v[c] for c in SMLC]); b = np.array([v[c] for c in SMUC]); a[a >= GAP] = np.nan; b[b >= GAP] = np.nan
        mlt[dt] = np.concatenate([a, b])
st48 = np.load(T + 'conditions_joint_9_zscore_stats_SMU_SML.npz'); m48, s48 = st48['mean'], st48['std']
c48 = np.zeros((N, 48), np.float32); miss48 = 0
for i, d in enumerate(dts):
    if d in mlt: c48[i] = np.nan_to_num((mlt[d] - m48) / (s48 + 1e-6), nan=0.0)
    else: miss48 += 1
print(f'  SuperMAG 48-d missing {miss48}/{N}')
c56 = np.concatenate([z8, c48], 1)
# ---- 4. self-check against the stored storm pot56 files (same minutes must give the same 56-d vector)
lab2i = {l: i for i, l in enumerate(labels)}; chk = []
for cat in ['super_storm', 'strong_storm', 'moderate_storm', 'before_super_storm', 'before_strong_storm', 'before_moderate_storm', 'after_super_storm', 'after_strong_storm', 'after_moderate_storm']:
    f = S + f'conditions_{cat}_minute_joint_9_zscore_pot56.npz'
    if not os.path.exists(f): continue
    z = np.load(f, allow_pickle=True); C = z['conditions']
    for j, p in enumerate(z['npy_paths']):
        l = os.path.basename(str(p)).replace('.npy', '')
        if l in lab2i and not np.isnan(c56[lab2i[l]]).any(): chk.append((C[j], c56[lab2i[l]]))
if chk:
    A = np.array([a for a, _ in chk]); B = np.array([b for _, b in chk]); D = np.abs(A - B)
    print(f'  SELF-CHECK on {len(chk)} storm minutes: max|diff| sw8 cols {np.round(D[:, :8].max(0), 4)}  SML/SMU max {D[:, 8:].max():.4f}  (tilt col 7 is analytic vs IGRF)')
else: print('  SELF-CHECK: no overlap found (unexpected)')
# ---- 5. stratify
valid = ~np.isnan(sw8).any(1); print(f'  valid minutes (OMNI present): {valid.sum()}/{N}')
eps_t = sw8[:, 0]; clock = (np.degrees(np.arctan2(sw8[:, 1], sw8[:, 2])) + 360) % 360
edges = np.array([-np.inf, 3000, 7500, 12000, 16500, 21000, 25500, 30000, np.inf]); centers = np.arange(0, 360, 45); width = 22.5
ebin = np.full(N, -1, int)
for i in range(8): ebin[(eps_t >= edges[i]) & (eps_t < edges[i + 1])] = i
ebin[~valid] = -1
cbin = np.full(N, -1, int)
for i, c in enumerate(centers):
    lo, hi = (c - width) % 360, (c + width) % 360
    cbin[((clock >= lo) & (clock < hi)) if lo < hi else ((clock >= lo) | (clock < hi))] = i
tmin = np.array([(d - datetime(1970, 1, 1)).total_seconds() // 60 for d in dts]).astype(np.int64)
MINUTE_FILE = os.environ.get('MINUTE_FILE', '')  # mj_20260915 optional: text file of npy labels (YYYYMMDD_minute) -> take exactly these valid minutes, no stratification (block-sampled BA set)
rng = np.random.default_rng(SEED); sel, seb, scb = [], [], []; counts = np.zeros((8, 8), int)
if MINUTE_FILE:  # mj_20260915
    want = [l.strip() for l in open(MINUTE_FILE) if l.strip()]; miss_l = [l for l in want if l not in lab2i]
    idx = np.array([lab2i[l] for l in want if l in lab2i]); ok = valid[idx] & (ebin[idx] >= 0) & (cbin[idx] >= 0); idx = idx[ok]
    print(f'  MINUTE_FILE: {len(want)} requested, {len(miss_l)} not in sampling set, {(~ok).sum()} without valid OMNI -> {len(idx)} used')
    sel = idx.tolist(); seb = ebin[idx].tolist(); scb = cbin[idx].tolist()
    for ie in range(8):
        for ic in range(8): counts[ie, ic] = np.sum((ebin[idx] == ie) & (cbin[idx] == ic))
for ie in (range(8) if not MINUTE_FILE else []):  # mj_20260915 stratified route unchanged
    for ic in range(8):
        idx = np.where((ebin == ie) & (cbin == ic))[0]
        if len(idx) == 0: print(f'  [eps={ie} clk={ic}] EMPTY'); continue
        o = np.argsort(tmin[idx]); idx = idx[o]; t = tmin[idx]; w = (t - t[0]) // MIN_SEP
        elig = np.array([rng.choice(idx[w == u]) for u in np.unique(w)])
        ch = rng.choice(elig, size=K_PER_BIN, replace=False) if len(elig) >= K_PER_BIN else elig
        if len(elig) < K_PER_BIN: print(f'  [eps={ie} clk={ic}] only {len(elig)} eligible (in bin {len(idx)})')
        sel += ch.tolist(); seb += [ie] * len(ch); scb += [ic] * len(ch); counts[ie, ic] = len(ch)
sel = np.array(sel); print('bin_counts:\n', counts, '\ntotal', len(sel))
np.savez(OUT, conditions=c56[sel], bin_epsilon=np.array(seb), bin_clock=np.array(scb), epsilon_bin_edges=edges, clock_bin_centers=centers.astype(float), clock_bin_width=width,
         bin_counts=counts, K_per_bin=np.int64(K_PER_BIN), min_separation_min=np.int64(MIN_SEP), selected_labels=np.array([labels[i] for i in sel]),
         selected_timestamps=np.array([dts[i].strftime('%Y-%m-%d %H:%M') for i in sel]), orig_index=sel, npy_files=np.array([files[i] for i in sel]),
         col_names=f'from105 sw8[0:8] (live generate_condition_all joint keep_dict, KEEP_MODE={os.environ.get("KEEP_MODE","live")}) + SML[8:32] + SMU[32:56]; {YEAR} TEST-year minutes from sampling_npy_joint (AMIE npy use test20 LINEAR ranges); built by build_pot56_2003_stratified.py mj_20260914')
print('saved', OUT)
