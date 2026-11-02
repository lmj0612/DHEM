#!/usr/bin/env python
"""
mj_20260903: ROUND 2b condition files = 52d (SML/SMU + Tilt + SYMH + SML_mean_30m + SML_mean_120m) + the ORIGINAL
per-category extras of the 2026-03 4cat design (paper draft):
    cat1 growth      + Newell + T1 + T2          -> 55d
    cat2 compound    + dSML_filt/dt              -> 53d
    cat3 isolated    + dSML_filt/dt              -> 53d
    cat4 convection  + dSML_filt/dt + Newell     -> 54d
Extras are copied (already z-scored with conditions_4cat_zscore_stats.npz) from the existing cat cond files (training)
and from conditions_climatology_stratified_sme_4cat_52d.npz (test: Newell[48] T1[49] T2[50] dSML_dt[51]).
"""
import os, numpy as np  # mj_20260903
B = '/home/lim2/data/amie_npy_joint_from105/'; S = B + 'sampling_npy_joint/'  # mj_20260903
CATS = {1: 'cat1_growth', 2: 'cat2_compound', 3: 'cat3_isolated', 4: 'cat4_convection'}  # mj_20260903
# training: which columns (>=48) of the old cat file to append, and the matching test columns of the 4cat_52d file
TRAIN_X = {1: [48, 49, 50], 2: [48], 3: [48], 4: [48, 49]}  # mj_20260903
TEST_X = {1: [48, 49, 50], 2: [51], 3: [51], 4: [51, 48]}   # mj_20260903 (cat4: dSML then Newell = training order)
NAMES = {1: 'Newell+T1+T2', 2: 'dSML_filt_dt', 3: 'dSML_filt_dt', 4: 'dSML_filt_dt+Newell'}  # mj_20260903
PREFIX = os.environ.get('TSH52X_PREFIX', 'climatology_stratified_sme')  # mj_20260906: e.g. super_storm -> test-only mode (no training files rewritten)
TEST_ONLY = PREFIX != 'climatology_stratified_sme'  # mj_20260906
tsh_test = np.load(S + f'conditions_{PREFIX}_tsh52.npz', allow_pickle=True)  # mj_20260903, mj_20260906
t52 = np.load(S + f'conditions_{PREFIX}_4cat_52d.npz', allow_pickle=True)  # mj_20260903, mj_20260906
assert (t52['npy_paths'] == tsh_test['npy_paths']).all()  # mj_20260903
for k, nm in CATS.items():  # mj_20260903
    t52k = np.load(f'{B}training_{nm}/conditions_{nm}_tsh52.npz', allow_pickle=True)  # mj_20260903
    cn = str(t52k['col_names']) + f'+{NAMES[k]}[52:{52 + len(TEST_X[k])}]'  # mj_20260903, mj_20260906
    if not TEST_ONLY:  # mj_20260906
        old = np.load(f'{B}training_{nm}/conditions_{nm}_zscore.npz', allow_pickle=True)  # mj_20260903
        assert (old['npy_files'] == t52k['npy_files']).all() and np.allclose(old['conditions'][:, :48], t52k['conditions'][:, :48])  # mj_20260903
        xtr = np.concatenate([t52k['conditions'], old['conditions'][:, TRAIN_X[k]]], 1).astype(np.float32)  # mj_20260903
        np.savez(f'{B}training_{nm}/conditions_{nm}_tsh52x.npz', conditions=xtr, npy_files=old['npy_files'], npy_paths=old['npy_paths'],  # mj_20260903
                 timestamps=old['timestamps'], col_names=cn)  # mj_20260903
    assert (t52['npy_paths'] == tsh_test['npy_paths']).all()  # mj_20260906
    xte = np.concatenate([tsh_test['conditions'], t52['conditions'][:, TEST_X[k]]], 1).astype(np.float32)  # mj_20260903
    np.savez(S + f'conditions_{PREFIX}_tsh52x_cat{k}.npz', conditions=xte, npy_paths=tsh_test['npy_paths'],  # mj_20260903, mj_20260906
             timestamps=tsh_test['timestamps'], col_names=cn)  # mj_20260903
    print(f'cat{k}: test {xte.shape} ({PREFIX}) | {cn}')  # mj_20260903, mj_20260906
    if not TEST_ONLY:  # mj_20260906
        print(f'   train extras mean/std {xtr[:, 52:].mean(0).round(2)} {xtr[:, 52:].std(0).round(2)} | test (cat{k} rows) '  # mj_20260903
          f'{xte[t52["forsyth_category"] == k, 52:].mean(0).round(2)} {xte[t52["forsyth_category"] == k, 52:].std(0).round(2)}')  # mj_20260903
