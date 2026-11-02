#!/usr/bin/env python
"""Assemble a FAMILY conditions file for scripts/sample.py --family (mj_20261031).

The time-series mode needs, for every minute, the condition vector of EVERY category model (at a substorm-phase switch
both models generate the minutes around the switch), so a family file holds all minutes x all layouts:
  dhem_ap (also used by dhem_ape): timestamps [N], npy_names [N], forsyth_category [N], cond_cat1 .. cond_cat4 [N, d_k]
  dhem_ep                        : timestamps [N], npy_names [N], forsyth_category [N] (0 if unknown), conditions [N, 56]
Rows are unique minutes sorted in time.

Research version: reads the per-tag condition files of the authors' sampling directory
  conditions_<tag>_4cat_52d.npz (timestamps, npy_paths, forsyth_category), conditions_<tag>_tsh52x_cat<k>.npz (k = 1..4),
  conditions_<tag>_minute_joint_9_zscore_pot56.npz
Several tags are merged in the given order (the first occurrence of a minute wins), as in the paper's event sampler.

  python scripts/data_prep/make_family_conditions.py --family dhem_ap --src-dir <dir> --tags before_super_storm super_storm \
      after_super_storm --window "2000-04-06 00:00,2000-04-08 00:00" --out examples/2000-04-06/conditions/conds_dhem_ap_2000-04-06.npz
  options: --names-from <npz with npy_names>  keep only these minutes;  --minute-filter 0,30  keep these minute-of-hour values
"""
import argparse, os  # mj_20261031
import numpy as np  # mj_20261031


def main():  # mj_20261031
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)  # mj_20261031
    ap.add_argument('--family', choices=['dhem_ap', 'dhem_ep'], required=True)  # mj_20261031
    ap.add_argument('--src-dir', required=True)  # mj_20261031
    ap.add_argument('--tags', nargs='+', required=True)  # mj_20261031
    ap.add_argument('--window', default=None, help="'YYYY-mm-dd HH:MM,YYYY-mm-dd HH:MM' (end exclusive)")  # mj_20261031
    ap.add_argument('--names-from', default=None)  # mj_20261031
    ap.add_argument('--minute-filter', default=None)  # mj_20261031
    ap.add_argument('--out', required=True)  # mj_20261031
    a = ap.parse_args()  # mj_20261031
    t0, t1 = ([np.datetime64(w.strip().replace(' ', 'T'), 'm') for w in a.window.split(',')] if a.window else (None, None))  # mj_20261031
    keep_names = set(str(x) for x in np.load(a.names_from, allow_pickle=True)['npy_names']) if a.names_from else None  # mj_20261031
    mfilt = {int(m) for m in a.minute_filter.split(',')} if a.minute_filter else None  # mj_20261031
    layouts = ['cond_cat1', 'cond_cat2', 'cond_cat3', 'cond_cat4'] if a.family == 'dhem_ap' else ['conditions']  # mj_20261031
    rows = {}  # mj_20261031 datetime64 -> dict
    for tag in a.tags:  # mj_20261031
        if a.family == 'dhem_ap':  # mj_20261031
            base = np.load(f'{a.src_dir}/conditions_{tag}_4cat_52d.npz', allow_pickle=True)  # mj_20261031
            ts, paths, cat = base['timestamps'], base['npy_paths'], base['forsyth_category'].astype(int)  # mj_20261031
            src = {}  # mj_20261031
            for k in (1, 2, 3, 4):  # mj_20261031
                d = np.load(f'{a.src_dir}/conditions_{tag}_tsh52x_cat{k}.npz', allow_pickle=True)  # mj_20261031
                assert (d['npy_paths'] == paths).all(), f'{tag} cat{k}: row order differs from the 4cat_52d file'  # mj_20261031
                src[f'cond_cat{k}'] = d['conditions'].astype(np.float32)  # mj_20261031
        else:  # mj_20261031
            d = np.load(f'{a.src_dir}/conditions_{tag}_minute_joint_9_zscore_pot56.npz', allow_pickle=True)  # mj_20261031
            ts, paths = d['timestamps'], d['npy_paths']; cat = np.zeros(len(ts), int)  # mj_20261031
            src = {'conditions': d['conditions'].astype(np.float32)}  # mj_20261031
        for i, s in enumerate(ts):  # mj_20261031
            t = np.datetime64(str(s)[:16].replace(' ', 'T'), 'm'); name = os.path.basename(str(paths[i]))  # mj_20261031
            if t in rows or (t0 is not None and (t < t0 or t >= t1)):  # mj_20261031
                continue  # mj_20261031
            if mfilt is not None and t.astype(object).minute not in mfilt:  # mj_20261031
                continue  # mj_20261031
            if keep_names is not None and name not in keep_names:  # mj_20261031
                continue  # mj_20261031
            rows[t] = dict(name=name, cat=int(cat[i]), **{k: v[i] for k, v in src.items()})  # mj_20261031
    keys = sorted(rows)  # mj_20261031
    assert keys, 'no minutes selected'  # mj_20261031
    out = dict(timestamps=np.array([str(k).replace('T', ' ') for k in keys]), npy_names=np.array([rows[k]['name'] for k in keys]),  # mj_20261031
               forsyth_category=np.array([rows[k]['cat'] for k in keys], np.int8),  # mj_20261031
               note=np.array(f'family conditions ({a.family}), tags {a.tags}, window {a.window}, built by make_family_conditions.py'))  # mj_20261031
    for k in layouts:  # mj_20261031
        out[k] = np.stack([rows[t][k] for t in keys]).astype(np.float32)  # mj_20261031
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)  # mj_20261031
    np.savez_compressed(a.out, **out)  # mj_20261031
    print(f'saved {a.out}: {len(keys)} minutes {out["timestamps"][0]} .. {out["timestamps"][-1]}; layouts ' + ', '.join(f'{k} {out[k].shape[1]}-d' for k in layouts)  # mj_20261031
          + (f'; categories {np.bincount(out["forsyth_category"], minlength=5)[1:].tolist()}' if a.family == 'dhem_ap' else ''))  # mj_20261031


if __name__ == '__main__':  # mj_20261031
    main()  # mj_20261031
