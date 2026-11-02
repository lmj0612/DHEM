#!/usr/bin/env python
"""
mj_20260903: 52-dim TEST-set conditions for the n99_tsh model (48d SML/SMU + Tilt + SYMH +
SML_mean_30m + SML_mean_120m) on the 44,349-sample stratified-SME climatology set (2000 + 2003).
Layout and z-score stats identical to the training file built by generate_conditions_tilt_symh_hist.py
(stats: conditions_joint_9_zscore_stats_SMU_SML_tilt_symh_hist.npz).
Sources (validated 2026-09-03 against the training 8-col file on 3000 random training minutes):
  Tilt : analytic dipole tilt (sun vector in GEO vs IGRF-ish dipole axis) -> median |diff| 0.04 deg
  SYMH : OMNI HRO 1-min SYM_H (cdaweb omni_hro2s) -> exact match
  SML history: SuperMAG global SML (1-min) rolling means, min_periods=5 (same recipe as surrogate table);
               NaN -> 0 after z-scoring (training did the same for 6084 rows).
"""
import os, sys, datetime as dt, numpy as np, pandas as pd  # mj_20260903
from cdflib import CDF, cdfepoch  # mj_20260903
sys.path.insert(0, os.path.dirname(__file__))  # mj_20260903
from generate_conditions_4cat_sampling import load_supermag_global_year, OMNI_CDF  # mj_20260903

JOINT = '/home/lim2/data/amie_npy_joint_from105/training_npy_joint/'  # mj_20260903
SAMP  = '/home/lim2/data/amie_npy_joint_from105/sampling_npy_joint/'  # mj_20260903
OUT   = os.path.join(SAMP, 'conditions_climatology_stratified_sme_tsh52.npz')  # mj_20260903
IN_FILE = os.environ.get('TSH52_IN', os.path.join(SAMP, 'conditions_climatology_stratified_sme.npz'))  # mj_20260906: any 48-d SMU/SML cond file (e.g. conditions_<cat>_minute_joint_9_zscore_SMU_SML.npz)
OUT = os.environ.get('TSH52_OUT', OUT)  # mj_20260906


def dipole_tilt_deg(epoch_s):  # mj_20260903
    """Analytic dipole tilt psi = asin(m_hat . s_hat) (deg), s = sun direction in GEO."""
    epoch_s = np.asarray(epoch_s, dtype=float)  # mj_20260903
    jd = epoch_s / 86400.0 + 2440587.5; n = jd - 2451545.0  # mj_20260903
    L = np.deg2rad((280.460 + 0.9856474 * n) % 360); g = np.deg2rad((357.528 + 0.9856003 * n) % 360)  # mj_20260903
    lam = L + np.deg2rad(1.915) * np.sin(g) + np.deg2rad(0.020) * np.sin(2 * g)  # mj_20260903
    eps = np.deg2rad(23.439 - 0.0000004 * n)  # mj_20260903
    sx, sy, sz = np.cos(lam), np.cos(eps) * np.sin(lam), np.sin(eps) * np.sin(lam)  # GEI  # mj_20260903
    T = n / 36525.0; th = np.deg2rad((280.46061837 + 360.98564736629 * n + 0.000387933 * T * T) % 360)  # mj_20260903
    gx, gy, gz = np.cos(th) * sx + np.sin(th) * sy, -np.sin(th) * sx + np.cos(th) * sy, sz  # GEO  # mj_20260903
    yr = 1970 + epoch_s / (365.25 * 86400)  # mj_20260903
    colat = np.deg2rad(10.26 + 0.03 * (yr - 2000)); lon = np.deg2rad(288.4 - 0.1 * (yr - 2000))  # mj_20260903
    mx, my, mz = np.sin(colat) * np.cos(lon), np.sin(colat) * np.sin(lon), np.cos(colat)  # mj_20260903
    return np.degrees(np.arcsin(np.clip(mx * gx + my * gy + mz * gz, -1, 1)))  # mj_20260903


def main():  # mj_20260903
    clim = np.load(IN_FILE, allow_pickle=True); print('input', IN_FILE)  # mj_20260903, mj_20260906 env TSH52_IN
    x48 = clim['conditions'].astype(np.float32); ts = pd.to_datetime(clim['timestamps'])  # mj_20260903
    N = len(ts); ep = (ts.values.astype('datetime64[s]').astype(np.int64)).astype(float)  # mj_20260903
    years = sorted(set(ts.year)); print('N', N, 'years', years)  # mj_20260903
    st = np.load(os.path.join(JOINT, 'conditions_joint_9_zscore_stats_SMU_SML_tilt_symh_hist.npz'), allow_pickle=True)  # mj_20260903
    mean, std = st['mean'], st['std']  # [52]  # mj_20260903
    assert str(st['col_names']).startswith('SML[0:24]+SMU[24:48]+Tilt[48]+SYMH[49]+SML_mean_30m[50]+SML_mean_120m[51]')  # mj_20260903

    # ---- Tilt ----
    tilt = dipole_tilt_deg(ep)  # mj_20260903
    # ---- SYMH from OMNI (restrict to needed years for speed) ----
    c = CDF(str(OMNI_CDF)); E = c.varget('Epoch'); SH = c.varget('SYM_H').astype(float)  # mj_20260903
    lo = cdfepoch.compute_epoch([years[0], 1, 1, 0, 0, 0, 0]); hi = cdfepoch.compute_epoch([years[-1] + 1, 1, 1, 0, 0, 0, 0])  # mj_20260903
    m = (E >= lo) & (E < hi); Es = np.asarray(cdfepoch.unixtime(E[m])); SH = SH[m]  # mj_20260903
    SH[(SH > 1000) | (SH < -1000)] = np.nan  # fill 99999  # mj_20260903
    i = np.clip(np.searchsorted(Es, ep), 0, len(Es) - 1); ok = np.abs(Es[i] - ep) <= 30  # mj_20260903
    symh = np.where(ok, SH[i], np.nan); print(f'SYMH matched {ok.sum()}/{N}, nan {np.isnan(symh).sum()}')  # mj_20260903
    # ---- SML history from SuperMAG global ----
    frames = []  # mj_20260903
    for y in years:  # mj_20260903
        t_arr, sml, smu = load_supermag_global_year(y)  # mj_20260903
        frames.append(pd.Series(sml, index=pd.to_datetime(t_arr)))  # mj_20260903
    s_sml = pd.concat(frames).sort_index(); s_sml = s_sml[~s_sml.index.duplicated()]  # mj_20260903
    full = pd.date_range(s_sml.index.min(), s_sml.index.max(), freq='1min'); s_sml = s_sml.reindex(full)  # mj_20260903
    h30 = s_sml.rolling(30, min_periods=5).mean(); h120 = s_sml.rolling(120, min_periods=5).mean()  # mj_20260903
    h30v = h30.reindex(ts).values; h120v = h120.reindex(ts).values  # mj_20260903
    print(f'SML hist nan: 30m {np.isnan(h30v).sum()}, 120m {np.isnan(h120v).sum()}')  # mj_20260903
    # sanity: SML_mean_30m should track the un-z-scored global-ish SML; compare with min over the 24 sectors
    s48 = np.load(os.path.join(JOINT, 'conditions_joint_9_zscore_stats_SMU_SML.npz'))  # mj_20260903
    sml_sec = x48[:, :24] * s48['std'][:24] + s48['mean'][:24]; sml_min = sml_sec.min(1)  # mj_20260903
    okc = ~np.isnan(h30v); print('corr(SML_mean_30m, min-sector SML) =', np.corrcoef(h30v[okc], sml_min[okc])[0, 1].round(3))  # mj_20260903

    extras = np.stack([tilt, symh, h30v, h120v], 1).astype(np.float32)  # mj_20260903
    extras_z = (extras - mean[48:]) / std[48:]; extras_z = np.nan_to_num(extras_z, nan=0.0)  # mj_20260903
    cond = np.concatenate([x48, extras_z], 1).astype(np.float32); assert cond.shape == (N, 52) and np.isfinite(cond).all()  # mj_20260903
    print('extras z mean/std:', extras_z.mean(0).round(3), extras_z.std(0).round(3))  # mj_20260903
    _opt = {k: clim[k] for k in ['bin_sme', 'sme_bin_edges', 'bin_counts', 'selected_indices'] if k in clim}  # mj_20260906 optional (storm-cat files lack them)
    np.savez(OUT, conditions=cond, npy_paths=clim['npy_paths'], timestamps=clim['timestamps'],  # mj_20260903, mj_20260906
             col_names=str(st['col_names']), raw_extras=extras, **_opt)  # mj_20260903, mj_20260906
    print('saved', OUT)  # mj_20260903


if __name__ == '__main__':  # mj_20260903
    main()  # mj_20260903
