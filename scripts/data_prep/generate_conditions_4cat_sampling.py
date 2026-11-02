#!/usr/bin/env python
"""
Generate universal 52-dim condition files for 4-category Forsyth-phase sampling.

For each sample in the existing analysis (9 storm cats) and climatology (SME-stratified)
workflows, compute all conditioning variables and assign Forsyth category.

Universal layout (52 cols):
  [0:48]  = 48d SML/SMU (z-scored, from existing condition files)
  [48]    = Newell coupling (z-scored)
  [49]    = T1 = log10(1 + min since last Forsyth expansion onset) (z-scored)
  [50]    = T2 = log10(1 + min to next Forsyth expansion onset) (z-scored)
  [51]    = dSML_filt/dt (z-scored)

Each model selects its required columns at sampling time:
  Cat1 Growth:      [0:48, 48, 49, 50] → 51d
  Cat2 Compound:    [0:48, 51]          → 49d
  Cat3 Isolated:    [0:48, 51]          → 49d
  Cat4 Convection:  [0:48, 51, 48]      → 50d

mj_20260311
"""
import os  # mj_20260311
import sys  # mj_20260311
import numpy as np  # mj_20260311
from datetime import datetime, timedelta  # mj_20260311
from pathlib import Path  # mj_20260311
from scipy.signal import butter, filtfilt  # mj_20260311
import pandas as pd  # mj_20260311
import glob  # mj_20260311

# ============================================================
# Config
# ============================================================
DATA_ROOT = Path('/home/lim2/data/amie_npy_joint_from105')  # mj_20260311
REPO = Path('/home/lim2/code/improved-diffusion_mj_1D_flux_claude')  # mj_20260311
SUPERMAG_GLOBAL_DIR = Path('/home/lim2/data/supermag/global')  # mj_20260311
OMNI_CDF = Path('/home/lim2/data/CDAW/omni_hro2s_1min_19970101000000_20110101000000_cdaweb.cdf')  # mj_20260311

SAMPLING_YEARS = [2000, 2003]  # mj_20260311
EPT = 75  # mj_20260311

# Forsyth filter params  # mj_20260311
FILTER_CUTOFF_MIN = 30  # mj_20260311
FILTER_ORDER = 4  # mj_20260311
GAP_THRESH_SEC = 120  # mj_20260311

# T1/T2 defaults  # mj_20260311
DEFAULT_T_MIN = 1440.0  # 24h default when no onset found  # mj_20260311

# Column mapping for the sampling script  # mj_20260311
COL_SELECT = {  # mj_20260311
    1: list(range(48)) + [48, 49, 50],  # Cat1: SML/SMU + Newell + T1 + T2 = 51d
    2: list(range(48)) + [51],           # Cat2: SML/SMU + dSML_filt/dt = 49d
    3: list(range(48)) + [51],           # Cat3: SML/SMU + dSML_filt/dt = 49d
    4: list(range(48)) + [51, 48],       # Cat4: SML/SMU + dSML_filt/dt + Newell = 50d
}


# ============================================================
# Reuse functions from generate_conditions_4cat.py
# ============================================================
def load_supermag_global_year(year):  # mj_20260311
    fpath = SUPERMAG_GLOBAL_DIR / f'{year}-supermag.txt'  # mj_20260311
    ts_list, sml_list, smu_list = [], [], []  # mj_20260311
    with open(fpath) as f:  # mj_20260311
        for _ in range(105):  # mj_20260311
            next(f)
        for line in f:  # mj_20260311
            parts = line.strip().split('\t')
            if len(parts) < 8:
                continue
            try:
                y, mo, d, h, mi, s = (int(parts[i]) for i in range(6))
                sml = float(parts[6])  # mj_20260311
                smu = float(parts[7])  # mj_20260311
            except (ValueError, IndexError):
                continue
            if np.isnan(sml) or np.isnan(smu):  # mj_20260311
                continue
            ts_list.append(datetime(y, mo, d, h, mi, s))  # mj_20260311
            sml_list.append(sml)
            smu_list.append(smu)
    return np.array(ts_list), np.array(sml_list, dtype=float), np.array(smu_list, dtype=float)


def find_segments(ts_arr, gap_sec=120):  # mj_20260311
    if len(ts_arr) <= 1:  # mj_20260311
        return [(0, len(ts_arr))]
    dt_sec = np.array([(ts_arr[i+1] - ts_arr[i]).total_seconds()
                        for i in range(len(ts_arr) - 1)])  # mj_20260311
    gaps = np.where(dt_sec > gap_sec)[0]  # mj_20260311
    segs = []  # mj_20260311
    start = 0
    for g in gaps:
        if g + 1 - start >= 3:  # mj_20260311
            segs.append((start, g + 1))
        start = g + 1
    if len(ts_arr) - start >= 3:  # mj_20260311
        segs.append((start, len(ts_arr)))
    return segs  # mj_20260311


def lowpass_filter(sml, segments, cutoff_min=30, order=4):  # mj_20260311
    sml_filt = np.full_like(sml, np.nan)  # mj_20260311
    nyq = 0.5  # mj_20260311
    wn = (1.0 / cutoff_min) / nyq  # mj_20260311
    b, a = butter(order, wn, btype='low')  # mj_20260311
    padlen = 3 * max(len(a), len(b))  # mj_20260311
    for s, e in segments:  # mj_20260311
        if e - s > padlen:
            sml_filt[s:e] = filtfilt(b, a, sml[s:e])  # mj_20260311
        else:
            sml_filt[s:e] = sml[s:e]  # mj_20260311
    return sml_filt  # mj_20260311


def deriv_3pt(data, segments, dt=1.0):  # mj_20260311
    deriv = np.full_like(data, np.nan)  # mj_20260311
    for s, e in segments:  # mj_20260311
        seg = data[s:e]
        n = len(seg)
        d = np.full(n, np.nan)
        if n >= 3:  # mj_20260311
            d[1:-1] = (seg[2:] - seg[:-2]) / (2 * dt)
            d[0] = (seg[1] - seg[0]) / dt
            d[-1] = (seg[-1] - seg[-2]) / dt
        elif n == 2:  # mj_20260311
            d[0] = d[1] = (seg[1] - seg[0]) / dt
        deriv[s:e] = d  # mj_20260311
    return deriv  # mj_20260311


def build_dsml_lookup(year):  # mj_20260311
    print(f'  Loading SuperMAG global data for {year}...')  # mj_20260311
    ts_arr, sml, smu = load_supermag_global_year(year)  # mj_20260311
    print(f'    {len(ts_arr)} data points')
    segments = find_segments(ts_arr, gap_sec=GAP_THRESH_SEC)  # mj_20260311
    sml_filt = lowpass_filter(sml, segments, FILTER_CUTOFF_MIN, FILTER_ORDER)  # mj_20260311
    dsml_filt = deriv_3pt(sml_filt, segments, dt=1.0)  # mj_20260311
    ts_sec = np.array([t.timestamp() for t in ts_arr])  # mj_20260311
    return ts_sec, dsml_filt  # mj_20260311


def query_dsml_dt_batch(sample_ts_sec_arr, lookup_ts_sec, dsml_filt):  # mj_20260311
    idx = np.searchsorted(lookup_ts_sec, sample_ts_sec_arr, side='left')  # mj_20260311
    idx = np.clip(idx, 0, len(lookup_ts_sec) - 1)  # mj_20260311
    idx_prev = np.clip(idx - 1, 0, len(lookup_ts_sec) - 1)  # mj_20260311
    dt_curr = np.abs(lookup_ts_sec[idx] - sample_ts_sec_arr)  # mj_20260311
    dt_prev = np.abs(lookup_ts_sec[idx_prev] - sample_ts_sec_arr)  # mj_20260311
    use_prev = dt_prev < dt_curr  # mj_20260311
    best_idx = np.where(use_prev, idx_prev, idx)  # mj_20260311
    best_dt = np.where(use_prev, dt_prev, dt_curr)  # mj_20260311
    vals = dsml_filt[best_idx]  # mj_20260311
    vals = np.where(best_dt <= 60.0, vals, 0.0)  # mj_20260311
    vals = np.where(np.isnan(vals), 0.0, vals)  # mj_20260311
    return vals  # mj_20260311


def load_forsyth_expansion_onsets(year):  # mj_20260311
    fpath = REPO / f'substorm_phases_{year}_ept{EPT}.txt'  # mj_20260311
    onsets = []  # mj_20260311
    with open(fpath) as f:  # mj_20260311
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2 and int(parts[1]) == 2:  # mj_20260311
                dt = datetime.strptime(parts[0], '%Y/%m/%d-%H:%M')  # mj_20260311
                onsets.append(dt)
    return sorted(onsets)  # mj_20260311


def compute_T1_T2_batch(ts_sec_arr, onsets_ts):  # mj_20260311
    idx = np.searchsorted(onsets_ts, ts_sec_arr, side='right')  # mj_20260311
    has_prev = idx > 0  # mj_20260311
    T1 = np.full(len(ts_sec_arr), DEFAULT_T_MIN)  # mj_20260311
    prev_idx = np.clip(idx - 1, 0, len(onsets_ts) - 1)  # mj_20260311
    T1[has_prev] = (ts_sec_arr[has_prev] - onsets_ts[prev_idx[has_prev]]) / 60.0  # mj_20260311
    has_next = idx < len(onsets_ts)  # mj_20260311
    T2 = np.full(len(ts_sec_arr), DEFAULT_T_MIN)  # mj_20260311
    next_idx = np.clip(idx, 0, len(onsets_ts) - 1)  # mj_20260311
    T2[has_next] = (onsets_ts[next_idx[has_next]] - ts_sec_arr[has_next]) / 60.0  # mj_20260311
    return T1, T2  # mj_20260311


def load_omni_newell_dict(years):  # mj_20260311
    from cdflib import CDF  # mj_20260311
    from cdflib.epochs import CDFepoch  # mj_20260311
    print('  Loading OMNI CDF for Newell coupling...')  # mj_20260311
    cdf = CDF(str(OMNI_CDF))  # mj_20260311
    epoch = cdf.varget('Epoch')  # mj_20260311
    by = cdf.varget('BY_GSM').astype(float)  # mj_20260311
    bz = cdf.varget('BZ_GSM').astype(float)  # mj_20260311
    n_p = cdf.varget('proton_density').astype(float)  # mj_20260311
    vx = cdf.varget('Vx').astype(float)  # mj_20260311
    vy = cdf.varget('Vy').astype(float)  # mj_20260311
    vz = cdf.varget('Vz').astype(float)  # mj_20260311
    times = CDFepoch.to_datetime(epoch)  # mj_20260311
    ts_pd = pd.DatetimeIndex(times).floor('min')  # mj_20260311
    v = np.sqrt(vx**2 + vy**2 + vz**2)  # mj_20260311
    BT = np.sqrt(by**2 + bz**2)  # mj_20260311
    bad = (v > 5000) | (BT > 500) | (n_p > 500) | (n_p <= 0) | (v <= 0)  # mj_20260311
    v[bad] = np.nan; BT[bad] = np.nan; n_p[bad] = np.nan  # mj_20260311
    newell = (v**(4.0/3.0)) * (BT**(2.0/3.0)) * (n_p**(1.0/6.0))  # mj_20260311
    newell = np.nan_to_num(newell, nan=0.0, posinf=0.0, neginf=0.0)  # mj_20260311
    newell_dict = {}  # mj_20260311
    for i, ts in enumerate(ts_pd):  # mj_20260311
        if ts.year in years:  # mj_20260311
            newell_dict[ts] = newell[i]  # mj_20260311
    print(f'  Loaded {len(newell_dict)} OMNI entries for years {years}')
    return newell_dict  # mj_20260311


# ============================================================
# Forsyth phase lookup and category classification
# ============================================================
def load_forsyth_phase_file(year):  # mj_20260311
    """Load Forsyth phase file and return entries list."""
    fpath = REPO / f'substorm_phases_{year}_ept{EPT}.txt'  # mj_20260311
    entries = []  # mj_20260311
    with open(fpath) as f:  # mj_20260311
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 3:  # mj_20260311
                dt = datetime.strptime(parts[0], '%Y/%m/%d-%H:%M')  # mj_20260311
                phase = int(parts[1])  # mj_20260311
                flag = int(parts[2])  # mj_20260311
                entries.append((dt, phase, flag))
    return entries  # mj_20260311


def build_phase_lookup(entries):  # mj_20260311
    """Build lookup arrays with phase, flag, exp_type, rec_type."""
    n = len(entries)  # mj_20260311
    starts = np.array([e[0].timestamp() for e in entries])  # mj_20260311
    phases = np.array([e[1] for e in entries])  # mj_20260311
    flags = np.array([e[2] for e in entries])  # mj_20260311

    exp_type = np.zeros(n, dtype=int)  # mj_20260311
    for i in range(n):  # mj_20260311
        if phases[i] != 2:  # mj_20260311
            continue
        prev_p = phases[i - 1] if i > 0 else 0  # mj_20260311
        next_p = phases[i + 1] if i < n - 1 else 0  # mj_20260311
        if prev_p == 1 and next_p == 3: exp_type[i] = 1  # mj_20260311
        elif prev_p == 1 and next_p == 1: exp_type[i] = 2  # mj_20260311
        elif prev_p == 3 and next_p == 3: exp_type[i] = 3  # mj_20260311
        elif prev_p == 3 and next_p == 1: exp_type[i] = 4  # mj_20260311

    rec_type = np.zeros(n, dtype=int)  # mj_20260311
    for i in range(n):  # mj_20260311
        if phases[i] != 3:  # mj_20260311
            continue
        prev_p = phases[i - 1] if i > 0 else 0  # mj_20260311
        next_p = phases[i + 1] if i < n - 1 else 0  # mj_20260311
        if prev_p == 2 and next_p == 1: rec_type[i] = 1  # mj_20260311
        elif prev_p == 2 and next_p == 2: rec_type[i] = 2  # mj_20260311
        elif prev_p == 1 and next_p == 1: rec_type[i] = 3  # mj_20260311
        elif prev_p == 1 and next_p == 2: rec_type[i] = 4  # mj_20260311

    return starts, phases, flags, exp_type, rec_type  # mj_20260311


def classify_category_batch(phase_arr, flag_arr, exp_type_arr, rec_type_arr):  # mj_20260311
    """Vectorized classification into 4 categories."""
    cat = np.ones(len(phase_arr), dtype=int)  # default: Cat1 growth  # mj_20260311
    is_exp = phase_arr == 2  # mj_20260311
    is_rec = phase_arr == 3  # mj_20260311
    unflagged_exp = is_exp & (flag_arr == 0)  # mj_20260311
    unflagged_rec = is_rec & (flag_arr == 0)  # mj_20260311
    flagged = (is_exp | is_rec) & (flag_arr == 1)  # mj_20260311

    # Cat2: compound  # mj_20260311
    compound_exp = unflagged_exp & ((exp_type_arr == 3) | (exp_type_arr == 4))  # mj_20260311
    compound_rec = unflagged_rec & ((rec_type_arr == 2) | (rec_type_arr == 4))  # mj_20260311
    cat[compound_exp | compound_rec] = 2  # mj_20260311

    # Cat3: isolated (unflagged exp/rec NOT compound)  # mj_20260311
    isolated_exp = unflagged_exp & ~compound_exp  # mj_20260311
    isolated_rec = unflagged_rec & ~compound_rec  # mj_20260311
    cat[isolated_exp | isolated_rec] = 3  # mj_20260311

    # Cat4: convection (flagged)  # mj_20260311
    cat[flagged] = 4  # mj_20260311
    return cat  # mj_20260311


def query_phase_batch(ts_sec_arr, starts, phases, flags, exp_type, rec_type):  # mj_20260311
    idx = np.searchsorted(starts, ts_sec_arr, side='right') - 1  # mj_20260311
    idx = np.clip(idx, 0, len(phases) - 1)  # mj_20260311
    return phases[idx], flags[idx], exp_type[idx], rec_type[idx]  # mj_20260311


# ============================================================
# Process a single condition file (add extra cols + category)
# ============================================================
def process_condition_file(cond_48d, timestamps_str, npy_paths,
                           dsml_lookups, newell_dict, all_onsets_ts,
                           phase_lookups, stats):  # mj_20260311
    """
    Given existing 48d z-scored SML/SMU conditions and timestamps,
    compute all 4 extra columns and Forsyth category.

    Returns: conditions_52d [N, 52], forsyth_cat [N]
    """
    n = len(timestamps_str)  # mj_20260311
    print(f'    Processing {n} samples...')

    # Parse timestamps  # mj_20260311
    datetimes = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S') if len(ts) == 19  # mj_20260311
                 else datetime.strptime(ts, '%Y-%m-%d %H:%M')
                 for ts in timestamps_str]
    ts_sec_arr = np.array([dt.timestamp() for dt in datetimes])  # mj_20260311
    file_years = np.array([dt.year for dt in datetimes])  # mj_20260311

    # ---- Newell coupling ---- # mj_20260311
    print(f'    Computing Newell...')
    newell_raw = np.zeros(n, dtype=np.float64)  # mj_20260311
    n_found = 0  # mj_20260311
    for i, dt in enumerate(datetimes):  # mj_20260311
        ts = pd.Timestamp(dt).floor('min')  # mj_20260311
        if ts in newell_dict:  # mj_20260311
            newell_raw[i] = newell_dict[ts]  # mj_20260311
            n_found += 1
    print(f'      Found {n_found}/{n} in OMNI dict ({100*n_found/n:.1f}%)')
    newell_z = ((newell_raw - stats['newell_mean']) / (stats['newell_std'] + 1e-6)).astype(np.float32)  # mj_20260311

    # ---- T1/T2 ---- # mj_20260311
    print(f'    Computing T1/T2...')
    T1_raw, T2_raw = compute_T1_T2_batch(ts_sec_arr, all_onsets_ts)  # mj_20260311
    T1_log = np.log10(1.0 + T1_raw)  # mj_20260311
    T2_log = np.log10(1.0 + T2_raw)  # mj_20260311
    T1_z = ((T1_log - stats['T1_mean']) / (stats['T1_std'] + 1e-6)).astype(np.float32)  # mj_20260311
    T2_z = ((T2_log - stats['T2_mean']) / (stats['T2_std'] + 1e-6)).astype(np.float32)  # mj_20260311

    # ---- dSML_filt/dt ---- # mj_20260311
    print(f'    Computing dSML_filt/dt...')
    dsml_vals = np.zeros(n, dtype=np.float64)  # mj_20260311
    for year in sorted(set(file_years)):  # mj_20260311
        yr_mask = file_years == year  # mj_20260311
        lookup_ts, lookup_dsml = dsml_lookups[year]  # mj_20260311
        dsml_vals[yr_mask] = query_dsml_dt_batch(  # mj_20260311
            ts_sec_arr[yr_mask], lookup_ts, lookup_dsml)
    dsml_z = ((dsml_vals - stats['dsml_mean']) / (stats['dsml_std'] + 1e-6)).astype(np.float32)  # mj_20260311

    # ---- Forsyth category ---- # mj_20260311
    print(f'    Classifying Forsyth categories...')
    forsyth_cat = np.ones(n, dtype=int)  # mj_20260311
    for year in sorted(set(file_years)):  # mj_20260311
        yr_mask = file_years == year  # mj_20260311
        starts, phases, flags, et, rt = phase_lookups[year]  # mj_20260311
        p_arr, f_arr, et_arr, rt_arr = query_phase_batch(  # mj_20260311
            ts_sec_arr[yr_mask], starts, phases, flags, et, rt)
        forsyth_cat[yr_mask] = classify_category_batch(p_arr, f_arr, et_arr, rt_arr)  # mj_20260311

    # Print category distribution  # mj_20260311
    for c in [1, 2, 3, 4]:
        nc = np.sum(forsyth_cat == c)  # mj_20260311
        print(f'      Cat{c}: {nc} ({100*nc/n:.1f}%)')

    # ---- Assemble 52d ---- # mj_20260311
    conditions_52d = np.concatenate([  # mj_20260311
        cond_48d.astype(np.float32),  # [0:48] SML/SMU
        newell_z[:, None],             # [48] Newell
        T1_z[:, None],                 # [49] T1
        T2_z[:, None],                 # [50] T2
        dsml_z[:, None],               # [51] dSML_filt/dt
    ], axis=1)

    return conditions_52d, forsyth_cat  # mj_20260311


# ============================================================
# Main
# ============================================================
def main():  # mj_20260311
    print('='*60)  # mj_20260311
    print('Generating universal 52-dim condition files for 4-cat sampling')
    print('='*60)

    # ----------------------------------------------------------
    # Load z-score stats (from training)
    # ----------------------------------------------------------
    print('\nLoading z-score stats...')  # mj_20260311
    s4 = np.load(DATA_ROOT / 'conditions_4cat_zscore_stats.npz')  # mj_20260311
    stats = {  # mj_20260311
        'newell_mean': float(s4['newell_mean']),
        'newell_std': float(s4['newell_std']),
        'T1_mean': float(s4['T1_mean']),
        'T1_std': float(s4['T1_std']),
        'T2_mean': float(s4['T2_mean']),
        'T2_std': float(s4['T2_std']),
        'dsml_mean': float(s4['dsml_mean']),
        'dsml_std': float(s4['dsml_std']),
    }
    for k, v in stats.items():
        print(f'  {k}: {v:.4f}')

    # ----------------------------------------------------------
    # Pre-compute dSML_filt/dt lookups
    # ----------------------------------------------------------
    print('\nPre-computing dSML_filt/dt lookups...')  # mj_20260311
    dsml_lookups = {}  # mj_20260311
    for year in SAMPLING_YEARS:  # mj_20260311
        dsml_lookups[year] = build_dsml_lookup(year)

    # ----------------------------------------------------------
    # Load OMNI Newell dict
    # ----------------------------------------------------------
    print('\nLoading OMNI Newell...')  # mj_20260311
    newell_dict = load_omni_newell_dict(SAMPLING_YEARS)  # mj_20260311

    # ----------------------------------------------------------
    # Load Forsyth expansion onsets (for T1/T2)
    # ----------------------------------------------------------
    print('\nLoading Forsyth expansion onsets...')  # mj_20260311
    all_onsets = []  # mj_20260311
    for year in range(1999, 2007):  # mj_20260311
        onsets = load_forsyth_expansion_onsets(year)  # mj_20260311
        print(f'  {year}: {len(onsets)} onsets')
        all_onsets.extend(onsets)
    all_onsets.sort()  # mj_20260311
    all_onsets_ts = np.array([o.timestamp() for o in all_onsets])  # mj_20260311
    print(f'  Total: {len(all_onsets_ts)} onsets')

    # ----------------------------------------------------------
    # Load Forsyth phase lookups (for category classification)
    # ----------------------------------------------------------
    print('\nLoading Forsyth phase lookups...')  # mj_20260311
    phase_lookups = {}  # mj_20260311
    for year in SAMPLING_YEARS:  # mj_20260311
        entries = load_forsyth_phase_file(year)  # mj_20260311
        phase_lookups[year] = build_phase_lookup(entries)  # mj_20260311
        print(f'  {year}: {len(entries)} phase entries')

    # ==========================================================
    # 1. Analysis workflow: 9 storm categories
    # ==========================================================
    print(f'\n{"="*60}')
    print('Processing ANALYSIS workflow (9 storm categories)')
    print(f'{"="*60}')

    samp_dir = DATA_ROOT / 'sampling_npy_joint'  # mj_20260311
    _only = os.environ.get('FOURCAT_PREFIX', '')  # mj_20260907: e.g. halloween -> process only that base file, skip climatology
    analysis_files = sorted(glob.glob(  # mj_20260311
        str(samp_dir / f'conditions_{_only or "*"}_minute_joint_9_zscore_SMU_SML.npz')))
    print(f'  Found {len(analysis_files)} category files')

    for fpath in analysis_files:  # mj_20260311
        fname = Path(fpath).name  # mj_20260311
        cat_name = fname.replace('conditions_', '').replace('_minute_joint_9_zscore_SMU_SML.npz', '')  # mj_20260311
        print(f'\n  --- {cat_name} ---')

        d = np.load(fpath, allow_pickle=True)  # mj_20260311
        cond_48d = d['conditions']  # already z-scored [N, 48]  # mj_20260311
        timestamps = d['timestamps']  # mj_20260311
        npy_paths = d['npy_paths']  # mj_20260311

        cond_52d, forsyth_cat = process_condition_file(  # mj_20260311
            cond_48d, timestamps, npy_paths,
            dsml_lookups, newell_dict, all_onsets_ts,
            phase_lookups, stats)

        # Save  # mj_20260311
        out_name = f'conditions_{cat_name}_4cat_52d.npz'  # mj_20260311
        out_path = samp_dir / out_name  # mj_20260311
        np.savez_compressed(out_path,  # mj_20260311
                            conditions=cond_52d,
                            npy_paths=npy_paths,
                            timestamps=timestamps,
                            forsyth_category=forsyth_cat,
                            col_names='SML[0:24]+SMU[24:48]+Newell[48]+T1[49]+T2[50]+dSML_dt[51]')
        print(f'    Saved: {out_path}')
        print(f'    Shape: {cond_52d.shape}')

    # ==========================================================
    # 2. Climatology workflow: SME-stratified
    # ==========================================================
    print(f'\n{"="*60}')
    print('Processing CLIMATOLOGY workflow (SME-stratified)')
    print(f'{"="*60}')

    if _only: print('FOURCAT_PREFIX set: skipping climatology workflow'); return  # mj_20260907
    clim_path = samp_dir / 'conditions_climatology_stratified_sme.npz'  # mj_20260311
    d = np.load(clim_path, allow_pickle=True)  # mj_20260311
    cond_48d = d['conditions']  # mj_20260311
    timestamps = d['timestamps']  # mj_20260311
    npy_paths = d['npy_paths']  # mj_20260311
    # Preserve all climatology metadata  # mj_20260311
    bin_sme = d['bin_sme']  # mj_20260311
    sme_bin_edges = d['sme_bin_edges']  # mj_20260311
    bin_counts = d['bin_counts']  # mj_20260311
    K_per_bin = d['K_per_bin']  # mj_20260311
    min_separation_min = d['min_separation_min']  # mj_20260311
    selected_indices = d['selected_indices']  # mj_20260311

    print(f'  {len(cond_48d)} samples, {len(np.unique(bin_sme))} SME bins')

    cond_52d, forsyth_cat = process_condition_file(  # mj_20260311
        cond_48d, timestamps, npy_paths,
        dsml_lookups, newell_dict, all_onsets_ts,
        phase_lookups, stats)

    out_path = samp_dir / 'conditions_climatology_stratified_sme_4cat_52d.npz'  # mj_20260311
    np.savez_compressed(out_path,  # mj_20260311
                        conditions=cond_52d,
                        npy_paths=npy_paths,
                        timestamps=timestamps,
                        forsyth_category=forsyth_cat,
                        bin_sme=bin_sme,
                        sme_bin_edges=sme_bin_edges,
                        bin_counts=bin_counts,
                        K_per_bin=K_per_bin,
                        min_separation_min=min_separation_min,
                        selected_indices=selected_indices,
                        col_names='SML[0:24]+SMU[24:48]+Newell[48]+T1[49]+T2[50]+dSML_dt[51]')
    print(f'  Saved: {out_path}')
    print(f'  Shape: {cond_52d.shape}')

    # ==========================================================
    # Verification
    # ==========================================================
    print(f'\n{"="*60}')
    print('Verification')
    print(f'{"="*60}')

    # Analysis  # mj_20260311
    for fpath in sorted(glob.glob(str(samp_dir / 'conditions_*_4cat_52d.npz'))):  # mj_20260311
        fname = Path(fpath).name  # mj_20260311
        d = np.load(fpath)  # mj_20260311
        c = d['conditions']  # mj_20260311
        fc = d['forsyth_category']  # mj_20260311
        cat_dist = {i: np.sum(fc == i) for i in [1, 2, 3, 4]}  # mj_20260311
        print(f'  {fname}: shape={c.shape}, cats={cat_dist}')
        # Check extra cols  # mj_20260311
        for col_i, col_name in [(48, 'Newell'), (49, 'T1'), (50, 'T2'), (51, 'dSML_dt')]:
            print(f'    col {col_i} ({col_name}): mean={c[:, col_i].mean():.4f}, '
                  f'std={c[:, col_i].std():.4f}')

    print('\nDone!')


if __name__ == '__main__':  # mj_20260311
    main()
