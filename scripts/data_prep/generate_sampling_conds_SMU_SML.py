#!/usr/bin/env python3
"""
Generate 9 storm-category sampling condition files (48-dim SMU/SML, z-scored).

Reads SuperMAG MLT data directly from /home/lim2/data/supermag/MLT/
to bypass the generate_condition_all_AU_AL dependency.

Output: 9 files in sampling_npy_joint/:
  conditions_{cat}_minute_joint_9_zscore_SMU_SML.npz
  (cats: before/after/during × super/strong/moderate)

mj_20260225
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


# ============================================================
# Paths
# ============================================================
SUPERMAG_DIR  = '/home/lim2/data/supermag/MLT'
STORM_CSV_2003 = '/home/lim2/code/amie_data_process/2003yr_storm.csv'
STORM_CSV_2000 = '/home/lim2/code/amie_data_process/2000yr_storm.csv'
SAMPLING_DIR   = '/home/lim2/data/amie_npy_joint_from105/sampling_npy_joint'
ZSCORE_STATS   = '/home/lim2/data/amie_npy_joint_from105/training_npy_joint/conditions_joint_9_zscore_stats_SMU_SML.npz'
GAP_VAL        = 999999   # SuperMAG missing-data sentinel

# ============================================================
# 1) Load SuperMAG MLT data (SML/SMU per 24 sectors, 1-min cadence)
# ============================================================
print("=" * 60)
print("Loading SuperMAG MLT sector data ...")
print("=" * 60)

SML_COLS = list(range(6, 30))    # SMLr00..SMLr23
SMU_COLS = list(range(30, 54))   # SMUr00..SMUr23

sm_rows = []
sm_skipped = 0
for fpath in sorted(os.path.join(SUPERMAG_DIR, f)
                    for f in os.listdir(SUPERMAG_DIR)
                    if f.endswith('-supermag.txt')):
    with open(fpath, 'r') as fh:
        for line in fh:
            s = line.strip()
            if s and s[0].isdigit():
                parts = s.split()
                if len(parts) == 54:
                    sm_rows.append(parts)
                else:
                    sm_skipped += 1

print(f"  Loaded {len(sm_rows)} rows, skipped {sm_skipped} malformed rows")
sm_arr = np.array(sm_rows, dtype=float)  # [N, 54]

# Build timestamp index
sm_ts = pd.to_datetime({
    'year':   sm_arr[:, 0].astype(int),
    'month':  sm_arr[:, 1].astype(int),
    'day':    sm_arr[:, 2].astype(int),
    'hour':   sm_arr[:, 3].astype(int),
    'minute': sm_arr[:, 4].astype(int),
    'second': sm_arr[:, 5].astype(int),
})
sm_ts_floor = pd.DatetimeIndex(sm_ts).floor('min')

sml_data = sm_arr[:, SML_COLS].copy()   # [N, 24]
smu_data = sm_arr[:, SMU_COLS].copy()   # [N, 24]
sml_data[sml_data >= GAP_VAL] = np.nan
smu_data[smu_data >= GAP_VAL] = np.nan

# Build dict: timestamp -> 48-dim raw condition [SML_r00..r23, SMU_r00..r23]
cond_dict_sm = {}
for i, ts in enumerate(sm_ts_floor):
    cond_dict_sm[ts] = np.concatenate([sml_data[i], smu_data[i]])  # [48]

print(f"  SM dict entries: {len(cond_dict_sm)}")

# ============================================================
# 2) Load z-score stats from training
# ============================================================
print("\nLoading z-score stats ...")
zscore_stats = np.load(ZSCORE_STATS)
mean_train = zscore_stats['mean']   # [48]
std_train  = zscore_stats['std']    # [48]
print(f"  mean shape: {mean_train.shape}, std shape: {std_train.shape}")

# ============================================================
# 3) Build storm categories (same random_state=42 as training block)
# ============================================================
print("\nBuilding storm categories ...")
storms_2003 = pd.read_csv(STORM_CSV_2003)
storms_2000 = pd.read_csv(STORM_CSV_2000)
for df in [storms_2003, storms_2000]:
    df['storm_class'] = df['storm_class'].str.strip()
    df['date']        = df['date'].str.strip()

PICK_PER_YEAR = 4
picked_all = pd.DataFrame()
for cls in ['super', 'strong', 'moderate']:
    for yr_label, yr_df in [('2003', storms_2003), ('2000', storms_2000)]:
        yr_c = yr_df[yr_df['storm_class'] == cls].copy()
        n_pick = min(PICK_PER_YEAR, len(yr_c))
        if n_pick > 0:
            picked = yr_c.sample(n=n_pick, random_state=42)
            picked_all = pd.concat([picked_all, picked], ignore_index=True)
            print(f"  {cls} ({yr_label}): picked {n_pick}")

storm_categories = {}
for _, row in picked_all.iterrows():
    storm_date  = datetime.strptime(row['date'], '%Y/%m/%d')
    storm_class = row['storm_class'].strip()
    for prefix, target_date in [
        ('before_', storm_date - timedelta(days=1)),
        ('',        storm_date),
        ('after_',  storm_date + timedelta(days=1)),
    ]:
        key = f"{prefix}{storm_class}_storm"
        storm_categories.setdefault(key, []).append(target_date)

print("\nStorm categories:")
for k in sorted(storm_categories.keys()):
    print(f"  {k}: {[d.strftime('%Y/%m/%d') for d in storm_categories[k]]}")

# ============================================================
# 4) Load sampling npy files and parse timestamps
# ============================================================
print("\nLoading sampling npy file list ...")
npy_files = sorted(
    os.path.join(SAMPLING_DIR, f)
    for f in os.listdir(SAMPLING_DIR)
    if f.lower().endswith('.npy')
)
print(f"  Found {len(npy_files)} npy files")
npy_files_arr = np.array(npy_files)  # mj_20260226

# Parse filename timestamps: YYYYMMDD_MMMM.npy -> datetime
def parse_npy_timestamp(fpath):
    stem  = os.path.basename(fpath)[:-4]   # e.g. '20000101_000'
    parts = stem.split('_')
    base  = datetime.strptime(parts[0], '%Y%m%d')
    mins  = int(parts[1])
    return base + timedelta(minutes=mins)

time_objects = np.array([parse_npy_timestamp(f) for f in npy_files])
time_obj_map = {t: i for i, t in enumerate(time_objects)}  # datetime -> file index

# Sort for category filtering
idx_sorted      = np.argsort(time_objects)
time_sorted     = time_objects[idx_sorted]

# ============================================================
# 5) Generate per-category condition files
# ============================================================
print("\n" + "=" * 60)
print("Generating per-category condition files ...")
print("=" * 60)

for cat_name in sorted(storm_categories.keys()):
    target_date_set = set(d.date() for d in storm_categories[cat_name])

    # Filter npy times to target dates
    filt_idx = np.array([i for i, t in enumerate(time_sorted)
                         if t.date() in target_date_set])
    if len(filt_idx) == 0:
        print(f"WARNING: No npy data for {cat_name}, skipping")
        continue

    time_filt = time_sorted[filt_idx]

    # 1-minute dedup
    seen_min = set()
    minute_idx = []
    for i, t in enumerate(time_filt):
        key = (t.year, t.month, t.day, t.hour, t.minute)
        if key not in seen_min:
            minute_idx.append(i)
            seen_min.add(key)
    time_1min = time_filt[minute_idx]

    # Match back to file indices and get conditions from SM dict
    conditions_list = []
    npy_paths_list = []    # mj_20260226
    timestamps_list = []   # mj_20260226
    n_missing_sm = 0
    for t in time_1min:
        t_floor = pd.Timestamp(t).floor('min')
        if t_floor in cond_dict_sm:
            raw_cond = cond_dict_sm[t_floor]   # [48] (raw nT, may have nan)
            conditions_list.append(raw_cond)
            npy_paths_list.append(npy_files_arr[time_obj_map[t]])   # mj_20260226
            timestamps_list.append(str(t))                           # mj_20260226
        else:
            n_missing_sm += 1

    if len(conditions_list) == 0:
        print(f"WARNING: No SM conditions for {cat_name}, skipping")
        continue

    conditions_raw = np.array(conditions_list, dtype=np.float32)   # [N, 48]

    # Fill NaN sectors with column mean (from training data mean)
    # (column mean is 0 after z-scoring, so fill-before-zscore with train mean)
    nan_mask = np.isnan(conditions_raw)
    n_nan = nan_mask.sum()
    if n_nan > 0:
        # Replace NaN with training mean (raw scale) before z-scoring
        # After z-scoring, this becomes 0 which is neutral
        for col_idx in range(48):
            col_nans = nan_mask[:, col_idx]
            if col_nans.any():
                # fill with value that z-scores to 0 = training mean
                conditions_raw[col_nans, col_idx] = mean_train[col_idx]

    # Z-score using training stats
    z_conditions = (conditions_raw - mean_train) / (std_train + 1e-6)

    out_path = os.path.join(
        SAMPLING_DIR,
        f'conditions_{cat_name}_minute_joint_9_zscore_SMU_SML.npz'
    )
    np.savez(out_path, conditions=z_conditions,                     # mj_20260226
             npy_paths=np.array(npy_paths_list),                   # mj_20260226
             timestamps=np.array(timestamps_list))                  # mj_20260226
    print(f"  {cat_name}: {len(conditions_list)} conds "
          f"(SM missing: {n_missing_sm}, NaN filled: {n_nan}) "
          f"-> {os.path.basename(out_path)}")

print("\nDone! 9 SMU/SML condition files generated.")
