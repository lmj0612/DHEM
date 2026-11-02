"""
Generate OMNI condition data for AMIE model training.

This script:
1. Loads 1-minute OMNI data (solar wind, IMF, geomagnetic indices)
2. Loads F10.7 solar flux data
3. Filters for valid data within specified date range
4. Computes derived quantities (Newell coupling, dynamic pressure, clock angle)
5. Creates condition arrays for different training modes

Output: Dictionary mapping filename timestamps to condition vectors
"""

#%%
import numpy as np
import os
from spacepy import pycdf
import pandas as pd
import datetime as dt
#%%

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date range for data extraction
START_DATE = "1999-01-01 00:00:00"
END_DATE = "2007-01-01 00:00:00"

# Input file paths
OMNI_1MIN_FILE = '/home/lim2/data/CDAW/omni_hro2s_1min_19970101000000_20110101000000_cdaweb.cdf'
OMNI_AE_FILE = OMNI_1MIN_FILE
OMNI_F107_FILE = '/home/lim2/data/CDAW/omni2_h0s_mrg1hr_19961231143000_20110301143000_cdaweb.cdf'

# Output file path
OUTPUT_FILE = '/home/lim2/data/amie_npy_joint_from105/1999_2010_condition/keep_dict_joint.npz'

# Interpolation settings
ENABLE_INTERPOLATION = True      # Enable gap filling via interpolation
MAX_GAP_MINUTES = 5              # Only interpolate gaps ≤ this many minutes

# Solar wind propagation delay settings
"""
IMPORTANT: Solar wind propagation time from L1 (~235 Re upstream) to ionosphere

Physical considerations:
- L1 to bow shock: ~30-60 min (depends on solar wind speed ~400-800 km/s)
- Magnetospheric response: ~5-15 min additional lag
- Total lag: typically 30-60 minutes, can be variable

Options:
1. APPLY_TIME_SHIFT = True (30 min):
   - Physically motivated preprocessing
   - Reduces model burden
   - Assumes fixed lag (may not be optimal for all conditions)
   - AMIE at time t corresponds to OMNI at t-30min

2. APPLY_TIME_SHIFT = False:
   - Let temporal encoder learn optimal lag from 60-min history
   - More flexible (can learn variable lags for different conditions)
   - Model must learn this relationship (more complex)
   - Recommended if using temporal encoder with sufficient history

Recommendation: 
- With 60-min temporal history: Set to False (let model learn)
- Without temporal encoder: Set to True (apply fixed 30-min shift)
"""
APPLY_TIME_SHIFT = False          # Apply 30-minute propagation delay
TIME_SHIFT_MINUTES = 30          # Propagation time (if enabled)

# Training mode selection (set one to True)
TRAINING_MODES = {
    'raw_solar_wind': False,
    'flux_training': False,
    'potential_training': False,
    'energy_training': False,
    'joint_training': False,  # Currently active
    'AU_AL_training': True,
    'time_only': False,
    'all_parameters': False,
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def newell_and_pdyn(bx, by, bz, n_cm3, vx, vy, vz, he_ratio=None):
    """
    Compute Newell coupling (dimensionless) and solar-wind dynamic pressure (nPa).

    Inputs (scalars or NumPy arrays, GSM coords):
      bx, by, bz : IMF components [nT]
      n_cm3      : proton number density [cm^-3]
      vx, vy, vz : solar-wind velocity components [km/s]
      he_ratio   : optional alpha/proton ratio (e.g., 0.04). If None, no He correction.

    Returns:
      newell : v^(4/3) * BT^(2/3) * n^(1/6) (dimensionless)
      pdyn   : dynamic pressure [nPa]
    """
    # Speed magnitude [km/s]
    v = np.sqrt(np.asarray(vx)**2 + np.asarray(vy)**2 + np.asarray(vz)**2)

    # Transverse IMF [nT]
    BT = np.sqrt(np.asarray(by)**2 + np.asarray(bz)**2)

    # Newell coupling (dimensionless)
    newell = (v**(4.0/3.0)) * (BT**(2.0/3.0)) * (n_cm3**(1.0/6.0))

    # Dynamic pressure [nPa], proton-only
    pdyn = 1.6726219e-6 * np.asarray(n_cm3) * (v**2)  # n in cm^-3, v in km/s

    # Optional He++ correction
    if he_ratio is not None:
        pdyn = pdyn * (1.0 + 4.0*he_ratio)

    # Clean NaN/Inf
    newell = np.nan_to_num(newell, nan=0.0, posinf=0.0, neginf=0.0)
    pdyn = np.nan_to_num(pdyn, nan=0.0, posinf=0.0, neginf=0.0)

    return newell, pdyn


def compute_dipole_tilt(datetimes_array, resolution_minutes=30, chunk_size=5000):
    """
    Compute dipole tilt angle (PS angle, in degrees) for an array of datetimes.

    Uses IGRF dipole axis direction + GEO->GSM coordinate transformation.
    Computes at coarse resolution and interpolates to 1-min for efficiency.

    The dipole tilt angle (psi) is the angle between Earth's magnetic dipole
    axis and the GSM Z-axis. In GSM coordinates the dipole lies in the XZ
    plane, so psi = arcsin(dipole_x_GSM / |dipole_GSM|).

    Args:
        datetimes_array: Array/list of datetime objects
        resolution_minutes: Coarse grid resolution for efficiency (default: 30 min)
        chunk_size: Number of time points to process per batch

    Returns:
        tilt_angles: Array of dipole tilt angles in degrees
    """
    from spacepy.igrf import IGRF
    from spacepy.time import Ticktock
    from spacepy import coordinates as coord
    from scipy.interpolate import interp1d

    print(f"  Computing dipole tilt angle using IGRF + GSM transformation...")
    print(f"  Coarse resolution: {resolution_minutes} min, then interpolating to 1-min")

    # Convert to pandas DatetimeIndex for easier manipulation
    ts_pd = pd.DatetimeIndex(datetimes_array)
    t_start = ts_pd[0]
    t_end = ts_pd[-1]

    # Create coarse time grid (tilt varies smoothly, no need for 1-min resolution)
    coarse_times = pd.date_range(start=t_start, end=t_end, freq=f'{resolution_minutes}min')
    n_coarse = len(coarse_times)
    print(f"  Coarse grid: {n_coarse} points ({t_start} to {t_end})")

    # Get IGRF dipole axis (compute once - barely changes over a few years)
    model = IGRF()
    mid_time = coarse_times[n_coarse // 2].to_pydatetime()
    model.initialize(mid_time)

    colat_rad = model.dipole['cd_gcolat_rad']
    lon_rad = model.dipole['cd_glon_rad']

    # Dipole axis unit vector in GEO Cartesian coordinates
    dipole_x = np.sin(colat_rad) * np.cos(lon_rad)
    dipole_y = np.sin(colat_rad) * np.sin(lon_rad)
    dipole_z = np.cos(colat_rad)
    print(f"  Dipole axis (GEO cart): [{dipole_x:.4f}, {dipole_y:.4f}, {dipole_z:.4f}]")
    print(f"  Dipole colatitude: {np.rad2deg(colat_rad):.2f} deg, longitude: {np.rad2deg(lon_rad):.2f} deg")

    # Compute tilt at coarse resolution in chunks
    coarse_tilts = np.full(n_coarse, np.nan)

    for start_idx in range(0, n_coarse, chunk_size):
        end_idx = min(start_idx + chunk_size, n_coarse)
        chunk_n = end_idx - start_idx

        # Create Ticktock for this chunk
        chunk_datetimes = [t.to_pydatetime() for t in coarse_times[start_idx:end_idx]]
        tt = Ticktock(chunk_datetimes, 'UTC')

        # Dipole axis repeated for each time point (same vector, different times)
        positions = np.tile([dipole_x, dipole_y, dipole_z], (chunk_n, 1))
        dipole_geo = coord.Coords(positions, 'GEO', 'car')
        dipole_geo.ticks = tt

        # Convert dipole axis from GEO to GSM
        dipole_gsm = dipole_geo.convert('GSM', 'car')

        # Dipole tilt: psi = arcsin(dipole_x_GSM / |dipole_GSM|)
        # In GSM, dipole lies in XZ plane: dipole_GSM = (sin(psi), 0, cos(psi))
        dx = np.array(dipole_gsm.x)
        dy = np.array(dipole_gsm.y)
        dz = np.array(dipole_gsm.z)
        norm = np.sqrt(dx**2 + dy**2 + dz**2)
        coarse_tilts[start_idx:end_idx] = np.rad2deg(
            np.arcsin(np.clip(dx / norm, -1, 1))
        )

        if (start_idx // chunk_size) % 5 == 0 or end_idx == n_coarse:
            print(f"    Processed {end_idx}/{n_coarse} coarse points...")

    # Interpolate to original 1-minute timestamps
    coarse_seconds = (coarse_times - t_start).total_seconds().values.astype(float)
    orig_seconds = (ts_pd - t_start).total_seconds().values.astype(float)

    interp_func = interp1d(coarse_seconds, coarse_tilts, kind='linear',
                           fill_value='extrapolate', bounds_error=False)
    tilt_angles = interp_func(orig_seconds)

    print(f"  Dipole tilt range: [{tilt_angles.min():.2f} deg, {tilt_angles.max():.2f} deg]")

    return tilt_angles


def load_and_filter_cdf(filepath, start_date, end_date, quality_filters):
    """
    Load CDF file and apply date range and quality filters.
    
    IMPORTANT: This function now uses per-parameter NaN masking instead of
    excluding entire timesteps. This preserves valid data and allows better
    interpolation downstream.
    
    Args:
        filepath: Path to CDF file
        start_date: Start date string
        end_date: End date string
        quality_filters: Dictionary of {variable_name: max_valid_value}
    
    Returns:
        cdf_data: Loaded CDF data (with invalid values replaced by NaN)
        flag: Boolean mask for date range only
        timestamps: Pandas datetime array
    """
    cdf = pycdf.CDF(filepath)
    cdf_data = cdf.copy()
    cdf.close()
    
    timestamps = pd.to_datetime(cdf_data['Epoch'])
    
    # Apply date range filter
    sta = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    time_diff_start = np.abs([date - sta for date in timestamps])
    sta_ind = time_diff_start.argmin()
    
    time_diff_end = np.abs([date - end for date in timestamps])
    end_ind = time_diff_end.argmin()
    
    print(f"Date range: {timestamps[sta_ind]} to {timestamps[end_ind]}")
    
    flag_date = np.zeros(len(timestamps), dtype=bool)
    flag_date[sta_ind:end_ind+1] = True
    
    # Apply quality filters PER PARAMETER (set invalid values to NaN)
    # This preserves valid data at each timestep instead of throwing away entire rows
    print("Applying per-parameter quality filters...")
    for var_name, max_val in quality_filters.items():
        invalid_mask = cdf_data[var_name] > max_val
        n_invalid = np.sum(invalid_mask & flag_date)
        if n_invalid > 0:
            cdf_data[var_name] = np.where(invalid_mask, np.nan, cdf_data[var_name])
            print(f"  {var_name}: {n_invalid} invalid values → NaN")
    
    # Return date filter only (not quality filter, as we use NaN now)
    return cdf_data, flag_date, timestamps


def interpolate_gaps(timestamps, data_arrays, max_gap_minutes=5):
    """
    Efficiently interpolate small gaps in 1-minute cadence OMNI data.
    
    Now handles NaN values properly:
    - Interpolates timestamp gaps (missing rows)
    - Also interpolates NaN values if they're in small gaps
    - Only interpolates if gap ≤ max_gap_minutes
    
    Args:
        timestamps: Pandas datetime array (should be filtered data only)
        data_arrays: Dictionary of {name: numpy_array} to interpolate (may contain NaN)
        max_gap_minutes: Maximum gap size to interpolate (default: 5 minutes)
    
    Returns:
        interpolated_data: Dictionary of interpolated arrays
        n_interpolated: Number of data points interpolated
        n_gaps_filled: Number of gaps filled
    """
    from scipy.interpolate import interp1d
    
    print(f"\nInterpolating gaps ≤ {max_gap_minutes} minutes...")
    
    # Create complete 1-minute timeline
    start_time = timestamps[0]
    end_time = timestamps[-1]
    complete_timeline = pd.date_range(start=start_time, end=end_time, freq='1T')
    
    # Convert to minutes since start (for easier calculation)
    timestamps_minutes = (timestamps - start_time).total_seconds() / 60.0
    complete_minutes = (complete_timeline - start_time).total_seconds() / 60.0
    
    # Find gaps in original timestamps
    gaps = []
    prev_minute = timestamps_minutes[0]
    for i in range(1, len(timestamps_minutes)):
        curr_minute = timestamps_minutes[i]
        gap_size = curr_minute - prev_minute - 1  # -1 because consecutive = no gap
        
        if gap_size > 0:
            gaps.append({
                'start_idx': i - 1,
                'end_idx': i,
                'gap_minutes': gap_size
            })
        
        prev_minute = curr_minute
    
    print(f"  Found {len(gaps)} timestamp gaps")
    
    # Separate small gaps (can interpolate) from large gaps (keep as-is)
    small_gaps = [g for g in gaps if g['gap_minutes'] <= max_gap_minutes]
    large_gaps = [g for g in gaps if g['gap_minutes'] > max_gap_minutes]
    
    print(f"  Small timestamp gaps (≤{max_gap_minutes} min): {len(small_gaps)}")
    print(f"  Large timestamp gaps (>{max_gap_minutes} min): {len(large_gaps)} (will NOT interpolate)")
    
    # For each variable, interpolate
    interpolated_data = {}
    total_points_added = 0
    
    for var_name, data_array in data_arrays.items():
        # Check for NaN values
        nan_mask = np.isnan(data_array)
        n_nans = np.sum(nan_mask)
        
        if n_nans > 0:
            print(f"  {var_name}: {n_nans} NaN values found")
        
        # Only use valid (non-NaN) points for interpolation
        valid_mask = ~nan_mask
        
        if np.sum(valid_mask) < 2:
            # Not enough valid points to interpolate
            print(f"  {var_name}: Too few valid points, keeping as-is")
            interpolated_data[var_name] = data_array
            continue
        
        # Create interpolator using only valid points
        interpolator = interp1d(
            timestamps_minutes[valid_mask], 
            data_array[valid_mask], 
            kind='linear',
            bounds_error=False,
            fill_value=np.nan  # Use NaN for extrapolation
        )
        
        # Build complete timeline with interpolated values
        if len(small_gaps) > 0:
            new_timestamps = [timestamps_minutes[0]]
            new_values = [data_array[0]]
            
            for i in range(1, len(timestamps_minutes)):
                prev_minute = timestamps_minutes[i-1]
                curr_minute = timestamps_minutes[i]
                gap_size = curr_minute - prev_minute - 1
                
                # If there's a small gap in timestamps, add interpolated points
                if gap_size > 0 and gap_size <= max_gap_minutes:
                    missing_minutes = np.arange(prev_minute + 1, curr_minute, 1)
                    # Only interpolate if both endpoints are valid
                    if valid_mask[i-1] and valid_mask[i]:
                        interpolated_vals = interpolator(missing_minutes)
                    else:
                        interpolated_vals = np.full(len(missing_minutes), np.nan)
                    
                    new_timestamps.extend(missing_minutes)
                    new_values.extend(interpolated_vals)
                
                # Add current point
                new_timestamps.append(curr_minute)
                new_values.append(data_array[i])
            
            interpolated_array = np.array(new_values)
            
            # Also interpolate NaN values within the data (not just timestamp gaps)
            nan_indices = np.where(np.isnan(interpolated_array))[0]
            if len(nan_indices) > 0:
                # Only interpolate NaN values that are in small gaps
                for nan_idx in nan_indices:
                    # Find nearest valid points
                    left_valid = None
                    right_valid = None
                    
                    # Search left
                    for j in range(nan_idx - 1, -1, -1):
                        if not np.isnan(interpolated_array[j]):
                            left_valid = j
                            break
                    
                    # Search right
                    for j in range(nan_idx + 1, len(interpolated_array)):
                        if not np.isnan(interpolated_array[j]):
                            right_valid = j
                            break
                    
                    # Interpolate if gap is small enough
                    if left_valid is not None and right_valid is not None:
                        gap_length = right_valid - left_valid - 1
                        if gap_length <= max_gap_minutes:
                            # Linear interpolation
                            weight = (nan_idx - left_valid) / (right_valid - left_valid)
                            interpolated_array[nan_idx] = (
                                (1 - weight) * interpolated_array[left_valid] +
                                weight * interpolated_array[right_valid]
                            )
            
            total_points_added = max(total_points_added, len(interpolated_array) - len(data_array))
        else:
            # No timestamp gaps, but may still have NaN values to interpolate
            interpolated_array = data_array.copy()
            
            # Interpolate NaN values
            if n_nans > 0:
                complete_minutes_orig = timestamps_minutes
                interpolated_array = interpolator(complete_minutes_orig)
                # Restore NaNs that are in large gaps
                interpolated_array[nan_mask] = data_array[nan_mask]
        
        interpolated_data[var_name] = interpolated_array
    
    if len(small_gaps) > 0:
        print(f"  Interpolated {total_points_added} timestamp gaps")
        
        # Return new timestamps as well
        new_timestamps = [timestamps[0]]
        for i in range(1, len(timestamps)):
            prev_minute = timestamps_minutes[i-1]
            curr_minute = timestamps_minutes[i]
            gap_size = curr_minute - prev_minute - 1
            
            if gap_size > 0 and gap_size <= max_gap_minutes:
                for j in range(1, int(gap_size) + 1):
                    new_timestamps.append(timestamps[i-1] + pd.Timedelta(minutes=j))
            
            new_timestamps.append(timestamps[i])
        
        interpolated_data['timestamps'] = pd.DatetimeIndex(new_timestamps)
    else:
        interpolated_data['timestamps'] = timestamps
    
    return interpolated_data, total_points_added, len(small_gaps)


# ============================================================================
# STEP 1: LOAD 1-MINUTE OMNI DATA (SOLAR WIND + GEOMAGNETIC INDICES)
# ============================================================================

print("=" * 70)
print("STEP 1: Loading 1-minute OMNI data...")
print("=" * 70)

# Load main OMNI data
quality_filters_omni = {
    'BX_GSE': 9999,
    'BY_GSM': 9999,
    'BZ_GSM': 9999,
    'Vx': 99999,
    'Vy': 99999,
    'Vz': 99999,
    'proton_density': 999,
    'SYM_H': 90000,
    'ASY_H': 9999,
    # 'PC_N_INDEX': 999,
}

cdf_dat, flag, tmp_dt = load_and_filter_cdf(
    OMNI_1MIN_FILE, 
    START_DATE, 
    END_DATE, 
    quality_filters_omni
)

# Extract filtered variables
bx_all = cdf_dat['BX_GSE'][flag]
by_all = cdf_dat['BY_GSM'][flag]
bz_all = cdf_dat['BZ_GSM'][flag]
vx_all = cdf_dat['Vx'][flag]
vy_all = cdf_dat['Vy'][flag]
vz_all = cdf_dat['Vz'][flag]
Np_all = cdf_dat['proton_density'][flag]
sym_h_all = cdf_dat['SYM_H'][flag]
asy_h_all = cdf_dat['ASY_H'][flag]

# Load PCI from the original omni_hros file (not available in omni_hro2s)
# OMNI_PCI_FILE = '/home/lim2/data/CDAW/omni_hros_1min_19970101000000_20110101000000_cdaweb.cdf'
# cdf_pci = pycdf.CDF(OMNI_PCI_FILE)
# cdf_dat_pci = cdf_pci.copy()
# cdf_pci.close()
# pci_epoch = pd.to_datetime(cdf_dat_pci['Epoch'])
# assert len(pci_epoch) == len(tmp_dt), "PCI file length mismatch"
# pci_raw = cdf_dat_pci['PC_N_INDEX'][flag]
# pci_all = np.where(pci_raw > 999, np.nan, pci_raw)


# Load AE/AL indices from separate file
cdf_ae = pycdf.CDF(OMNI_AE_FILE)
cdf_dat_ae = cdf_ae.copy()
cdf_ae.close()

au_all = cdf_dat_ae['AU_INDEX'][flag]
ae_all = cdf_dat_ae['AE_INDEX'][flag]
al_all = cdf_dat_ae['AL_INDEX'][flag]

# Store timestamps before interpolation
timestamps_original = tmp_dt[flag]

# Save original data (before interpolation) for gap visualization
bx_original = bx_all.copy()
by_original = by_all.copy()
bz_original = bz_all.copy()
vx_original = vx_all.copy()
vy_original = vy_all.copy()
vz_original = vz_all.copy()
Np_original = Np_all.copy()
sym_h_original = sym_h_all.copy()
asy_h_original = asy_h_all.copy()
# pci_original = pci_all.copy()
au_original = au_all.copy()
ae_original = ae_all.copy()
al_original = al_all.copy()

# Interpolate small gaps if enabled
if ENABLE_INTERPOLATION:
    data_to_interpolate = {
        'bx': bx_all,
        'by': by_all,
        'bz': bz_all,
        'vx': vx_all,
        'vy': vy_all,
        'vz': vz_all,
        'Np': Np_all,
        'sym_h': sym_h_all,
        'asy_h': asy_h_all,
        # 'pci': pci_all,
        'au':au_all,
        'ae': ae_all,
        'al': al_all,
    }
    
    interpolated_data, n_points, n_gaps = interpolate_gaps(
        timestamps_original,
        data_to_interpolate,
        max_gap_minutes=MAX_GAP_MINUTES
    )
    
    # Update arrays with interpolated data
    bx_all = interpolated_data['bx']
    by_all = interpolated_data['by']
    bz_all = interpolated_data['bz']
    vx_all = interpolated_data['vx']
    vy_all = interpolated_data['vy']
    vz_all = interpolated_data['vz']
    Np_all = interpolated_data['Np']
    sym_h_all = interpolated_data['sym_h']
    asy_h_all = interpolated_data['asy_h']
    # pci_all = interpolated_data['pci']
    au_all = interpolated_data['au']
    ae_all = interpolated_data['ae']
    al_all = interpolated_data['al']
    
    # Update timestamps
    timestamps_current = interpolated_data['timestamps']
    
    # Remove any remaining NaN values (large gaps that couldn't be interpolated)
    print("\nRemoving timesteps with remaining NaN values...")
    valid_mask = np.ones(len(timestamps_current), dtype=bool)
    
    # for var_name in ['bx', 'by', 'bz', 'vx', 'vy', 'vz', 'Np', 'sym_h', 'asy_h', 'pci', 'ae', 'al']:
    for var_name in ['bx', 'by', 'bz', 'vx', 'vy', 'vz', 'Np', 'sym_h', 'asy_h', 'ae', 'al']:
        nan_mask = np.isnan(interpolated_data[var_name])
        n_nans = np.sum(nan_mask)
        if n_nans > 0:
            print(f"  {var_name}: {n_nans} NaN values (large gaps)")
        valid_mask &= ~nan_mask
    
    n_removed = np.sum(~valid_mask)
    if n_removed > 0:
        print(f"  Removing {n_removed} timesteps with NaN values")
        
        bx_all = bx_all[valid_mask]
        by_all = by_all[valid_mask]
        bz_all = bz_all[valid_mask]
        vx_all = vx_all[valid_mask]
        vy_all = vy_all[valid_mask]
        vz_all = vz_all[valid_mask]
        Np_all = Np_all[valid_mask]
        sym_h_all = sym_h_all[valid_mask]
        asy_h_all = asy_h_all[valid_mask]
        # pci_all = pci_all[valid_mask]
        au_all = au_all[valid_mask]
        ae_all = ae_all[valid_mask]
        al_all = al_all[valid_mask]
        timestamps_current = timestamps_current[valid_mask]
    else:
        print("  No NaN values remaining - all gaps successfully interpolated!")
        
else:
    timestamps_current = timestamps_original
    print("\nInterpolation disabled - using original data with gaps")


# ============================================================================
# STEP 1.5: PLOT ORIGINAL DATA TO VISUALIZE GAPS (OPTIONAL)
# ============================================================================

if_plot_gaps = True  # Set to True to visualize data gaps before interpolation

if if_plot_gaps:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    print("\n" + "=" * 70)
    print("STEP 1.5: Plotting original data to visualize gaps...")
    print("=" * 70)
    
    # Define storm events (same as before)
    DATE_RANGES = [
        ((2000, 4, 6), (2000, 4, 7)),
        ((2000, 7, 13), (2000, 7, 17)),
        ((2003, 10, 28), (2003, 11, 1)),
        ((2003, 11, 19), (2003, 11, 21)),
        ((2003, 3, 30), (2003, 4, 2)),
        ((2003, 5, 28), (2003, 6, 1)),
    ]
    
    # Plot for each storm event
    for event_idx, ((y1, m1, d1), (y2, m2, d2)) in enumerate(DATE_RANGES):
        start_date = dt.datetime(y1, m1, d1)
        end_date = dt.datetime(y2, m2, d2)
        
        print(f"\nPlotting event {event_idx + 1}: {start_date.date()} to {end_date.date()}")
        
        # Filter data for this date range
        mask = (timestamps_original >= start_date) & (timestamps_original <= end_date)
        
        if not np.any(mask):
            print(f"  No data found for this period, skipping...")
            continue
        
        times_orig = timestamps_original[mask]
        bz_orig = bz_original[mask]
        by_orig = by_original[mask]
        bx_orig = bx_original[mask]
        vx_orig = vx_original[mask]
        vy_orig = vy_original[mask]
        vz_orig = vz_original[mask]
        np_orig = Np_original[mask]
        symh_orig = sym_h_original[mask]
        asyh_orig = asy_h_original[mask]
        al_orig = al_original[mask]
        ae_orig = ae_original[mask]
        # pci_orig = pci_original[mask]
        
        # Create figure with 12 subplots (all raw parameters)
        fig, axes = plt.subplots(12, 1, figsize=(14, 20), sharex=True)
        plt.subplots_adjust(hspace=0.05)
        
        # Use markers to show individual data points - gaps will be visible
        marker_style = dict(marker='o', markersize=2, linestyle='-', linewidth=0.5, alpha=0.7)
        
        # (a) BX
        ax = axes[0]
        ax.plot(times_orig, bx_orig, color='#264653', **marker_style)
        ax.set_ylabel('BX [nT]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(a)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (b) BY
        ax = axes[1]
        ax.plot(times_orig, by_orig, color='#005f73', **marker_style)
        ax.set_ylabel('BY [nT]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(b)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (c) BZ
        ax = axes[2]
        ax.plot(times_orig, bz_orig, color='#e76f51', **marker_style)
        ax.set_ylabel('BZ [nT]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(c)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (d) VX
        ax = axes[3]
        ax.plot(times_orig, vx_orig, color='#2a9d8f', **marker_style)
        ax.set_ylabel('VX [km/s]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(d)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        # (e) VY
        ax = axes[4]
        ax.plot(times_orig, vy_orig, color='#e9c46a', **marker_style)
        ax.set_ylabel('VY [km/s]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(e)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (f) VZ
        ax = axes[5]
        ax.plot(times_orig, vz_orig, color='#f4a261', **marker_style)
        ax.set_ylabel('VZ [km/s]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(f)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (g) Proton Density
        ax = axes[6]
        ax.plot(times_orig, np_orig, color='#e63946', **marker_style)
        ax.set_ylabel('Np [cm⁻³]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(g)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        # (h) SYM-H
        ax = axes[7]
        ax.plot(times_orig, symh_orig, color='#9467BD', **marker_style)
        ax.set_ylabel('SYM-H [nT]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(h)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (i) ASY-H
        ax = axes[8]
        ax.plot(times_orig, asyh_orig, color='#457b9d', **marker_style)
        ax.set_ylabel('ASY-H [nT]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(i)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        # (j) PCI
        # ax = axes[9]
        # ax.plot(times_orig, pci_orig, color='#06aed5', **marker_style)
        # ax.set_ylabel('PCI', fontsize=16)
        # ax.grid(True, alpha=0.3)
        # ax.text(0.02, 0.85, '(j)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        # (k) AL
        ax = axes[10]
        ax.plot(times_orig, al_orig, color='#0077b6', **marker_style)
        ax.set_ylabel('AL [nT]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(k)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        # (l) AE
        ax = axes[11]
        ax.plot(times_orig, ae_orig, color='#bc6c25', **marker_style)
        ax.set_ylabel('AE [nT]', fontsize=16)
        ax.set_xlabel('Time [UTC]', fontsize=16)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(l)', transform=ax.transAxes, fontsize=16, fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        # Set tick parameters for all axes
        for ax in axes:
            ax.tick_params(labelsize=14)
        
        # Add title
        fig.suptitle(f'Original OMNI Data (Before Interpolation): {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                    fontsize=20, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save figure
        save_dir = 'home/lim2/code/ amie_data_process/plots_hro2s/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 
                                f'omni_original_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    print("\nGap visualization complete!")


# Apply solar wind propagation time delay if enabled
if APPLY_TIME_SHIFT:
    print(f"\nApplying {TIME_SHIFT_MINUTES}-minute propagation delay...")
    print(f"  OMNI data at time t will correspond to AMIE at time t+{TIME_SHIFT_MINUTES}min")
    # Shift timestamps forward (AMIE time = OMNI time + delay)
    timestamps_for_filenames = timestamps_current + pd.Timedelta(minutes=TIME_SHIFT_MINUTES)
else:
    timestamps_for_filenames = timestamps_current
    print("\nNo time shift applied - OMNI and AMIE times are synchronous")

# Generate filenames for each timestamp (format: YYYYMMDD_MMMM.npy)
f_name_all = np.array([
    f"{dt.strftime('%Y%m%d')}_{dt.hour * 60 + dt.minute}" 
    for dt in timestamps_for_filenames
])

# Print final data statistics
print("\n" + "=" * 70)
print("FINAL DATA STATISTICS (after interpolation and time shift)")
print("=" * 70)
print(f"Total timesteps: {len(f_name_all)}")

if len(f_name_all) == 0:
    print("\n" + "=" * 70)
    print("ERROR: No timesteps remaining after filtering!")
    print("=" * 70)
    print("\nAll data was removed. Check:")
    print("1. Date range settings")
    print("2. Quality filter thresholds")
    print("3. NaN removal step")
    print("\nExiting...")
    import sys
    sys.exit(1)

print(f"Date range: {timestamps_for_filenames[0]} to {timestamps_for_filenames[-1]}")
print("\nVariable ranges:")
print(f"  BX:    [{bx_all.min():.2f}, {bx_all.max():.2f}] nT")
print(f"  BY:    [{by_all.min():.2f}, {by_all.max():.2f}] nT")
print(f"  BZ:    [{bz_all.min():.2f}, {bz_all.max():.2f}] nT")
print(f"  Vx:    [{vx_all.min():.2f}, {vx_all.max():.2f}] km/s")
print(f"  Vy:    [{vy_all.min():.2f}, {vy_all.max():.2f}] km/s")
print(f"  Vz:    [{vz_all.min():.2f}, {vz_all.max():.2f}] km/s")
print(f"  Np:    [{Np_all.min():.2f}, {Np_all.max():.2f}] cm^-3")
print(f"  SYM-H: [{sym_h_all.min():.2f}, {sym_h_all.max():.2f}] nT")
print(f"  ASY-H: [{asy_h_all.min():.2f}, {asy_h_all.max():.2f}] nT")
# print(f"  PCI:   [{pci_all.min():.2f}, {pci_all.max():.2f}]")
print(f"  AE:    [{au_all.min():.2f}, {au_all.max():.2f}] nT")
print(f"  AE:    [{ae_all.min():.2f}, {ae_all.max():.2f}] nT")
print(f"  AL:    [{al_all.min():.2f}, {al_all.max():.2f}] nT")

# Print data statistics (moved after interpolation)


# ============================================================================
# STEP 2: LOAD F10.7 SOLAR FLUX DATA
# ============================================================================

print("\n" + "=" * 70)
print("STEP 2: Loading F10.7 solar flux data...")
print("=" * 70)

# Load F10.7 data (hourly resolution)
cdf_f107 = pycdf.CDF(OMNI_F107_FILE)
cdf_dat_f107 = cdf_f107.copy()
cdf_f107.close()

tmp_dt_f107 = pd.to_datetime(cdf_dat_f107['Epoch'])

# Apply filters
sta = pd.to_datetime("1998-10-01 00:00:00")  # Earlier start for 81-day averaging
end = pd.to_datetime(END_DATE)

time_diff = np.abs([date - sta for date in tmp_dt_f107])
sta_ind = time_diff.argmin()
time_diff = np.abs([date - end for date in tmp_dt_f107])
end_ind = time_diff.argmin()

flag_date_f107 = np.zeros(len(tmp_dt_f107), dtype=bool)
flag_date_f107[sta_ind:end_ind+1] = True

flag_value_f107 = ~(cdf_dat_f107['F10_INDEX1800'] > 999)
flag_f107 = flag_value_f107 & flag_date_f107

f_name_f107_tmp = [
    f"{dt.strftime('%Y%m%d')}_{dt.hour * 60 + dt.minute}" 
    for dt in tmp_dt_f107[flag_f107]
]
f107_all_tmp = cdf_dat_f107['F10_INDEX1800'][flag_f107]

print(f"F10.7 range: [{f107_all_tmp.min():.2f}, {f107_all_tmp.max():.2f}] SFU")

# Extract unique daily F10.7 values
f_name_f107_tmp_yearmon = np.array([s.split('_')[0] for s in f_name_f107_tmp])
f_name_f107_tmp_yearmon_unique = np.unique(f_name_f107_tmp_yearmon)

f107_all_tmp_yearmon_unique = np.full(len(f_name_f107_tmp_yearmon_unique), np.nan)

for i, yearmon in enumerate(f_name_f107_tmp_yearmon_unique):
    flag_yearmon = f_name_f107_tmp_yearmon == yearmon
    f107_all_tmp_yearmon_unique[i] = f107_all_tmp[flag_yearmon][0]

print(f"Unique F10.7 days: {len(f107_all_tmp_yearmon_unique)}")
if len(f_name_f107_tmp_yearmon_unique) > 0:
    print(f"  First F10.7 day: {f_name_f107_tmp_yearmon_unique[0]}")
    print(f"  Last F10.7 day: {f_name_f107_tmp_yearmon_unique[-1]}")
    print(f"  Sample F10.7 filenames: {f_name_f107_tmp[:3]}")


# ============================================================================
# STEP 3: MATCH F10.7 DATA TO 1-MINUTE OMNI TIMESTAMPS
# ============================================================================

print("\n" + "=" * 70)
print("STEP 3: Matching F10.7 to 1-minute data...")
print("=" * 70)

f_name_all_yearmon = np.array([s.split('_')[0] for s in f_name_all])
f_name_all_yearmon_unique = np.unique(f_name_all_yearmon)

print(f"\nOMNI data summary:")
print(f"  Total timesteps: {len(f_name_all)}")
print(f"  Unique days: {len(f_name_all_yearmon_unique)}")
if len(f_name_all_yearmon_unique) > 0:
    print(f"  First day: {f_name_all_yearmon_unique[0]}")
    print(f"  Last day: {f_name_all_yearmon_unique[-1]}")
    print(f"  Sample filenames: {f_name_all[:3]}")

# Storage for matched data
f_name_save = []
bx_save = []
by_save = []
bz_save = []
vx_save = []
vy_save = []
vz_save = []
Np_save = []
sym_h_save = []
asy_h_save = []
# pci_save = []
au_save = []
ae_save = []
al_save = []
f107_save = []
f107_81day_save = []

# Match daily F10.7 values to each minute
print("Matching daily F10.7...")
print(f"  OMNI unique days: {len(f_name_all_yearmon_unique)}")
print(f"  F10.7 unique days: {len(f_name_f107_tmp_yearmon_unique)}")
if len(f_name_all_yearmon_unique) > 0:
    print(f"  OMNI date range: {f_name_all_yearmon_unique[0]} to {f_name_all_yearmon_unique[-1]}")
if len(f_name_f107_tmp_yearmon_unique) > 0:
    print(f"  F10.7 date range: {f_name_f107_tmp_yearmon_unique[0]} to {f_name_f107_tmp_yearmon_unique[-1]}")

matches_found = 0
for i in range(len(f_name_all_yearmon_unique)):
    flag_find_f107 = f_name_f107_tmp_yearmon_unique == f_name_all_yearmon_unique[i]
    
    if np.sum(flag_find_f107) == 0:
        continue  # Skip days without F10.7 data
    
    matches_found += 1
    flag_find_bx = f_name_all_yearmon == f_name_all_yearmon_unique[i]
    
    # Save all variables for this day
    f_name_save.append(f_name_all[flag_find_bx].flatten())
    bx_save.append(bx_all[flag_find_bx].flatten())
    by_save.append(by_all[flag_find_bx].flatten())
    bz_save.append(bz_all[flag_find_bx].flatten())
    vx_save.append(vx_all[flag_find_bx].flatten())
    vy_save.append(vy_all[flag_find_bx].flatten())
    vz_save.append(vz_all[flag_find_bx].flatten())
    Np_save.append(Np_all[flag_find_bx].flatten())
    sym_h_save.append(sym_h_all[flag_find_bx].flatten())
    asy_h_save.append(asy_h_all[flag_find_bx].flatten())
    # pci_save.append(pci_all[flag_find_bx].flatten())
    au_save.append(au_all[flag_find_bx].flatten())
    ae_save.append(ae_all[flag_find_bx].flatten())
    al_save.append(al_all[flag_find_bx].flatten())
    
    # Replicate daily F10.7 for all minutes
    f107_save.append(
        np.repeat(f107_all_tmp_yearmon_unique[flag_find_f107][0], np.sum(flag_find_bx)).flatten()
    )

print(f"  Matched {matches_found} days with F10.7 data")

# Compute 81-day averaged F10.7 (centered on each day)
print("Computing 81-day averaged F10.7...")
for i in range(len(f_name_all_yearmon_unique)):
    flag_find_f107 = f_name_f107_tmp_yearmon_unique == f_name_all_yearmon_unique[i]
    
    if np.sum(flag_find_f107) == 0:
        continue
    
    flag_find_bx = f_name_all_yearmon == f_name_all_yearmon_unique[i]
    
    # Get date range: ±40 days around current day
    d = dt.datetime.strptime(f_name_all_yearmon_unique[i], '%Y%m%d')
    minus_plus_40 = np.array([
        (d - dt.timedelta(days=40) + dt.timedelta(days=delta_day)).strftime('%Y%m%d') 
        for delta_day in range(81)
    ])
    
    # Find F10.7 values in this 81-day window
    flag_find_f107_81day_range = np.isin(f_name_f107_tmp_yearmon_unique, minus_plus_40)
    
    # Compute mean and replicate
    f107_81day_save.append(
        np.repeat(
            np.mean(f107_all_tmp_yearmon_unique[flag_find_f107_81day_range]), 
            np.sum(flag_find_bx)
        ).flatten()
    )

# Check if any data was matched
if len(f_name_save) == 0:
    print("\n" + "=" * 70)
    print("ERROR: No data matched between OMNI and F10.7!")
    print("=" * 70)
    print("\nPossible issues:")
    print("1. Date ranges don't overlap")
    print("2. All OMNI data was filtered out (check quality filters)")
    print("3. F10.7 data doesn't cover the OMNI date range")
    print("\nExiting...")
    import sys
    sys.exit(1)

# Concatenate all saved data
print(f"\nConcatenating {len(f_name_save)} days of matched data...")
f_name_save_np = np.concatenate(f_name_save)
bx_save_np = np.concatenate(bx_save)
by_save_np = np.concatenate(by_save)
bz_save_np = np.concatenate(bz_save)
vx_save_np = np.concatenate(vx_save)
vy_save_np = np.concatenate(vy_save)
vz_save_np = np.concatenate(vz_save)
Np_save_np = np.concatenate(Np_save)
sym_h_save_np = np.concatenate(sym_h_save)
asy_h_save_np = np.concatenate(asy_h_save)
# pci_save_np = np.concatenate(pci_save)
au_save_np = np.concatenate(au_save)
ae_save_np = np.concatenate(ae_save)
al_save_np = np.concatenate(al_save)
f107_save_np = np.concatenate(f107_save)
f107_81day_save_np = np.concatenate(f107_81day_save)

print(f"\nFinal matched data: {len(f_name_save_np)} timesteps")


# ============================================================================
# STEP 4: COMPUTE DERIVED QUANTITIES
# ============================================================================

print("\n" + "=" * 70)
print("STEP 4: Computing derived quantities...")
print("=" * 70)

# Universal Time (UT) and Day of Year (DOY)
ut_save_np = np.empty(len(f_name_save_np))
doy_save_np = np.empty(len(f_name_save_np))
datetimes_save_np = []

for i in range(len(f_name_save_np)):
    date_part, minute_part = f_name_save_np[i].split('_')
    d = dt.datetime.strptime(date_part, '%Y%m%d') + dt.timedelta(minutes=int(minute_part))
    
    ut_save_np[i] = d.hour + d.minute / 60.0
    doy_save_np[i] = d.timetuple().tm_yday
    datetimes_save_np.append(d)

# Normalize time to [0, 2π]
ut_save_np_norm = ut_save_np / 24.0 * 2 * np.pi
doy_save_np_norm = doy_save_np / 365.0 * 2 * np.pi

# Sinusoidal encoding of time
ut_save_np_norm_sin = np.sin(ut_save_np_norm)
ut_save_np_norm_cos = np.cos(ut_save_np_norm)
doy_save_np_norm_sin = np.sin(doy_save_np_norm)
doy_save_np_norm_cos = np.cos(doy_save_np_norm)

# Dipole tilt angle (proper IGRF + GSM calculation)
dipole_tilt = compute_dipole_tilt(datetimes_save_np)

# IMF clock angle components
bt_save_np = np.sqrt(by_save_np**2 + bz_save_np**2)
sin_imf_clock = by_save_np / (bt_save_np + 1e-12)
cos_imf_clock = bz_save_np / (bt_save_np + 1e-12)

print("Derived quantities computed:")
print(f"  UT:         [{ut_save_np.min():.2f}, {ut_save_np.max():.2f}] hours")
print(f"  DOY:        [{doy_save_np.min():.0f}, {doy_save_np.max():.0f}]")
print(f"  BT:         [{bt_save_np.min():.2f}, {bt_save_np.max():.2f}] nT")
print(f"  Dipole tilt: [{dipole_tilt.min():.2f}, {dipole_tilt.max():.2f}] degrees")


# ============================================================================
# STEP 5: CREATE CONDITION VECTORS FOR TRAINING
# ============================================================================

print("\n" + "=" * 70)
print("STEP 5: Creating condition vectors...")
print("=" * 70)

# Determine which training mode is active
active_mode = [k for k, v in TRAINING_MODES.items() if v]
if len(active_mode) != 1:
    print(f"WARNING: Expected 1 active mode, found {len(active_mode)}: {active_mode}")
    print("Defaulting to joint_training")
    active_mode = ['joint_training']
else:
    print(f"Active training mode: {active_mode[0]}")

keep_dict = {}

# Build condition vectors based on training mode
if TRAINING_MODES['raw_solar_wind']:
    # Raw solar wind parameters
    print("Building: [Vx, Vy, Vz, Np, BX, BY, BZ, AL, sin(UT), cos(UT), sin(DOY), cos(DOY)]")
    for i in range(len(f_name_save_np)):
        keep_dict[f_name_save_np[i]] = [
            vx_save_np[i], vy_save_np[i], vz_save_np[i], Np_save_np[i],
            bx_save_np[i], by_save_np[i], bz_save_np[i], al_save_np[i],
            ut_save_np_norm_sin[i], ut_save_np_norm_cos[i],
            doy_save_np_norm_sin[i], doy_save_np_norm_cos[i],
        ]

elif TRAINING_MODES['flux_training']:
    # For energy flux prediction
    print("Building: [Newell, BY, BZ, Pdyn, AL, sin(UT), cos(UT), sin(DOY), cos(DOY)]")
    for i in range(len(f_name_save_np)):
        newell_i, pdyn_i = newell_and_pdyn(
            bx_save_np[i], by_save_np[i], bz_save_np[i], Np_save_np[i],
            vx_save_np[i], vy_save_np[i], vz_save_np[i]
        )
        keep_dict[f_name_save_np[i]] = [
            newell_i, by_save_np[i], bz_save_np[i], pdyn_i, al_save_np[i],
            ut_save_np_norm_sin[i], ut_save_np_norm_cos[i],
            doy_save_np_norm_sin[i], doy_save_np_norm_cos[i],
        ]

elif TRAINING_MODES['potential_training']:
    # For electric potential prediction
    # print("Building: [Newell, BT, sin(clock), cos(clock), Pdyn, F107, SYM-H, PCI, tilt]")
    # print("Building: [Newell, BT, sin(clock), cos(clock), Pdyn, F107, SYM-H, tilt]")
    print("Building: [PCI, BT, sin(clock), cos(clock), Pdyn, F107, SYM-H, tilt]")
    for i in range(len(f_name_save_np)):
        newell_i, pdyn_i = newell_and_pdyn(
            bx_save_np[i], by_save_np[i], bz_save_np[i], Np_save_np[i],
            vx_save_np[i], vy_save_np[i], vz_save_np[i]
        )
        # keep_dict[f_name_save_np[i]] = [
        #     newell_i, bt_save_np[i], sin_imf_clock[i], cos_imf_clock[i],
        #     pdyn_i, f107_81day_save_np[i], sym_h_save_np[i],
        #     # pci_save_np[i], 
        #     dipole_tilt[i],
        # ]
        keep_dict[f_name_save_np[i]] = [ 
            # pci_save_np[i],
            bt_save_np[i], sin_imf_clock[i], cos_imf_clock[i],
            pdyn_i, f107_81day_save_np[i], sym_h_save_np[i],
            dipole_tilt[i],
        ]

elif TRAINING_MODES['energy_training']:
    # For energy deposition prediction
    print("Building: [Newell, ASY-H, sin(clock), cos(clock), Pdyn, BY, SYM-H, AL, tilt]")
    for i in range(len(f_name_save_np)):
        newell_i, pdyn_i = newell_and_pdyn(
            bx_save_np[i], by_save_np[i], bz_save_np[i], Np_save_np[i],
            vx_save_np[i], vy_save_np[i], vz_save_np[i]
        )
        keep_dict[f_name_save_np[i]] = [
            newell_i, asy_h_save_np[i], sin_imf_clock[i], cos_imf_clock[i],
            pdyn_i, by_save_np[i], sym_h_save_np[i], al_save_np[i],
            dipole_tilt[i],
        ]

elif TRAINING_MODES['joint_training']:
    # Joint training: combined conditions for multi-output model
    # print("Building: [Newell, sin(clock), cos(clock), SYM-H, Pdyn, F107, AL, AE]")
    print("Building: [Newell, sin(clock), cos(clock), SYM-H, Pdyn, F107, AE, tilt]")
    for i in range(len(f_name_save_np)):
        newell_i, pdyn_i = newell_and_pdyn(
            bx_save_np[i], by_save_np[i], bz_save_np[i], Np_save_np[i],
            vx_save_np[i], vy_save_np[i], vz_save_np[i]
        )
        keep_dict[f_name_save_np[i]] = [
            newell_i,
            sin_imf_clock[i],
            cos_imf_clock[i],
            sym_h_save_np[i],
            pdyn_i,
            f107_81day_save_np[i],
            # al_save_np[i], # mj_20260211
            ae_save_np[i],
            dipole_tilt[i],
        ]

elif TRAINING_MODES['AU_AL_training']:
    # Joint training: combined conditions for multi-output model
    # print("Building: [Newell, sin(clock), cos(clock), SYM-H, Pdyn, F107, AL, AE]")
    # print("Building: [Newell, sin(clock), cos(clock), SYM-H, Pdyn, F107, AE, tilt]")
    print("Building: [AU,AL]")
    for i in range(len(f_name_save_np)):
        # newell_i, pdyn_i = newell_and_pdyn(
        #     bx_save_np[i], by_save_np[i], bz_save_np[i], Np_save_np[i],
        #     vx_save_np[i], vy_save_np[i], vz_save_np[i]
        # )
        keep_dict[f_name_save_np[i]] = [
            au_save_np[i],
            al_save_np[i], # mj_20260211
        ]

elif TRAINING_MODES['time_only']:
    # Time features only (for testing)
    print("Building: [sin(UT), cos(UT), sin(DOY), cos(DOY), AE]")
    for i in range(len(f_name_save_np)):
        keep_dict[f_name_save_np[i]] = [
            ut_save_np_norm_sin[i], ut_save_np_norm_cos[i],
            doy_save_np_norm_sin[i], doy_save_np_norm_cos[i],
            ae_save_np[i]
        ]

elif TRAINING_MODES['all_parameters']:
    # All available parameters (for exploratory analysis)
    print("Building: [sin(UT), cos(UT), sin(DOY), cos(DOY), F107, F107_81d, Vx, Vy, Vz, Np, BX, BY, BZ, SYM-H, ASY-H, PCI]")
    for i in range(len(f_name_save_np)):
        keep_dict[f_name_save_np[i]] = [
            ut_save_np_norm_sin[i], ut_save_np_norm_cos[i],
            doy_save_np_norm_sin[i], doy_save_np_norm_cos[i],
            f107_save_np[i], f107_81day_save_np[i],
            vx_save_np[i], vy_save_np[i], vz_save_np[i], Np_save_np[i],
            bx_save_np[i], by_save_np[i], bz_save_np[i],
            sym_h_save_np[i], asy_h_save_np[i], 
            # pci_save_np[i],
        ]

print(f"Created condition vectors for {len(keep_dict)} timesteps")


# ============================================================================
# STEP 5.5: PLOT OMNI PARAMETERS FOR STORM EVENTS (OPTIONAL)
# ============================================================================

if_plot = True  # Set to True to generate plots

if if_plot:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    print("\n" + "=" * 70)
    print("STEP 5.5: Plotting OMNI parameters for storm events...")
    print("=" * 70)
    
    # Define storm events (start_date, end_date)
    DATE_RANGES = [
        ((2000, 4, 6), (2000, 4, 7)),
        ((2000, 7, 13), (2000, 7, 17)),
        ((2003, 10, 28), (2003, 11, 1)),
        ((2003, 11, 19), (2003, 11, 21)),
        ((2003, 3, 30), (2003, 4, 2)),
        ((2003, 5, 28), (2003, 6, 1)),
    ]
    
    # Convert f_name_save_np to datetime objects for easier filtering
    datetimes_save = np.array([
        dt.datetime.strptime(fname.split('_')[0], '%Y%m%d') + 
        dt.timedelta(minutes=int(fname.split('_')[1]))
        for fname in f_name_save_np
    ])
    
    # Compute Newell coupling and dynamic pressure for all timesteps
    newell_save_np = np.empty(len(f_name_save_np))
    pdyn_save_np = np.empty(len(f_name_save_np))
    
    for i in range(len(f_name_save_np)):
        newell_i, pdyn_i = newell_and_pdyn(
            bx_save_np[i], by_save_np[i], bz_save_np[i], Np_save_np[i],
            vx_save_np[i], vy_save_np[i], vz_save_np[i]
        )
        newell_save_np[i] = newell_i
        pdyn_save_np[i] = pdyn_i
    
    # Plot for each storm event
    for event_idx, ((y1, m1, d1), (y2, m2, d2)) in enumerate(DATE_RANGES):
        start_date = dt.datetime(y1, m1, d1)
        end_date = dt.datetime(y2, m2, d2)
        
        print(f"\nPlotting event {event_idx + 1}: {start_date.date()} to {end_date.date()}")
        
        # Filter data for this date range
        mask = (datetimes_save >= start_date) & (datetimes_save <= end_date)
        
        if not np.any(mask):
            print(f"  No data found for this period, skipping...")
            continue
        
        times = datetimes_save[mask]
        bz = bz_save_np[mask]
        by = by_save_np[mask]
        symh = sym_h_save_np[mask]
        newell = newell_save_np[mask]
        pdyn = pdyn_save_np[mask]
        f107 = f107_81day_save_np[mask]
        al = al_save_np[mask]
        ae = ae_save_np[mask]
        sin_clock = sin_imf_clock[mask]
        cos_clock = cos_imf_clock[mask]
        
        # Create figure with 8 subplots
        fig, axes = plt.subplots(8, 1, figsize=(14, 16), sharex=True)
        plt.subplots_adjust(hspace=0.05)
        
        # (a) IMF Bz and By
        ax = axes[0]
        ax.plot(times, bz, '-', color='#e76f51', label='Bz', linewidth=1.5)
        ax.plot(times, by, '-', color='#005f73', label='By', linewidth=1.5)
        ax.set_ylabel(r'$B_{\mathrm{IMF}}$ [nT]', fontsize=18)
        ax.legend(loc='upper right', fontsize=14, ncol=2, frameon=True, 
                 facecolor='white', framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(a)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (b) SYM-H
        ax = axes[1]
        ax.plot(times, symh, '-', color='#9467BD', linewidth=1.5)
        ax.set_ylabel('Sym-H [nT]', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(b)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (c) Newell Coupling
        ax = axes[2]
        ax.plot(times, newell, '-', color='#2a9d8f', linewidth=1.5)
        ax.set_ylabel('Newell\n[mV/m]', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(c)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        
        # (d) Dynamic Pressure
        ax = axes[3]
        ax.plot(times, pdyn, '-', color='#f4a261', linewidth=1.5)
        ax.set_ylabel(r'$P_{\mathrm{dyn}}$ [nPa]', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(d)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        
        # (e) IMF Clock Angle (sin and cos)
        ax = axes[4]
        ax.plot(times, sin_clock, '-', color='#e63946', label='sin(θ)', linewidth=1.5)
        ax.plot(times, cos_clock, '-', color='#457b9d', label='cos(θ)', linewidth=1.5)
        ax.set_ylabel('Clock Angle', fontsize=18)
        ax.legend(loc='upper right', fontsize=14, ncol=2, frameon=True,
                 facecolor='white', framealpha=0.8)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(e)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        
        # (f) F10.7 (81-day average)
        ax = axes[5]
        ax.plot(times, f107, '-', color='#ff9f1c', linewidth=1.5)
        ax.set_ylabel('F10.7\n[SFU]', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(f)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        
        # (g) AL Index
        ax = axes[6]
        ax.plot(times, al, '-', color='#06aed5', linewidth=1.5)
        ax.set_ylabel('AL [nT]', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(g)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        
        # (h) AE Index
        ax = axes[7]
        ax.plot(times, ae, '-', color='#bc6c25', linewidth=1.5)
        ax.set_ylabel('AE [nT]', fontsize=18)
        ax.set_xlabel('Time [UTC]', fontsize=18)
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.85, '(h)', transform=ax.transAxes, fontsize=18, fontweight='bold')
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d\n%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        
        # Set tick parameters for all axes
        for ax in axes:
            ax.tick_params(labelsize=16)
        
        # Add title
        fig.suptitle(f'OMNI Parameters: {start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
                    fontsize=20, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        
        # Save figure
        save_dir = 'home/lim2/code/ amie_data_process/plots_hro2s_interp/'
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 
                                f'omni_params_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {save_path}")
        plt.close()
    
    print("\nPlotting complete!")


# ============================================================================
# STEP 6: SAVE OUTPUT
# ============================================================================

print("\n" + "=" * 70)
print("STEP 6: Saving output...")
print("=" * 70)

SAVE_OUTPUT = False  # Set to True to actually save

if SAVE_OUTPUT:
    np.savez(OUTPUT_FILE, **keep_dict)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"  Keys: {len(keep_dict)} timesteps")
    print(f"  Vector dimension: {len(list(keep_dict.values())[0])}")
else:
    print("Output NOT saved (SAVE_OUTPUT = False)")
    print(f"Would save to: {OUTPUT_FILE}")
    print(f"  Keys: {len(keep_dict)} timesteps")
    print(f"  Vector dimension: {len(list(keep_dict.values())[0])}")

print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)

# %%
"""
===============================================================================
CONFIGURATION RECOMMENDATIONS
===============================================================================

1. INTERPOLATION (ENABLE_INTERPOLATION):
   ✓ RECOMMENDED: True with MAX_GAP_MINUTES = 5
   
   Rationale:
   - Small gaps (1-5 min) are often due to data dropouts, not physical changes
   - Linear interpolation is reasonable for slowly-varying solar wind
   - Improves temporal continuity for the 60-minute history window
   - Large gaps (>5 min) are preserved to avoid artificial data
   
   WARNING: Do NOT interpolate if gaps represent actual data unavailability 
            during storm times (when OMNI may have real gaps)


2. TIME DELAY (APPLY_TIME_SHIFT):
   
   OPTION A: APPLY_TIME_SHIFT = False (RECOMMENDED for temporal encoder)
   -----------------------------------------------------------------------
   Advantages:
   - Temporal encoder with 60-min history can learn optimal lag
   - More flexible: lag varies with solar wind speed (30-60 min typical)
   - Model can learn non-linear relationships
   - Handles complex propagation effects (foreshock, magnetosheath)
   
   Disadvantages:
   - Requires model to learn this relationship
   - May need more training data
   
   When to use:
   - ✓ Using temporal encoder (transformer/LSTM) with ≥60 min history
   - ✓ Want model flexibility
   - ✓ Have sufficient training data
   
   
   OPTION B: APPLY_TIME_SHIFT = True (30 min shift)
   -------------------------------------------------
   Advantages:
   - Physically motivated preprocessing
   - Reduces model complexity
   - Works well for simple models
   
   Disadvantages:
   - Fixed lag (actual lag is ~30-60 min, varies with Vsw)
   - May not be optimal for all conditions
   - Loses some information (model can't see "future" OMNI)
   
   When to use:
   - ✓ NOT using temporal encoder (single-timestep conditioning)
   - ✓ Simple model architecture
   - ✓ Want physically-motivated preprocessing
   
   
   SCIENTIFIC CONTEXT:
   ------------------
   Solar wind propagation from L1 (1.5 million km upstream) to Earth:
   
   Distance to bow shock: ~235 Re (~1.5 million km)
   Typical solar wind speed: 400-800 km/s
   Propagation time: distance / speed
   
   Examples:
   - 400 km/s → ~62 minutes
   - 500 km/s → ~50 minutes  
   - 600 km/s → ~42 minutes
   - 800 km/s → ~31 minutes
   
   Additional magnetospheric response time: ~5-15 minutes
   
   Total lag: 30-75 minutes (highly variable!)
   
   
   RECOMMENDATION FOR YOUR CASE:
   -----------------------------
   Given you're using a temporal encoder (transformer) with 60-minute history:
   
   → Set APPLY_TIME_SHIFT = False
   
   Reasoning:
   1. Your 60-min history window includes the typical lag range
   2. Transformer attention can learn to focus on relevant past timesteps
   3. Model can adapt lag based on solar wind speed (implicitly)
   4. More flexible for different geomagnetic conditions
   
   You can verify the model learned appropriate lags by:
   - Plotting attention weights vs time offset
   - Computing lagged correlations between OMNI and AMIE in outputs
   - Expected peak correlation at ~30-50 min lag
   
===============================================================================
"""

# %%
