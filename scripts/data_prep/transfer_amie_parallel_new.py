#%%
from generate_condition_all import * # take 5 minutes
# def load_cond_keys(cond_path):
#     d = np.load(cond_path, allow_pickle=True)
#     return set(d.files)
cond_keys = list(keep_dict.keys())
#%%
def cpcp_from_pcn(pcn,method='Ridley',pcn_month=None):
    # method: 'Troshichev', 'Ridley'
    if method == 'Troshichev':
        cpcp = 19.35 * pcn + 8.78
    elif method == 'Ridley':
        cpcp = 29.28 - 3.31 * np.sin((pcn_month - 1) * 2 * np.pi / 12 + 1.49) + 17.81 * pcn
    return cpcp
#%%
from scipy.io import readsav
import numpy as np
from PIL import Image
import os
import glob
from PIL import Image
import sklearn.model_selection
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import json
import multiprocessing as mp
from tqdm import tqdm
import time
#%%
from scipy.interpolate import PchipInterpolator
from typing import Tuple

def resample_field_24x25_to_hw(
    field_24x25: np.ndarray,  # [24,25], second axis MLT = 0..24 (24 == 0 repeated end point)
    mlat_24: np.ndarray,  # [24], monotonic (usually 90 -> ~44)
    H: int = 64,
    W: int = 64,
    avg_endpoints: bool = False,  # True: replace column 0 by the mean of columns 0 and -1
    nan_policy: str = "raise"      # "raise" | "skipcol" | "filllin"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resample the (24 x 25) grid to (H x W):
      - MLAT: PCHIP (no overshoot)
      - MLT: periodic linear interpolation
    Returns (field_HxW, mlat_new[H], mlt_new[W])
    """
    assert field_24x25.shape == (24, 25), f"expect (24,25), got {field_24x25.shape}"
    assert mlat_24.shape == (24,), f"expect (24,), got {mlat_24.shape}"

    f = np.asarray(field_24x25, dtype=np.float64).copy()
    mlat = np.asarray(mlat_24, dtype=np.float64).copy()

    # optional: reconcile the 0/24 end points
    if avg_endpoints:
        mean_col0 = 0.5 * (f[:, 0] + f[:, -1])
        f[:, 0] = mean_col0
        f[:, -1] = mean_col0

    # drop the repeated end column -> [24,24]
    f = f[:, :-1]

    # MLAT ascending for PCHIP
    if mlat[0] > mlat[-1]:
        mlat_inc = mlat[::-1]
        f_inc = f[::-1, :]
        flipped = True
    else:
        mlat_inc = mlat
        f_inc = f
        flipped = False

    # target coordinates
    mlat_new_inc = np.linspace(mlat_inc[0], mlat_inc[-1], H)
    mlt_old = np.linspace(0.0, 24.0, f_inc.shape[1], endpoint=False)  # 24 points
    mlt_new = np.linspace(0.0, 24.0, W, endpoint=False)  # W points

    # MLAT: column-wise PCHIP (with NaN / Inf handling)
    f_mlat = np.empty((H, f_inc.shape[1]), dtype=np.float64)
    for j in range(f_inc.shape[1]):
        col = f_inc[:, j]
        finite = np.isfinite(col)
        if finite.all():
            pj = PchipInterpolator(mlat_inc, col, axis=0)
            f_mlat[:, j] = pj(mlat_new_inc)
        else:
            if nan_policy == "raise":
                raise ValueError("Column has non-finite values; set nan_policy to 'filllin' or 'skipcol'.")
            elif nan_policy == "filllin":
                # fill linearly along MLAT, then PCHIP
                good = finite
                if good.sum() < 2:
                    # too few points: whole column NaN
                    f_mlat[:, j] = np.nan
                else:
                    col_lin = np.interp(mlat_inc, mlat_inc[good], col[good])
                    pj = PchipInterpolator(mlat_inc, col_lin, axis=0)
                    f_mlat[:, j] = pj(mlat_new_inc)
            elif nan_policy == "skipcol":
                # linear interpolation instead of PCHIP
                good = finite
                if good.sum() < 2:
                    f_mlat[:, j] = np.nan
                else:
                    f_mlat[:, j] = np.interp(mlat_new_inc, mlat_inc[good], col[good])
            else:
                raise ValueError(f"Unknown nan_policy: {nan_policy}")

    # flip back if flipped above
    if flipped:
        mlat_new = mlat_new_inc[::-1]
        f_mlat = f_mlat[::-1, :]
    else:
        mlat_new = mlat_new_inc

    # MLT: periodic linear interpolation of every row to W columns
    out = np.empty((H, W), dtype=np.float64)
    for i in range(H):
        row = f_mlat[i, :]
        if np.isfinite(row).all():
            out[i, :] = np.interp(mlt_new, mlt_old, row, period=24.0)
        else:
            # NaN in the row: nearest-neighbour circular fill, then interpolate
            rr = row.copy()
            finite = np.isfinite(rr)
            if finite.any():
                idx = np.where(finite, np.arange(len(rr)), np.nan)
                # forward fill
                for k in range(len(rr)):
                    if not np.isfinite(idx[k]):
                        idx[k] = idx[k-1] if k > 0 else np.nan
                # leading NaN: wrap around from the last valid index
                if not np.isfinite(idx[0]):
                    last = np.where(finite)[0][-1]
                    for k in range(len(rr)):
                        if not np.isfinite(idx[k]):
                            idx[k] = last
                        else:
                            break
                rr[~finite] = rr[idx.astype(int)[~finite]]
                out[i, :] = np.interp(mlt_new, mlt_old, rr, period=24.0)
            else:
                out[i, :] = np.nan  # whole row invalid

    return out, mlat_new, mlt_new

#%%
print("CPU logical cores seen by OS:", mp.cpu_count())
print("SLURM assigned cores:", os.environ.get("SLURM_CPUS_ON_NODE"))
#%%
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True, help="Year to process, e.g. 2003")
parser.add_argument("--phys",  choices=["logflux", "logenergy", "potential", "all"], required=True)    
args = parser.parse_args()


year_series = [args.year]   # <<< instead of hard-coded [1999,...]
print("Processing year:", year_series)


sorted_filepaths_all = []
for yr in year_series:
    # base_dir = f'/home/lim2/data/Amie/{yr}'
    base_dir = f'/data/minjing/Data/Amie/{yr}'
    pattern = os.path.join(base_dir, '**', '*.save')
    filepaths = glob.glob(pattern, recursive=True)
    filepaths = [f for f in filepaths if '_all' not in os.path.basename(f)]
    sorted_filepaths = sorted(filepaths, key=lambda x: x.split('/')[-1][1:9])
    print(len(sorted_filepaths),sorted_filepaths[0],sorted_filepaths[-1])
    sorted_filepaths_all.extend(sorted_filepaths)

print(len(sorted_filepaths_all),sorted_filepaths_all[0],sorted_filepaths_all[-1])


sorted_filenames_all = []
for path in sorted_filepaths_all[:]:
    date_str = os.path.basename(path)[1:9]  
    for i in range(1440):
        sorted_filenames_all.append(f"{date_str}_{i}.npy")
print(len(sorted_filenames_all),1440*len(sorted_filepaths_all))


# shuffle_filenames = np.random.permutation(sorted_filenames_all)
# sampled_filenames = np.random.choice(shuffle_filenames, size=len(sorted_filenames_all), replace=False)
# training_file_names, validation_file_names = sklearn.model_selection.train_test_split(sampled_filenames, test_size=0.2)
# print(len(training_file_names), len(validation_file_names))
#%%
# data = readsav('/home/lim2/data/Amie/2003/01/b20030101n.save', verbose=True)
data = readsav('/data/minjing/Data/Amie/2003/01/b20030101n.save', verbose=True)
lat = data['lats']
mlt = data['mlts']
lat_interp_rad = np.radians(lat)
mlt_interp_rad = np.radians(mlt/24*360)
R = 6371e3 # Earth radius in meters
area = np.zeros((len(lat)-1,len(mlt)-1)) # (lat,mlt)
dlon = mlt_interp_rad[1] - mlt_interp_rad[0]
for i in range(len(lat)-1):
    phi1 = lat_interp_rad[i+1]
    phi2 = lat_interp_rad[i]
    band_area = R**2 * dlon * (np.sin(phi2) - np.sin(phi1))
    area[i, :] = band_area
flag_lat_polar = np.where((lat[1:] > 50) & (lat[1:] <= 90),1,0)

def compute_total_flux(flux_map,area,if_polar_region=True,flag_lat_polar=None):
    if flux_map.ndim == 3:
        flux_map_reduced = 0.25 * (
            flux_map[:-1, :-1, :] + flux_map[1:, :-1, :] +
            flux_map[:-1, 1:, :] + flux_map[1:, 1:, :]
        )  # shape: (lat,mlt,time)
        if if_polar_region:
            return np.sum(flux_map_reduced * area[:,:,np.newaxis] * flag_lat_polar[:,np.newaxis,np.newaxis],axis=(0,1))
        else:
            return np.sum(flux_map_reduced * area[:,:,np.newaxis],axis=(0,1))
    elif flux_map.ndim == 2:
        flux_map_reduced = 0.25 * (
            flux_map[:-1, :-1] + flux_map[1:, :-1] +
            flux_map[:-1, 1:] + flux_map[1:, 1:]
        )  # shape: (lat,mlt)
        if if_polar_region:
            return np.sum(flux_map_reduced * area * flag_lat_polar[:,np.newaxis])
        else:
            return np.sum(flux_map_reduced * area)
    else:
        raise ValueError("Invalid dimension of flux_map")

def process_single_sample(args_list):
    i,fn, sorted_filepaths_all, output_folder_tot = args_list

    try:
        indx_data_path = next(idx for idx, path_ in enumerate(sorted_filepaths_all) if fn[:8] in path_)

        data = readsav(sorted_filepaths_all[indx_data_path], verbose=False)


        # extract the raw data
        energy = data['data'][:, :, 5, int(fn[9:-4])] # (mlat,mlt)
        flux = data['data'][:, :, 6, int(fn[9:-4])]
        potential  = data['data'][:, :, 0, int(fn[9:-4])]


        # clean data
        yr_ = int(fn[:4])
        month_ = int(fn[4:6])
        if (b'(W/m2)' in data['fields'][6]) or (b'(w/m2)' in data['fields'][6]):
            flux = flux * 1e3 
        if yr_ < 2001:
            flux = flux / 1e3
        if (yr == 2001) and (month_ >= 5):
            flux = flux / 1e3

        if args.phys == "logflux":
            hp = compute_total_flux(flux, # shape: (lat,mlt)
                                                area,
                                                if_polar_region=True,
                                                flag_lat_polar=flag_lat_polar)/1e12
            flag_data_clean = np.where(hp>350,True,False) | (flux > 150).any() | np.isnan(flux).any()
        elif args.phys == "potential":

            fn_label = fn.split('.')[0]
            if fn_label in cond_keys:
                pcn = keep_dict[fn_label][7]
                cpcp_ridley_pcn = cpcp_from_pcn(pcn,method='Ridley',pcn_month=int(fn[4:6]))
                cpcp = np.max(potential) - np.min(potential)
                ratio_pcn = cpcp_ridley_pcn / cpcp

                flag_data_clean = np.where( ((ratio_pcn>0)&(ratio_pcn<1/6)) | (np.isnan(cpcp)), True, False)
            else:
                flag_data_clean = True

            # flag_data_clean = (cpcp > 650) | (np.max(np.abs(potential)) > 400) | np.isnan(potential).any() | flag_tot

        elif args.phys == "logenergy":
            flag_data_clean = (energy > 50).any() | np.isnan(energy).any()

        elif (args.phys == "all") or (args.phys == "joint"):

            fn_label = fn.split('.')[0]
            if fn_label in cond_keys:

                # flux 
                hp = compute_total_flux(flux, # shape: (lat,mlt)
                                                    area,
                                                    if_polar_region=True,
                                                    flag_lat_polar=flag_lat_polar)/1e12
                # flag_flux_clean = np.where(hp>350,True,False) | (flux > 150).any() | np.isnan(flux).any()
                flag_flux_clean = np.where(hp>500,True,False) | (flux > 300).any() | np.isnan(flux).any()


                # potential
                cpcp = np.max(potential) - np.min(potential)
                # flag_potential_clean = (cpcp > 400) | (np.max(np.abs(potential)) > 350) | np.isnan(potential).any()
                flag_potential_clean = (cpcp > 600) | (np.max(np.abs(potential)) > 500) | np.isnan(potential).any()

                # energy
                # flag_energy_clean = (energy > 50).any() | np.isnan(energy).any()
                flag_energy_clean = (energy > 100).any() | np.isnan(energy).any()

                flag_data_clean = flag_flux_clean | flag_potential_clean | flag_energy_clean


            else:
                flag_data_clean = True

        
        if not flag_data_clean:
            arr_energy, _, _ = resample_field_24x25_to_hw(energy, lat,H=32,W=32)
            arr_flux, _, _ = resample_field_24x25_to_hw(flux, lat,H=32,W=32)
            arr_pot, _, _ = resample_field_24x25_to_hw(potential, lat,H=32,W=32)


            # stack to [3, H, W]
            arr_stack = np.stack([arr_energy, arr_flux, arr_pot], axis=0).astype(np.float32)

            save_path = os.path.join(output_folder_tot, fn)

            return (i,arr_stack, save_path)


    except Exception as e:
        print(f"Error in {fn}: {e}")
        return None
    
def benchmark_chunksizes(process_func, args_list, chunksizes, num_processes=32, test_size=200000):
    print(f"\nTesting chunksize performance on {test_size} samples...\n")
    test_args = args_list[:test_size]
    
    for chunk in chunksizes:
        print(f"Chunksize = {chunk}")
        start = time.time()

        results = []
        with mp.Pool(processes=num_processes) as pool:
            for r in tqdm(pool.imap_unordered(process_func, test_args, chunksize=chunk), total=len(test_args), desc=f"Chunksize {chunk}"):
                results.append(r)
        
        elapsed = time.time() - start
        print(f"Time taken: {elapsed:.2f} seconds\n")

#%%
# All 100%
if True:
    # output_folder_tot = f'/home/lim2/data/amie_npy_joint/1999_2010_all_{args.phys}/'
    output_folder_tot = f'/data/minjing/Data/amie_npy_joint/1999_2010_all_{args.phys}/'
    if not os.path.exists(output_folder_tot):
        os.makedirs(output_folder_tot,exist_ok=True)


    args_list = [(i, fn, sorted_filepaths_all, output_folder_tot) for i, fn in enumerate(sorted_filenames_all)]
    total_tasks = len(args_list)

    results = []
    # with mp.Pool(processes=32) as pool:
    with mp.Pool(processes=32) as pool:
        # for r in pool.imap_unordered(process_single_sample, args_list, chunksize=100):
        for r in tqdm(pool.imap_unordered(process_single_sample, args_list, chunksize=10), total=total_tasks, desc="Processing"):
            if r is not None:
                results.append(r)

    #%%
    # allocate the output
    N = len(sorted_filenames_all)
    arr_energy = np.zeros((N, 32, 32), dtype=np.float32)
    arr_flux = np.zeros((N, 32, 32), dtype=np.float32)
    arr_potential = np.zeros((N, 32, 32), dtype=np.float32)
    arr_save_path = np.empty(N, dtype=object)

    # write back at index i
    for i, arr_stack, path in results:
        arr_energy[i] = arr_stack[0]
        arr_flux[i] = arr_stack[1]
        arr_potential[i] = arr_stack[2]
        arr_save_path[i] = path


    np.savez(os.path.join(output_folder_tot,f'AllData_{args.year}.npz'),
            energy=arr_energy,
            flux=arr_flux,
            pot=arr_potential,
            paths=arr_save_path)



    # save files
    vmins = np.full(3, np.inf)
    vmaxs = np.full(3, -np.inf)

    vmins[0] = np.min(arr_energy)
    vmaxs[0] = np.max(arr_energy)
    vmins[1] = np.min(arr_flux)
    vmaxs[1] = np.max(arr_flux)
    vmins[2] = np.min(arr_potential)
    vmaxs[2] = np.max(arr_potential)

    range_dict = {
        "channels": ["mean energy", "energy flux", "potential"],
        "vmins": vmins.tolist(),
        "vmaxs": vmaxs.tolist()
    }
    with open(os.path.join(output_folder_tot,f'amie_range_{args.year}.json'), "w") as f:
        json.dump(range_dict, f, indent=2)


#%%
# import numpy as np
# arr_path = np.load(f'/home/lim2/data/amie_npy_new/1999_2010_all_potential/AllData_2006.npz',allow_pickle=True)['paths']
# arr_pot = np.load(f'/home/lim2/data/amie_npy_new/1999_2010_all_potential/AllData_2006.npz',allow_pickle=True)['pot']
# print('removed data number:',len(np.where(arr_path==None)[0]))
# print('saved data number:',len(arr_path)-len(np.where(arr_path==None)[0]))

# %%