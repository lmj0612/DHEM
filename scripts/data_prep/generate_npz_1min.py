#%%
from pickle import FALSE
from typing import Any
import numpy as np
from datetime import datetime, timedelta

from sympy.functions.elementary.piecewise import true
#%%
from generate_condition_all import * # take 5 minutes


#%%
if False:
    # flux train
    npy_files_train = sorted(
        [os.path.join('/home/lim2/data/amie_npy_new_all/training_npy_all', f) for f in os.listdir('/home/lim2/data/amie_npy_new_all/training_npy_all') if f.lower().endswith(".npy")]
    )

    npy_files_train_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_train])
    print('len(npy_files_train)',len(npy_files_train))
    
    conditions = [keep_dict[f] for f in npy_files_train_label]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]

    if True:
        phys = conditions[:, :-4]   # all except last 4
        periodic = conditions[:, -4:]  # last 4 stay untouched

        # Z-score only physical part
        mean = phys.mean(axis=0)   # [K_phys]
        std  = phys.std(axis=0)    # [K_phys]
        z_phys = (phys - mean) / (std + 1e-6)

        # Optional: clip
        # z_phys = np.clip(z_phys, -5, 5)

        # Recombine
        z_conditions = np.concatenate([z_phys, periodic], axis=1)

        print(
            "z_conditions shape:", z_conditions.shape,
            "\nphys max:", z_phys.max(axis=0),
            "\nphys min:", z_phys.min(axis=0),
            "\nperiodic range:", periodic.min(), periodic.max()
        )
        np.savez(
            '/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_flux_9_zscore_stats.npz',
            mean=mean,
            std=std
        )
        print('mean',mean,
              'std',std)
        np.savez('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_flux_9_zscore.npz',conditions=z_conditions)
    else:
        np.savez('/home/lim2/data/amie_npy/training_npy_logflux/conditions_flux_12.npz',conditions=conditions)
    print('finsh save conditions npz file')

if False:
    # potential train
    npy_files_train = sorted(
        [os.path.join('/home/lim2/data/amie_npy_new_all/training_npy_all', f) for f in os.listdir('/home/lim2/data/amie_npy_new_all/training_npy_all') if f.lower().endswith(".npy")]
    )

    npy_files_train_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_train])
    print('len(npy_files_train)',len(npy_files_train))
    
    conditions = [keep_dict[f] for f in npy_files_train_label]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]


    # Indices by column (for clarity)
    NOZ_IDX = np.array([2, 3, 8], dtype=int)                 # sinθc, cosθc, tilt_proxy  -> keep as-is
    Z_IDX   = np.setdiff1d(np.arange(conditions.shape[1]), NOZ_IDX)  # the rest -> z-score
    
    conds = conditions.copy()

    # Fit stats on z-scored subset only
    mean = conds[:, Z_IDX].mean(axis=0)
    std  = conds[:, Z_IDX].std(axis=0)
    zblock = (conds[:, Z_IDX] - mean) / (std + 1e-6)

    # Recombine: put z-scored block back; leave NOZ columns untouched
    z_conditions = conds.copy()
    z_conditions[:, Z_IDX] = zblock

    print("z_conditions:", z_conditions.shape)
    print("z-scored cols:", Z_IDX.tolist(), " (mean≈0, std≈1)")
    print("kept-as-is cols:", NOZ_IDX.tolist(), " (in [-1,1])")


    # Save stats (include indices + log info so you can apply identically at inference)
    np.savez('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_potential_9_zscore_stats.npz',
            mean=mean, std=std, z_idx=Z_IDX, noz_idx=NOZ_IDX, log_cols=np.array([0,1,4], int))

    np.savez('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_potential_9_zscore.npz',
            conditions=z_conditions)

    print('finsh save conditions npz file')


if False:
    # energy train
    npy_files_train = sorted(
        [os.path.join('/home/lim2/data/amie_npy_new_all/training_npy_all', f) for f in os.listdir('/home/lim2/data/amie_npy_new_all/training_npy_all') if f.lower().endswith(".npy")]
    )

    npy_files_train_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_train])
    print('len(npy_files_train)',len(npy_files_train))
    
    conditions = [keep_dict[f] for f in npy_files_train_label]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]


    # Indices by column (for clarity)
    NOZ_IDX = np.array([2, 3, 8], dtype=int)                 # sinθc, cosθc, tilt_proxy  -> keep as-is
    Z_IDX   = np.setdiff1d(np.arange(conditions.shape[1]), NOZ_IDX)  # the rest -> z-score
    
    conds = conditions.copy()

    # Fit stats on z-scored subset only
    mean = conds[:, Z_IDX].mean(axis=0)
    std  = conds[:, Z_IDX].std(axis=0)
    zblock = (conds[:, Z_IDX] - mean) / (std + 1e-6)

    # Recombine: put z-scored block back; leave NOZ columns untouched
    z_conditions = conds.copy()
    z_conditions[:, Z_IDX] = zblock

    print("z_conditions:", z_conditions.shape)
    print("z-scored cols:", Z_IDX.tolist(), " (mean≈0, std≈1)")
    print("kept-as-is cols:", NOZ_IDX.tolist(), " (in [-1,1])")


    # Save stats (include indices + log info so you can apply identically at inference)
    np.savez('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_energy_9_zscore_stats.npz',
            mean=mean, std=std, z_idx=Z_IDX, noz_idx=NOZ_IDX, log_cols=np.array([0,1,4], int))

    np.savez('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_energy_9_zscore.npz',
            conditions=z_conditions)

    print('finsh save conditions npz file')

if False:
    # joint train
    npy_files_train = sorted(
        [os.path.join('/data/minjing/Data/amie_npy_joint/training_npy_joint', f) for f in os.listdir('/data/minjing/Data/amie_npy_joint/training_npy_joint') if f.lower().endswith(".npy")]
    )

    npy_files_train_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_train])
    print('len(npy_files_train)',len(npy_files_train))
    
    conditions = [keep_dict[f] for f in npy_files_train_label]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]


    # Indices by column (for clarity)
    NOZ_IDX = np.array([1,2], dtype=int)                 # sinθc, cosθc, tilt_proxy  -> keep as-is
    Z_IDX   = np.setdiff1d(np.arange(conditions.shape[1]), NOZ_IDX)  # the rest -> z-score
    
    conds = conditions.copy()

    # Fit stats on z-scored subset only
    mean = conds[:, Z_IDX].mean(axis=0)
    std  = conds[:, Z_IDX].std(axis=0)
    zblock = (conds[:, Z_IDX] - mean) / (std + 1e-6)

    # Recombine: put z-scored block back; leave NOZ columns untouched
    z_conditions = conds.copy()
    z_conditions[:, Z_IDX] = zblock

    print("z_conditions:", z_conditions.shape)
    print("z-scored cols:", Z_IDX.tolist(), " (mean≈0, std≈1)")
    print("kept-as-is cols:", NOZ_IDX.tolist(), " (in [-1,1])")


    # Save stats (include indices + log info so you can apply identically at inference)
    # np.savez('/home/lim2/data/amie_npy_joint/training_npy_joint/conditions_joint_9_zscore_stats.npz',
    np.savez('/data/minjing/Data/amie_npy_joint/training_npy_joint/conditions_joint_9_zscore_stats.npz',
            mean=mean, std=std, z_idx=Z_IDX, noz_idx=NOZ_IDX, log_cols=np.array([0,1,4], int))

    # np.savez('/home/lim2/data/amie_npy_joint/training_npy_joint/conditions_joint_9_zscore.npz',
    np.savez('/data/minjing/Data/amie_npy_joint/training_npy_joint/conditions_joint_9_zscore.npz',
            conditions=z_conditions)

    print('finsh save conditions npz file')

if True:
    # AU/AL train
    npy_files_train = sorted(
        [os.path.join('/data/minjing/Data/amie_npy_joint_from105/training_npy_joint', f) for f in os.listdir('/data/minjing/Data/amie_npy_joint_from105/training_npy_joint') if f.lower().endswith(".npy")]
    )

    npy_files_train_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_train])
    print('len(npy_files_train)',len(npy_files_train))
    
    conditions = [keep_dict[f] for f in npy_files_train_label]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]


    # Indices by column (for clarity)
    NOZ_IDX = np.array([], dtype=int)                 # sinθc, cosθc, tilt_proxy  -> keep as-is
    Z_IDX   = np.setdiff1d(np.arange(conditions.shape[1]))  # the rest -> z-score
    
    conds = conditions.copy()

    # Fit stats on z-scored subset only
    mean = conds[:, Z_IDX].mean(axis=0)
    std  = conds[:, Z_IDX].std(axis=0)
    zblock = (conds[:, Z_IDX] - mean) / (std + 1e-6)

    # Recombine: put z-scored block back; leave NOZ columns untouched
    z_conditions = conds.copy()
    z_conditions[:, Z_IDX] = zblock

    print("z_conditions:", z_conditions.shape)
    print("z-scored cols:", Z_IDX.tolist(), " (mean≈0, std≈1)")
    print("kept-as-is cols:", NOZ_IDX.tolist(), " (in [-1,1])")


    # Save stats (include indices + log info so you can apply identically at inference)
    np.savez('/data/minjing/Data/amie_npy_joint_from105/training_npy_joint/conditions_joint_9_zscore_stats_AU_AL.npz',
            mean=mean, std=std, z_idx=Z_IDX, noz_idx=NOZ_IDX, log_cols=np.array([0,1,4], int))

    np.savez('/data/minjing/Data/amie_npy_joint_from105/training_npy_joint/conditions_joint_9_zscore_AU_AL.npz',
            conditions=z_conditions)

    print('finsh save conditions npz file')


#%%
if False:
    # flux test
    npy_files_test = sorted(
        [os.path.join('/home/lim2/data/amie_npy_new_all/sampling_npy_all', f) for f in os.listdir('/home/lim2/data/amie_npy_new_all/sampling_npy_all') if f.lower().endswith(".npy")]
    )

    npy_files_test_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_test])
    print('len(npy_files_test)',len(npy_files_test))

    selected_file = np.array(npy_files_test)
    formatted_time_list = [
        (
            datetime.strptime(f.split("/")[-1][:-4].split("_")[0], "%Y%m%d")
            + timedelta(minutes=int(f.split("/")[-1][:-4].split("_")[1]))
        ).strftime("%Y-%m-%d %H:%M")
        for f in selected_file
    ]
    formatted_time_array = np.array(formatted_time_list)
    time_objects = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M") for t in formatted_time_array])


    indices_sorted = np.argsort(time_objects)
    time_objects_sorted = np.array(time_objects)[indices_sorted]


    indices_filter = [i for i, t in enumerate(time_objects_sorted) if t.year == 2000]
    time_objects_sorted_filter = time_objects_sorted[indices_filter]


    hour_indices = []
    hour_seen = set()
    for i, t in enumerate(time_objects_sorted_filter):
        key = (t.year, t.month, t.day, t.hour)
        if key not in hour_seen:
            hour_indices.append(i)
            hour_seen.add(key)
    time_objects_sorted_filter_EveryHour = time_objects_sorted_filter[hour_indices]


    time_obj_index_map = {t: i for i, t in enumerate(time_objects)}

    matched_indices = [time_obj_index_map.get(t, 99999999999999999) for t in time_objects_sorted_filter_EveryHour]
    npy_files_test_label_EveryHour = npy_files_test_label[matched_indices]

    conditions = [keep_dict[f] for f in npy_files_test_label_EveryHour]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]

    phys = conditions[:, :-4]   # all except last 4
    periodic = conditions[:, -4:]  # last 4 stay untouched

    # Z-score only physical part
    zscore_stats = np.load('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_flux_9_zscore_stats.npz')
    mean_train = zscore_stats['mean']   # [K_phys]
    std_train  = zscore_stats['std']    # [K_phys]
    z_phys = (phys - mean_train) / (std_train+1e-6)

    # Optional: clip
    # z_phys = np.clip(z_phys, -5, 5)

    # Recombine
    z_conditions = np.concatenate([z_phys, periodic], axis=1)

    print(
        "z_conditions shape:", z_conditions.shape,
        "\nphys max:", z_phys.max(axis=0),
        "\nphys min:", z_phys.min(axis=0),
        "\nperiodic range:", periodic.min(), periodic.max()
    )
    np.savez('/home/lim2/data/amie_npy_new_all/sampling_npy_all/conditions_hour_flux_9_zscore.npz',conditions=z_conditions)


    print('finsh save conditions npz file',len(conditions))


if False:
    # potential test
    npy_files_test = sorted(
        [os.path.join('/home/lim2/data/amie_npy_new_all/sampling_npy_all', f) for f in os.listdir('/home/lim2/data/amie_npy_new_all/sampling_npy_all') if f.lower().endswith(".npy")]
    )

    npy_files_test_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_test])
    print('len(npy_files_test)',len(npy_files_test))

    selected_file = np.array(npy_files_test)
    formatted_time_list = [
        (
            datetime.strptime(f.split("/")[-1][:-4].split("_")[0], "%Y%m%d")
            + timedelta(minutes=int(f.split("/")[-1][:-4].split("_")[1]))
        ).strftime("%Y-%m-%d %H:%M")
        for f in selected_file
    ]
    formatted_time_array = np.array(formatted_time_list)
    time_objects = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M") for t in formatted_time_array])


    indices_sorted = np.argsort(time_objects)
    time_objects_sorted = np.array(time_objects)[indices_sorted]


    # indices_filter = [i for i, t in enumerate(time_objects_sorted) if t.year == 2001]
    indices_filter = [i for i, t in enumerate(time_objects_sorted) if t.year == 2000]
    time_objects_sorted_filter = time_objects_sorted[indices_filter]


    hour_indices = []
    hour_seen = set()
    for i, t in enumerate(time_objects_sorted_filter):
        key = (t.year, t.month, t.day, t.hour)
        if key not in hour_seen:
            hour_indices.append(i)
            hour_seen.add(key)
    time_objects_sorted_filter_EveryHour = time_objects_sorted_filter[hour_indices]


    time_obj_index_map = {t: i for i, t in enumerate(time_objects)}

    matched_indices = [time_obj_index_map.get(t, 99999999999999999) for t in time_objects_sorted_filter_EveryHour]
    npy_files_test_label_EveryHour = npy_files_test_label[matched_indices]

    conditions = [keep_dict[f] for f in npy_files_test_label_EveryHour]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]

    # Indices by column (for clarity)
    NOZ_IDX = np.array([2, 3, 8], dtype=int)                 # sinθc, cosθc, tilt_proxy  -> keep as-is
    Z_IDX   = np.setdiff1d(np.arange(conditions.shape[1]), NOZ_IDX)  # the rest -> z-score

    conds = conditions.copy()

    # Z-score only physical part
    zscore_stats = np.load('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_potential_9_zscore_stats.npz')
    mean_train = zscore_stats['mean']   # [K_phys]
    std_train  = zscore_stats['std']    # [K_phys]
    z_phys = (conds[:, Z_IDX] - mean_train) / (std_train+1e-6)


    # Recombine
    z_conditions = conds.copy()
    z_conditions[:, Z_IDX] = z_phys

    print("z_conditions:", z_conditions.shape)
    print("z-scored cols:", Z_IDX.tolist(), " (mean≈0, std≈1)")
    print("kept-as-is cols:", NOZ_IDX.tolist(), " (in [-1,1])")


    np.savez('/home/lim2/data/amie_npy_new_all/sampling_npy_all/conditions_hour_potential_9_zscore.npz',conditions=z_conditions)


    print('finsh save conditions npz file',len(conditions))


if False:
    # energy test
    npy_files_test = sorted(
        [os.path.join('/home/lim2/data/amie_npy_new_all/sampling_npy_all', f) for f in os.listdir('/home/lim2/data/amie_npy_new_all/sampling_npy_all') if f.lower().endswith(".npy")]
    )

    npy_files_test_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_test])
    print('len(npy_files_test)',len(npy_files_test))

    selected_file = np.array(npy_files_test)
    formatted_time_list = [
        (
            datetime.strptime(f.split("/")[-1][:-4].split("_")[0], "%Y%m%d")
            + timedelta(minutes=int(f.split("/")[-1][:-4].split("_")[1]))
        ).strftime("%Y-%m-%d %H:%M")
        for f in selected_file
    ]
    formatted_time_array = np.array(formatted_time_list)
    time_objects = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M") for t in formatted_time_array])


    indices_sorted = np.argsort(time_objects)
    time_objects_sorted = np.array(time_objects)[indices_sorted]


    indices_filter = [i for i, t in enumerate(time_objects_sorted) if t.year == 2000]
    time_objects_sorted_filter = time_objects_sorted[indices_filter]


    hour_indices = []
    hour_seen = set()
    for i, t in enumerate(time_objects_sorted_filter):
        key = (t.year, t.month, t.day, t.hour)
        if key not in hour_seen:
            hour_indices.append(i)
            hour_seen.add(key)
    time_objects_sorted_filter_EveryHour = time_objects_sorted_filter[hour_indices]


    time_obj_index_map = {t: i for i, t in enumerate(time_objects)}

    matched_indices = [time_obj_index_map.get(t, 99999999999999999) for t in time_objects_sorted_filter_EveryHour]
    npy_files_test_label_EveryHour = npy_files_test_label[matched_indices]

    conditions = [keep_dict[f] for f in npy_files_test_label_EveryHour]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]

    # Indices by column (for clarity)
    NOZ_IDX = np.array([2, 3, 8], dtype=int)                 # sinθc, cosθc, tilt_proxy  -> keep as-is
    Z_IDX   = np.setdiff1d(np.arange(conditions.shape[1]), NOZ_IDX)  # the rest -> z-score

    conds = conditions.copy()

    # Z-score only physical part
    zscore_stats = np.load('/home/lim2/data/amie_npy_new_all/training_npy_all/conditions_energy_9_zscore_stats.npz')
    mean_train = zscore_stats['mean']   # [K_phys]
    std_train  = zscore_stats['std']    # [K_phys]
    z_phys = (conds[:, Z_IDX] - mean_train) / (std_train+1e-6)


    # Recombine
    z_conditions = conds.copy()
    z_conditions[:, Z_IDX] = z_phys

    print("z_conditions:", z_conditions.shape)
    print("z-scored cols:", Z_IDX.tolist(), " (mean≈0, std≈1)")
    print("kept-as-is cols:", NOZ_IDX.tolist(), " (in [-1,1])")


    np.savez('/home/lim2/data/amie_npy_new_all/sampling_npy_all/conditions_hour_energy_9_zscore.npz',conditions=z_conditions)


    print('finsh save conditions npz file',len(conditions))

if False:
    # joint test
    npy_files_test = sorted(
        [os.path.join('/data/minjing/Data/amie_npy_joint/sampling_npy_joint', f) for f in os.listdir('/data/minjing/Data/amie_npy_joint/sampling_npy_joint') if f.lower().endswith(".npy")]
    )

    npy_files_test_label = np.array([f.split('/')[-1].split('.')[0] for f in npy_files_test])
    print('len(npy_files_test)',len(npy_files_test))

    selected_file = np.array(npy_files_test)
    formatted_time_list = [
        (
            datetime.strptime(f.split("/")[-1][:-4].split("_")[0], "%Y%m%d")
            + timedelta(minutes=int(f.split("/")[-1][:-4].split("_")[1]))
        ).strftime("%Y-%m-%d %H:%M")
        for f in selected_file
    ]
    formatted_time_array = np.array(formatted_time_list)
    time_objects = np.array([datetime.strptime(t, "%Y-%m-%d %H:%M") for t in formatted_time_array])


    indices_sorted = np.argsort(time_objects)
    time_objects_sorted = np.array(time_objects)[indices_sorted]


    # indices_filter = np.array([i for i, t in enumerate(time_objects_sorted) if (t.year == 2000) or (t.year == 2003)])
    # indices_filter = np.array([i for i, t in enumerate(time_objects_sorted) if (t.year == 2003)])
    # indices_filter = np.array([i for i, t in enumerate(time_objects_sorted) if (t.year == 2000) and (t.month == 4) and ((t.day == 6) or (t.day == 7) or (t.day == 8))])
    indices_filter = np.array([i for i, t in enumerate(time_objects_sorted) if ((t.year == 2000) and (t.month == 4) and ((t.day == 6) or (t.day == 7) or (t.day == 8) or (t.day == 9) )) or
                                                                                    ((t.year == 2003) and (t.month == 11) and ((t.day == 19) or (t.day == 20) or (t.day == 21) or (t.day == 22))) or 
                                                                                    ((t.year == 2003) and (t.month == 3) and ((t.day == 28) or (t.day == 29) or (t.day == 30) or (t.day == 31))) or
                                                                                    ((t.year == 2003) and (t.month == 4) and ((t.day == 1) or (t.day == 2) or (t.day == 3))) or
                                                                                    ((t.year == 2003) and (t.month == 5) and ((t.day == 28) or (t.day == 29) or (t.day == 30) or (t.day == 31))) or
                                                                                    ((t.year == 2003) and (t.month == 6) and (t.day == 1))
                                                                                    ])
    time_objects_sorted_filter = time_objects_sorted[indices_filter]


    hour_indices = []
    hour_seen = set()
    for i, t in enumerate(time_objects_sorted_filter):
        # key = (t.year, t.month, t.day, t.hour)
        key = (t.year, t.month, t.day, t.hour, t.minute)
        if key not in hour_seen:
            hour_indices.append(i)
            hour_seen.add(key)
    time_objects_sorted_filter_EveryHour = time_objects_sorted_filter[hour_indices]


    time_obj_index_map = {t: i for i, t in enumerate(time_objects)}

    matched_indices = [time_obj_index_map.get(t, 99999999999999999) for t in time_objects_sorted_filter_EveryHour]
    npy_files_test_label_EveryHour = npy_files_test_label[matched_indices]

    # Filter to only labels that exist in keep_dict (some may be missing due to NaN removal)
    valid_labels = [f for f in npy_files_test_label_EveryHour if f in keep_dict]
    print(f"Filtered: {len(valid_labels)}/{len(npy_files_test_label_EveryHour)} labels found in keep_dict")
    conditions = [keep_dict[f] for f in valid_labels]
    conditions = np.array(conditions, dtype=np.float32)   # shape [N, K]

    # Indices by column (for clarity)
    NOZ_IDX = np.array([1,2], dtype=int)                 # sinθc, cosθc, tilt_proxy  -> keep as-is
    Z_IDX   = np.setdiff1d(np.arange(conditions.shape[1]), NOZ_IDX)  # the rest -> z-score

    conds = conditions.copy()

    # Z-score only physical part
    zscore_stats = np.load('/data/minjing/Data/amie_npy_joint/training_npy_joint/conditions_joint_9_zscore_stats.npz')
    mean_train = zscore_stats['mean']   # [K_phys]
    std_train  = zscore_stats['std']    # [K_phys]
    z_phys = (conds[:, Z_IDX] - mean_train) / (std_train+1e-6)


    # Recombine
    z_conditions = conds.copy()
    z_conditions[:, Z_IDX] = z_phys

    print("z_conditions:", z_conditions.shape)
    print("z-scored cols:", Z_IDX.tolist(), " (mean≈0, std≈1)")
    print("kept-as-is cols:", NOZ_IDX.tolist(), " (in [-1,1])")


    np.savez('/data/minjing/Data/amie_npy_joint/sampling_npy_joint/conditions_4storm_minute_joint_9_zscore.npz',conditions=z_conditions)


    print('finsh save conditions npz file',len(conditions))


# %%
if False:
    import os
    import json
    import numpy as np
    data_dir = '/home/lim2/data/amie_npy/sampling_npy_logflux/'
    cond_path = os.path.join(data_dir, "conditions_hour_9_zscore.npz")
    npy_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".npy")]
    )
    cond_load = np.load(cond_path)['conditions']
    print(cond_load[0])
    print('len(npy_files)',len(npy_files),'cond_load.shape',cond_load.shape)
    print('test_npy_values',
          'mean energy',np.load(npy_files[0])[0].min(),np.load(npy_files[0])[0].max(),
          'energy flux',np.load(npy_files[0])[1].min(),np.load(npy_files[0])[1].max(),
          'potential',np.load(npy_files[0])[2].min(),np.load(npy_files[0])[2].max())

    # with open('/home/lim2/data/amie_npy/sampling_npy_logflux/amie_range_test20.json', 'r') as f:
    #     data_range = json.load(f)
    # print('data_range',data_range)
    
# %%
if False:
    import os
    import json
    import numpy as np
    data_dir = '/home/lim2/data/amie_npy/training_npy_logflux/'
    cond_path = os.path.join(data_dir, "conditions_flux_9_zscore.npz")
    npy_files = sorted(
        [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".npy")]
    )
    cond_load = np.load(cond_path)['conditions']
    print('len(npy_files)',len(npy_files),'cond_load.shape',cond_load.shape)
    print(cond_load[0])
    print('test_npy_values',
          'mean energy',np.load(npy_files[0])[0].min(),np.load(npy_files[0])[0].max(),
          'energy flux',np.load(npy_files[0])[1].min(),np.load(npy_files[0])[1].max(),
          'potential',np.load(npy_files[0])[2].min(),np.load(npy_files[0])[2].max())

    # with open('/home/lim2/data/amie_npy/ranges/amie_range_train80.json', 'r') as f:
    #     data_range = json.load(f)
    # print('data_range',data_range)
# %%
