#%%
# npy_pipeline.py
import os, json, glob, argparse
import numpy as np
from collections import defaultdict
#%%
def load_cond_keys(cond_path):
    d = np.load(cond_path, allow_pickle=True)
    return set(d.files)

def year_from_npz(path):
    return os.path.basename(path).split('_')[-1].split('.')[0]

def pick_year_file(data_root, year):
    f = os.path.join(data_root, f"AllData_{year}.npz")
    if not os.path.isfile(f):
        raise FileNotFoundError(f"not found: {f}")
    return f

def load_names_from_paths(npz_file):
    # Only read the small 'paths' object array with pickle enabled
    with np.load(npz_file, allow_pickle=True) as d:
        paths = d['paths']
        names = np.array([os.path.basename(str(p)).split('.')[0] for p in paths], dtype=object)
    return names

#%%
def scan_year_minmax(npz_file, cond_keys, years_set, mode, block=2048):
    """Per-channel min/max via streaming; load 'paths' separately with pickle."""
    vmins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    vmaxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    # 1) get names (needs allow_pickle=True)
    names = load_names_from_paths(npz_file)
    year_mask = np.isin(np.array([n[:4] for n in names]), list(years_set))
    cond_mask = np.array([n in cond_keys for n in names])
    mask = year_mask & cond_mask
    # print('year_mask number',np.sum(year_mask))
    # print('cond_mask number',np.sum(cond_mask))
    # print('mask number',np.sum(mask))
    idx = np.nonzero(mask)[0]
    count = int(idx.size)
    if count == 0:
        return vmins, vmaxs, 0

    # 2) reopen for large arrays with mmap + allow_pickle=False
    with np.load(npz_file, allow_pickle=False, mmap_mode='r') as d:
        # energy
        arr = d['energy']
        if mode == 'train':
            eps = 1e-2
            for s in range(0, count, block):
                sl = idx[s:s+block]
                chunk = arr[sl]
                chunk = np.where(chunk < 0, 0, chunk)
                chunk = np.log10(chunk + eps)
                vmins[0] = min(vmins[0], np.nanmin(chunk))
                vmaxs[0] = max(vmaxs[0], np.nanmax(chunk))
        else:
            for s in range(0, count, block):
                sl = idx[s:s+block]
                chunk = arr[sl]
                vmins[0] = min(vmins[0], np.nanmin(chunk))
                vmaxs[0] = max(vmaxs[0], np.nanmax(chunk))

        # flux (log for train)
        arr = d['flux']
        if mode == "train":
            eps = 1e-2
            for s in range(0, count, block):
                sl = idx[s:s+block]
                chunk = arr[sl]
                chunk = np.where(chunk < 0, 0, chunk)
                chunk = np.log10(chunk + eps)
                vmins[1] = min(vmins[1], np.nanmin(chunk))
                vmaxs[1] = max(vmaxs[1], np.nanmax(chunk))
        else:
            for s in range(0, count, block):
                sl = idx[s:s+block]
                chunk = arr[sl]
                vmins[1] = min(vmins[1], np.nanmin(chunk))
                vmaxs[1] = max(vmaxs[1], np.nanmax(chunk))

        # pot
        arr = d['pot']
        for s in range(0, count, block):
            sl = idx[s:s+block]
            chunk = arr[sl]
            vmins[2] = min(vmins[2], np.nanmin(chunk))
            vmaxs[2] = max(vmaxs[2], np.nanmax(chunk))

    return vmins, vmaxs, count



def scan_year_minmax_allPara(npz_file, cond_keys_list, years_set, mode, block=2048):


    vmins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    vmaxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    # 1) get names (needs allow_pickle=True)
    
    with np.load(npz_file, allow_pickle=False, mmap_mode='r') as d:

        N = d['energy'].shape[0]
        mask_Paras = np.zeros((3,N), dtype=bool)
        names = load_names_from_paths(npz_file)
        year_mask = np.isin(np.array([n[:4] for n in names]), list(years_set))

        for i, cond_keys in enumerate(cond_keys_list):
            cond_mask = np.array([n in cond_keys for n in names])
            mask_Paras[i,:] = year_mask & cond_mask


        mask_allPara = mask_Paras.all(axis=0)
        count = np.sum(mask_allPara)
        if count != 0:
            idx = np.nonzero(mask_allPara)[0]
            for s in range(0, idx.size, block):
                sl = idx[s:s+block]
                if (mode == 'train') or (mode == 'validate'):
                    eps = 1e-2
                    # energy
                    chunk = d['energy'][sl]
                    chunk = np.where(chunk < 0, 0, chunk)
                    chunk = np.log10(chunk + eps)
                    vmins[0] = min(vmins[0], np.nanmin(chunk))
                    vmaxs[0] = max(vmaxs[0], np.nanmax(chunk))
                    # flux
                    chunk = d['flux'][sl]
                    chunk = np.where(chunk < 0, 0, chunk)
                    chunk = np.log10(chunk + eps)
                    vmins[1] = min(vmins[1], np.nanmin(chunk))
                    vmaxs[1] = max(vmaxs[1], np.nanmax(chunk))

                elif mode == 'test':
                    # energy
                    chunk = d['energy'][sl]
                    vmins[0] = min(vmins[0], np.nanmin(chunk))
                    vmaxs[0] = max(vmaxs[0], np.nanmax(chunk))
                    # flux
                    chunk = d['flux'][sl]
                    vmins[1] = min(vmins[1], np.nanmin(chunk))
                    vmaxs[1] = max(vmaxs[1], np.nanmax(chunk))

                # pot
                chunk = d['pot'][sl]
                vmins[2] = min(vmins[2], np.nanmin(chunk))
                vmaxs[2] = max(vmaxs[2], np.nanmax(chunk))

        else:
            return vmins, vmaxs, 0


    return vmins, vmaxs, count




def write_year_npys_old(npz_file, cond_keys, years_set, mode, range_json, out_dir, block=2048):
    with open(range_json, 'r') as f:
        rng = json.load(f)
    vmins = np.array(rng["vmins"], dtype=np.float32)
    vmaxs = np.array(rng["vmaxs"], dtype=np.float32)
    den = vmaxs - vmins
    den[den == 0] = 1.0

    os.makedirs(out_dir, exist_ok=True)
    wrote = 0

    with np.load(npz_file, allow_pickle=False, mmap_mode='r') as d:
        names = np.array([os.path.basename(str(p)).split('.')[0] for p in d['paths']], dtype=object)
        year_mask = np.isin(np.array([n[:4] for n in names]), list(years_set))
        cond_mask = np.array([n in cond_keys for n in names])
        mask = year_mask & cond_mask
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return 0

        eps = 1e-2
        for s in range(0, idx.size, block):
            sl = idx[s:s+block]
            sel_names = names[sl]

            # read in chunks
            e = d['energy'][sl].astype(np.float32, copy=False)
            x = d['flux'][sl]
            p = d['pot'][sl].astype(np.float32, copy=False)

            if mode == "train":
                x = np.where(x < 0, 0, x)
                x = np.log10(x + eps)

                e = np.where(e < 0, 0, e)
                e = np.log10(e + eps)

            x = x.astype(np.float32, copy=False)
            e = e.astype(np.float32, copy=False)

            # write sample by sample
            for i, nm in enumerate(sel_names):
                arr = np.stack([e[i], x[i], p[i]], axis=0)
                arr = (arr - vmins[:, None, None]) / den[:, None, None] * 2.0 - 1.0
                np.save(os.path.join(out_dir, f"{nm}.npy"), arr.astype(np.float32))
                wrote += 1

    return wrote


def write_year_npys(npz_file, cond_keys, years_set, mode, range_json, out_dir, block=2048):
    with open(range_json, 'r') as f:
        rng = json.load(f)
    vmins = np.array(rng["vmins"], dtype=np.float32)
    vmaxs = np.array(rng["vmaxs"], dtype=np.float32)
    den = vmaxs - vmins
    den[den == 0] = 1.0
    os.makedirs(out_dir, exist_ok=True)

    # names first (pickle)
    names = load_names_from_paths(npz_file)
    year_mask = np.isin(np.array([n[:4] for n in names]), list(years_set))
    cond_mask = np.array([n in cond_keys for n in names])
    mask = year_mask & cond_mask
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return 0

    wrote = 0
    eps = 1e-2
    # reopen big arrays with mmap
    with np.load(npz_file, allow_pickle=False, mmap_mode='r') as d:
        for s in range(0, idx.size, block):
            sl = idx[s:s+block]
            sel_names = names[sl]

            e = d['energy'][sl].astype(np.float32, copy=False)
            x = d['flux'][sl]
            p = d['pot'][sl].astype(np.float32, copy=False)

            if (mode == "train") or (mode == "validate"):
                x = np.where(x < 0, 0, x)
                x = np.log10(x + eps)

                e = np.where(e < 0, 0, e)
                e = np.log10(e + eps)

            x = x.astype(np.float32, copy=False)
            e = e.astype(np.float32, copy=False)

            for i, nm in enumerate(sel_names):
                arr = np.stack([e[i], x[i], p[i]], axis=0)
                arr = (arr - vmins[:, None, None]) / den[:, None, None] * 2.0 - 1.0
                np.save(os.path.join(out_dir, f"{nm}.npy"), arr.astype(np.float32))
                wrote += 1

    return wrote

def write_year_npys_allPara(npz_file, cond_keys_list, years_set, mode, range_json, out_dir, block=2048):
    with open(range_json, 'r') as f:
        rng = json.load(f)
    vmins = np.array(rng["vmins"], dtype=np.float32)
    vmaxs = np.array(rng["vmaxs"], dtype=np.float32)
    den = vmaxs - vmins
    den[den == 0] = 1.0
    os.makedirs(out_dir, exist_ok=True)

    # names first (pickle)
    names = load_names_from_paths(npz_file)
    year_mask = np.isin(np.array([n[:4] for n in names]), list(years_set))

    N = len(names)
    mask_Paras = np.zeros((len(cond_keys_list),N), dtype=bool)
    for i, cond_keys in enumerate(cond_keys_list):
        cond_mask = np.array([n in cond_keys for n in names])
        mask_Paras[i,:] = year_mask & cond_mask

    mask_allPara = mask_Paras.all(axis=0)
    count = np.sum(mask_allPara)
    if count == 0:
        return 0
    idx = np.nonzero(mask_allPara)[0]
    
    wrote = 0
    eps = 1e-2
    # reopen big arrays with mmap
    with np.load(npz_file, allow_pickle=False, mmap_mode='r') as d:
        for s in range(0, idx.size, block):
            sl = idx[s:s+block]
            sel_names = names[sl]

            e = d['energy'][sl].astype(np.float32, copy=False)
            x = d['flux'][sl]
            p = d['pot'][sl].astype(np.float32, copy=False)

            if (mode == "train") or (mode == "validate"):
                x = np.where(x < 0, 0, x)
                x = np.log10(x + eps)

                e = np.where(e < 0, 0, e)
                e = np.log10(e + eps)

            x = x.astype(np.float32, copy=False)
            e = e.astype(np.float32, copy=False)

            for i, nm in enumerate(sel_names):
                arr = np.stack([e[i], x[i], p[i]], axis=0)
                arr = (arr - vmins[:, None, None]) / den[:, None, None] * 2.0 - 1.0
                np.save(os.path.join(out_dir, f"{nm}.npy"), arr.astype(np.float32))
                wrote += 1

    return wrote

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["scan", "write", "reduce"], required=True)
    ap.add_argument("--mode", choices=["train", "test", "validate"], required=True)
    ap.add_argument("--year", type=int, help="YYYY; required for scan/write")
    ap.add_argument("--phys", choices=["logflux", "logenergy", "potential","all","joint"], required=True)
    ap.add_argument("--block", type=int, default=2048, help="chunk size for streaming I/O")

    # parse first
    args = ap.parse_args()

    # then set the paths from phys
    key_parameter = args.phys
    folder_name = 'amie_npy_joint'
    if not hasattr(args, "cond_path") or args.cond_path is None:
        args.cond_path = f"/data/minjing/Data/{folder_name}/1999_2010_condition/keep_dict_{key_parameter}.npz"
    if not hasattr(args, "data_root") or args.data_root is None:
        args.data_root = f"/data/minjing/Data/{folder_name}/1999_2010_all_{key_parameter}"
    if not hasattr(args, "out_dir_train") or args.out_dir_train is None:
        args.out_dir_train = f"/data/minjing/Data/{folder_name}/training_npy_{key_parameter}"
    if not hasattr(args, "out_dir_test") or args.out_dir_test is None:
        args.out_dir_test = f"/data/minjing/Data/{folder_name}/sampling_npy_{key_parameter}"
    if not hasattr(args, "out_dir_validate") or args.out_dir_validate is None:
        args.out_dir_validate = f"/data/minjing/Data/{folder_name}/validation_npy_{key_parameter}"
    if not hasattr(args, "range_dir") or args.range_dir is None:
        args.range_dir = f"/data/minjing/Data/{folder_name}/ranges_{key_parameter}"


    test_years  = {'2000','2003'}
    train_years = {'1999','2001','2002','2004','2005','2006'}

    # validate_years = {'2004'}
    # train_years = {'1999','2000','2002','2003','2005','2006','2007','2008','2010'}
    # test_years  = {'2001'}

    os.makedirs(args.range_dir, exist_ok=True)
    if args.phys == "all":
        base_path = f"/data/minjing/Data/{folder_name}/1999_2010_condition/keep_dict_"
        cond_keys_list = [load_cond_keys(f"{base_path}{phys}.npz") for phys in ["logflux", "logenergy", "potential"]]
    else:
        cond_keys = load_cond_keys(args.cond_path)

    if args.phase == "scan":
        if args.year is None:
            raise SystemExit("--year is required for phase=scan")
        year = str(args.year)
        npz_file = pick_year_file(args.data_root, year)
        years_set = validate_years if args.mode=="validate" else test_years if args.mode=="test" else train_years
        if args.phys == "all":
            vmins, vmaxs, count = scan_year_minmax_allPara(npz_file, cond_keys_list, years_set, args.mode, args.block)
        else:
            vmins, vmaxs, count = scan_year_minmax(npz_file, cond_keys, years_set, args.mode, args.block)
        out_json = os.path.join(args.range_dir, f"minmax_{args.mode}_{year}.json")
        with open(out_json, "w") as f:
            json.dump({"year":year, "count":int(count), "vmins":vmins.tolist(), "vmaxs":vmaxs.tolist()}, f, indent=2)
        print(f"[scan] {year} count={count} -> {out_json}")

    elif args.phase == "reduce":
        # merge the minmax_* of all years into the global amie_range_*.json
        files = sorted(glob.glob(os.path.join(args.range_dir, f"minmax_{args.mode}_*.json")))
        if not files:
            raise SystemExit("no per-year minmax jsons to reduce")
        vmins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        vmaxs = np.array([-np.inf,-np.inf,-np.inf], dtype=np.float64)
        total = 0
        for f in files:
            with open(f,'r') as j:
                d = json.load(j)
            if d["count"] <= 0: 
                continue
            vmins = np.minimum(vmins, np.array(d["vmins"]))
            vmaxs = np.maximum(vmaxs, np.array(d["vmaxs"]))
            total += d["count"]

        out_json = os.path.join(args.range_dir, f"amie_range_{'train80' if args.mode=='train' else 'test20' if args.mode=='test' else 'validate20'}.json")
        with open(out_json,"w") as f:
            json.dump({"channels":["mean energy","energy flux","potential"],
                       "vmins":vmins.tolist(),"vmaxs":vmaxs.tolist(),"count":total}, f, indent=2)
        print(f"[reduce] total={total} -> {out_json}")

    elif args.phase == "write":
        if args.year is None:
            raise SystemExit("--year is required for phase=write")
        year = str(args.year)
        npz_file = pick_year_file(args.data_root, year)
        years_set = validate_years if args.mode=="validate" else test_years if args.mode=="test" else train_years
        range_json = os.path.join(args.range_dir, f"amie_range_{'test20' if args.mode=='test' else 'train80'}.json")
        if not os.path.isfile(range_json):
            raise SystemExit(f"missing global range json: {range_json}")
        out_dir = args.out_dir_validate if args.mode=="validate" else args.out_dir_test if args.mode=="test" else args.out_dir_train
        if args.phys == "all":
            n = write_year_npys_allPara(npz_file, cond_keys_list, years_set, args.mode, range_json, out_dir, args.block)
        else:
            n = write_year_npys(npz_file, cond_keys, years_set, args.mode, range_json, out_dir, args.block)
        print(f"[write] {year} wrote {n} npy files -> {out_dir}")

if __name__ == "__main__":
    main()
