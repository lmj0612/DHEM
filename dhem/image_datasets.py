from PIL import Image
import blobfile as bf
from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os # mj_20250529
import torch # mj_20250529


def load_data(*, data_dir, batch_size, image_size, use_structured_cond, class_cond=False, deterministic=False, spatial_basis=None, use_hp_weights=False, mlt_grid=None, cond_file=None, cols_to_use=None, use_oval_boundaries=False, exclude_years=None, prev_cfg=None): # mj_20250529, mj_20260205 added hp_weights, mj_20260225 added mlt_grid, mj_20260301 added cond_file/cols_to_use, mj_20260302 added oval_boundaries, mj_20260901 added exclude_years, mj_20260927 added prev_cfg (previous-frame conditioning)
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param use_hp_weights: if True, load and return HP-based sample weights.
    """

    print("IMPROVED DIFFUSION 1D_flux: [DEBUG] Custom load_data is called with use_structured_cond =", use_structured_cond)

    if not data_dir:
        raise ValueError("unspecified data directory")
    
    if use_structured_cond: # mj_20250529
        # cond_path = os.path.join(data_dir, "conditions_flux_9_zscore.npz")
        # mj_20260301: allow overriding cond file and column selection via params
        _cond_filename = cond_file if cond_file is not None else "conditions_joint_9_zscore_SMU_SML.npz"  # mj_20260301
        cond_path = os.path.join(data_dir, _cond_filename)  # mj_20260301
        png_files = sorted(
            # [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".png")]
            [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(".npy")]
        )
        # conds = np.load(cond_path).astype(np.float32)
        conds_all = np.load(cond_path)['conditions'].astype(np.float32)

        # mj_20260301: configurable column selection (default: all 48 for SML/SMU)
        _cols = cols_to_use if cols_to_use is not None else list(range(conds_all.shape[1]))  # mj_20260301
        conds = conds_all[:, _cols]   # shape: [N, cond_dim]

        assert len(png_files) == len(conds), "number of PNG files != rows of conditions.npz"

        # mj_20260901: optionally exclude whole years (filenames are YYYYMMDD_*.npy).
        # Applied AFTER the count assert so conds rows stay aligned to the sorted
        # full listing; hp_weights/oval_boundaries lookups below are filename-keyed
        # and therefore unaffected. exclude_years=None keeps the old behavior.
        if exclude_years:  # mj_20260901
            _excl = tuple(str(y) for y in exclude_years)  # mj_20260901
            _keep = [i for i, f in enumerate(png_files)
                     if not os.path.basename(f).startswith(_excl)]  # mj_20260901
            print(f"[Exclude Years] {_excl}: keeping {len(_keep)}/{len(png_files)} samples")  # mj_20260901
            png_files = [png_files[i] for i in _keep]  # mj_20260901
            conds = conds[_keep]  # mj_20260901

        # Load HP weights if requested (mj_20260205)
        hp_weights = None
        if use_hp_weights:
            hp_weights_path = os.path.join(data_dir, "hp_weights.npz")
            if os.path.exists(hp_weights_path):
                hp_data = np.load(hp_weights_path, allow_pickle=True)
                hp_filenames = hp_data['filenames']
                hp_sample_weights = hp_data['sample_weights'].astype(np.float32)
                
                # Create mapping from filename to weight
                filename_to_weight = {fn: w for fn, w in zip(hp_filenames, hp_sample_weights)}
                
                # Match weights to png_files order
                hp_weights = np.ones(len(png_files), dtype=np.float32)
                for i, fpath in enumerate(png_files):
                    fname = os.path.basename(fpath)
                    if fname in filename_to_weight:
                        hp_weights[i] = filename_to_weight[fname]
                
                print(f"[HP Weights] Loaded weights for {len(hp_filenames)} samples")
                print(f"[HP Weights] Weight range: [{hp_weights.min():.4f}, {hp_weights.max():.4f}]")
            else:
                print(f"[HP Weights] WARNING: hp_weights.npz not found at {hp_weights_path}")
                print(f"[HP Weights] Run compute_hp_weights.py first to generate weights.")

        # Load oval boundaries if requested (mj_20260302)
        oval_eb = None  # mj_20260302
        oval_pb = None  # mj_20260302
        if use_oval_boundaries:  # mj_20260302
            oval_path = os.path.join(data_dir, "oval_boundaries.npz")  # mj_20260302
            # mj_20260927: TRAIN_OVAL_FILE overrides the location (per-cat dirs hold byte-identical copies of the
            # joint npy files, so the joint oval_boundaries.npz can be reused; lookup is filename-keyed)
            oval_path = os.environ.get("TRAIN_OVAL_FILE", "") or oval_path  # mj_20260927
            if os.path.exists(oval_path):  # mj_20260302
                oval_data = np.load(oval_path, allow_pickle=True)  # mj_20260302
                oval_filenames = oval_data['filenames']  # mj_20260302
                oval_eb_all = oval_data['eb']  # [N, 32] uint8  # mj_20260302
                oval_pb_all = oval_data['pb']  # [N, 32] uint8  # mj_20260302
                # Build mapping from filename to index  # mj_20260302
                fname_to_idx = {fn: idx for idx, fn in enumerate(oval_filenames)}  # mj_20260302
                # Match to png_files order  # mj_20260302
                oval_eb = np.full((len(png_files), 32), 255, dtype=np.uint8)  # mj_20260302
                oval_pb = np.full((len(png_files), 32), 255, dtype=np.uint8)  # mj_20260302
                matched = 0  # mj_20260302
                for i, fpath in enumerate(png_files):  # mj_20260302
                    fname = os.path.basename(fpath)  # mj_20260302
                    if fname in fname_to_idx:  # mj_20260302
                        idx = fname_to_idx[fname]  # mj_20260302
                        oval_eb[i] = oval_eb_all[idx]  # mj_20260302
                        oval_pb[i] = oval_pb_all[idx]  # mj_20260302
                        matched += 1  # mj_20260302
                print(f"[Oval Boundaries] Loaded {matched}/{len(png_files)} boundary records from {oval_path}")  # mj_20260302, mj_20260927
                assert matched >= 0.99 * len(png_files), f"[Oval Boundaries] only {matched}/{len(png_files)} files matched -- wrong TRAIN_OVAL_FILE?"  # mj_20260927
            else:  # mj_20260302
                print(f"[Oval Boundaries] WARNING: oval_boundaries.npz not found at {oval_path}")  # mj_20260302
                print(f"[Oval Boundaries] Run compute_oval_boundaries.py first.")  # mj_20260302

        # dataset = ConditionalImageDataset(
        #     image_size,
        #     png_files,
        #     conds,
        #     shard=MPI.COMM_WORLD.Get_rank(),
        #     num_shards=MPI.COMM_WORLD.Get_size(),
        # )
        prev_set = None  # mj_20260927
        if prev_cfg:  # mj_20260927 previous-frame conditioning: the frame delta minutes earlier is looked up in the JOINT dir (a category dir holds only its own minutes)
            _pd = prev_cfg['dir']; prev_set = set(f for f in os.listdir(_pd) if f.endswith('.npy'))  # mj_20260927
            print(f"[PrevFrame] delta 1..{prev_cfg['delta_max']} min, drop {prev_cfg['drop']}, noise U(0,{prev_cfg['noise']}), lookup {len(prev_set)} files in {_pd}")  # mj_20260927
        dataset = ConditionalNpyDataset(
            image_size,
            png_files,
            conds,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
            spatial_basis=spatial_basis, # mj_20250911
            hp_weights=hp_weights, # mj_20260205
            mlt_grid=mlt_grid, # mj_20260225
            oval_eb=oval_eb, # mj_20260302
            oval_pb=oval_pb, # mj_20260302
            prev_cfg=prev_cfg, prev_set=prev_set, # mj_20260927
        )


    else:
        all_files = _list_image_files_recursively(data_dir)
        classes = None
        if class_cond:
            # Assume classes are the first part of the filename,
            # before an underscore.
            class_names = [bf.basename(path).split("_")[0] for path in all_files]
            sorted_classes = {x: i for i, x in enumerate(sorted(set(class_names)))}
            classes = [sorted_classes[x] for x in class_names]
        dataset = ImageDataset(
            image_size,
            all_files,
            classes=classes,
            shard=MPI.COMM_WORLD.Get_rank(),
            num_shards=MPI.COMM_WORLD.Get_size(),
        )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, resolution, image_paths, classes=None, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.local_classes = None if classes is None else classes[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()

        # We are not on a new enough PIL to support the `reducing_gap`
        # argument, which uses BOX downsampling at powers of two first.
        # Thus, we do it by hand to improve downsample quality.
        while min(*pil_image.size) >= 2 * self.resolution:
            pil_image = pil_image.resize(
                tuple(x // 2 for x in pil_image.size), resample=Image.BOX
            )

        scale = self.resolution / min(*pil_image.size)
        pil_image = pil_image.resize(
            tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
        )

        arr = np.array(pil_image.convert("RGB"))
        crop_y = (arr.shape[0] - self.resolution) // 2
        crop_x = (arr.shape[1] - self.resolution) // 2
        arr = arr[crop_y : crop_y + self.resolution, crop_x : crop_x + self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict



class ConditionalImageDataset(Dataset): # mj_20250529
    def __init__(self, resolution, image_paths, cond_array, shard=0, num_shards=1):
        super().__init__()
        self.resolution = resolution
        # MPI sharding
        self.local_images = image_paths[shard:][::num_shards]
        self.local_cond   = cond_array[shard:][::num_shards]

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            img = Image.open(f)
            img.load()

        # --- same downsampling / cropping as ImageDataset ---
        while min(*img.size) >= 2 * self.resolution:
            img = img.resize(tuple(x // 2 for x in img.size), resample=Image.BOX)
        scale = self.resolution / min(*img.size)
        img = img.resize(tuple(round(x * scale) for x in img.size), resample=Image.BICUBIC)

        arr = np.array(img.convert("RGB"))
        cy = (arr.shape[0] - self.resolution) // 2
        cx = (arr.shape[1] - self.resolution) // 2
        arr = arr[cy:cy+self.resolution, cx:cx+self.resolution]
        arr = arr.astype(np.float32) / 127.5 - 1       # [0,255] → [-1,1]
        arr = np.transpose(arr, [2, 0, 1])             # CHW

        cond_vec = self.local_cond[idx]                # [12]
        return torch.from_numpy(arr), {"cond": torch.from_numpy(cond_vec)}
    

class ConditionalNpyDataset(Dataset):
    def __init__(self, resolution, image_paths, cond_array, shard=0, num_shards=1, spatial_basis=None, hp_weights=None, mlt_grid=None, oval_eb=None, oval_pb=None, prev_cfg=None, prev_set=None): # mj_20250911, mj_20260205 added hp_weights, mj_20260225 added mlt_grid, mj_20260302 added oval_eb/oval_pb, mj_20260927 added prev_cfg/prev_set
        super().__init__()
        self.prev_cfg = prev_cfg; self.prev_set = prev_set  # mj_20260927
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]  # e.g. *.npy paths
        self.local_cond   = cond_array[shard:][::num_shards]
        self.spatial_basis = spatial_basis  # e.g., sinMLT, cosMLT, normMLAT # mj_20250911
        # HP-based sample weights (mj_20260205)
        self.local_hp_weights = hp_weights[shard:][::num_shards] if hp_weights is not None else None
        self.mlt_grid = mlt_grid  # [W] MLT hours (0-24) for each image column; used for aligned SML/SMU injection  # mj_20260225
        # Oval boundaries (mj_20260302)
        self.local_oval_eb = oval_eb[shard:][::num_shards] if oval_eb is not None else None  # mj_20260302
        self.local_oval_pb = oval_pb[shard:][::num_shards] if oval_pb is not None else None  # mj_20260302


    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        # read the [3, H, W] array
        path = self.local_images[idx]


        # arr = np.load(path).astype(np.float32)  # shape: [3, H, W]
        # arr = np.repeat(np.load(path).astype(np.float32)[2,:,:][np.newaxis,:,:],3,axis=0)  # shape: [3, H, W]
        arr = np.load(path).astype(np.float32)[1, :, :][np.newaxis, ...]  # shape: [1, H, W]
        if False: # mj_20250804
            print(f"[DEBUG][RANK 0] Loaded file: {path},\
                  arr shape: {arr.shape}, \
                  1 min/max: {arr[0].min():.4f}/{arr[0].max():.4f},\
                  2 min/max: {arr[1].min():.4f}/{arr[1].max():.4f},\
                  3 min/max: {arr[2].min():.4f}/{arr[2].max():.4f}") # mj_20250804


        cond_vec = self.local_cond[idx]  # shape: [D]

        # spatial basis (optional)
        cond_dict = {"cond": torch.from_numpy(cond_vec)}
        if self.spatial_basis is not None:
            cs = self.spatial_basis.astype(np.float32)              # [C_spatial, H, W]
            # mj_20260225: MLT-aligned SML/SMU injection -----------------------------------
            # Interpolate the 24-sector SML and SMU values onto each image column's MLT
            # coordinate, then append as 2 extra spatial channels.
            # Sector k is centred at MLT = k hours (k = 0..23). np.interp with period=24
            # handles the circular boundary (sector 23 wraps back to sector 0).
            if self.mlt_grid is not None:
                sml = cond_vec[:24]                                          # [24] z-scored SML per sector
                smu = cond_vec[24:48]                                        # [24] z-scored SMU per sector
                sec = np.arange(24, dtype=np.float32)                       # sector centres: 0,1,...,23 h
                sml_col = np.interp(self.mlt_grid, sec, sml, period=24.0).astype(np.float32)  # [W]
                smu_col = np.interp(self.mlt_grid, sec, smu, period=24.0).astype(np.float32)  # [W]
                H_g, W_g = cs.shape[1], cs.shape[2]
                sml_ch = np.broadcast_to(sml_col[None, :], (H_g, W_g)).copy()   # [H, W]
                smu_ch = np.broadcast_to(smu_col[None, :], (H_g, W_g)).copy()   # [H, W]
                extra = np.stack([sml_ch, smu_ch], axis=0)                   # [2, H, W]
                cs = np.concatenate([cs, extra], axis=0)                     # [C_spatial+2, H, W]
            # -------------------------------------------------------------------------------
            if self.prev_cfg:  # mj_20260927 autoregressive conditioning: append [previous frame, availability mask] as 2 spatial channels
                cs = np.concatenate([cs, self._prev_frame(path)], axis=0)  # [C_spatial+2, H, W]  # mj_20260927
            cond_dict["cond_spatial"] = torch.from_numpy(cs)
        
        # HP-based sample weight (mj_20260205)
        if self.local_hp_weights is not None:
            cond_dict["hp_weight"] = torch.tensor(self.local_hp_weights[idx], dtype=torch.float32)

        # Oval boundaries (mj_20260302)
        if self.local_oval_eb is not None:  # mj_20260302
            cond_dict["oval_eb"] = torch.from_numpy(self.local_oval_eb[idx].astype(np.int16))  # [W] int16 (255=sentinel)  # mj_20260302
            cond_dict["oval_pb"] = torch.from_numpy(self.local_oval_pb[idx].astype(np.int16))  # [W] int16  # mj_20260302

        return torch.from_numpy(arr), cond_dict # mj_20250911

    def _prev_frame(self, path):  # mj_20260927
        """[2, H, W]: channel 0 = mean-energy-flux frame delta in {1..delta_max} min earlier (same train80 log10 normalisation as the
        target) + Gaussian conditioning noise sigma ~ U(0, noise) (Ho et al. cascaded-diffusion augmentation, limits error compounding in
        the free-running chain); channel 1 = +1 if a previous frame is given, -1 if missing or dropped (prob drop -> the model also learns
        the unconditional-in-history mode used for the first frame of a chain)."""
        cfg = self.prev_cfg; H, W = self.spatial_basis.shape[1], self.spatial_basis.shape[2]  # mj_20260927
        name = os.path.basename(path); day, m = name[:8], int(name[9:-4])  # 'YYYYMMDD_<minute of day>.npy'  # mj_20260927
        delta = int(np.random.randint(1, cfg['delta_max'] + 1)); pm = m - delta  # mj_20260927
        if pm < 0:  # mj_20260927
            from datetime import datetime, timedelta  # mj_20260927
            day = (datetime.strptime(day, '%Y%m%d') - timedelta(days=1)).strftime('%Y%m%d'); pm += 1440  # mj_20260927
        pname = f'{day}_{pm}.npy'  # mj_20260927
        if pname in self.prev_set and np.random.random() >= cfg['drop']:  # mj_20260927
            prev = np.load(os.path.join(cfg['dir'], pname)).astype(np.float32)[1]  # mj_20260927
            sig = np.random.uniform(0.0, cfg['noise'])  # mj_20260927
            if sig > 0: prev = np.clip(prev + sig * np.random.randn(H, W).astype(np.float32), -1.0, 1.0)  # mj_20260927
            return np.stack([prev, np.ones((H, W), np.float32)], axis=0)  # mj_20260927
        return np.stack([np.zeros((H, W), np.float32), -np.ones((H, W), np.float32)], axis=0)  # mj_20260927
