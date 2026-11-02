#!/usr/bin/env python
"""
Pre-compute auroral oval boundaries (EB/PB) for all training npy files
using 2D k-means clustering on (intensity, MLAT_index) features.

For each 32x32 image (channel [1] = mean energy flux, log10 scale):
  1. Build feature matrix F = [intensity, mlat_index/31] for all pixels
  2. k-means(k=8) to cluster pixels
  3. Classify clusters as "aurora" if cluster centroid intensity > T=-0.4
  4. Per MLT column j=0..31: EB[j]=max aurora row, PB[j]=min aurora row
     (row 0 = poleward/pole, row 31 = equatorward)  # mj_20260719
  5. Columns with no aurora pixels → EB=PB=255 (sentinel for "no boundary")

Output: oval_boundaries.npz
  - filenames: [N] array of basename strings
  - eb: [N, 32] uint8 — equatorward boundary index per MLT column
  - pb: [N, 32] uint8 — poleward boundary index per MLT column

Usage:
  python compute_oval_boundaries.py [--max_samples N]  (default: all)

mj_20260302
"""

import os
import sys
import argparse
import numpy as np
from sklearn.cluster import MiniBatchKMeans  # mj_20260302

# ---- config ----
DATA_DIR = "/home/lim2/data/amie_npy_joint_from105/training_npy_joint/"  # mj_20260302
K = 8                   # number of k-means clusters  # mj_20260302
INTENSITY_THRESH = -0.4 # log10 threshold for aurora vs background  # mj_20260302
SENTINEL = 255          # uint8 sentinel for "no boundary detected"  # mj_20260302
H, W = 32, 32           # image spatial dimensions  # mj_20260302
CHANNEL_IDX = 1         # mean energy flux channel  # mj_20260302


def compute_boundaries_single(arr_2d):  # mj_20260302
    """
    Compute EB and PB for a single [H, W] image (log10 intensity).
    Returns eb [W], pb [W] as uint8 arrays (SENTINEL=255 if no aurora in column).
    """
    H_img, W_img = arr_2d.shape  # mj_20260302

    # Build feature matrix: [H*W, 2] = (intensity, normalized_row_index)  # mj_20260302
    row_idx = np.arange(H_img, dtype=np.float32)  # mj_20260302
    row_grid = np.broadcast_to(row_idx[:, None], (H_img, W_img)).ravel()  # mj_20260302
    intensity = arr_2d.ravel().astype(np.float32)  # mj_20260302
    features = np.column_stack([intensity, row_grid / max(H_img - 1, 1)])  # mj_20260302

    # k-means clustering  # mj_20260302
    kmeans = MiniBatchKMeans(n_clusters=K, n_init=1, max_iter=30,  # mj_20260302
                             batch_size=256, random_state=42)  # mj_20260302
    labels = kmeans.fit_predict(features)  # mj_20260302
    centroids = kmeans.cluster_centers_  # [K, 2]  # mj_20260302

    # Classify clusters: aurora if centroid intensity > threshold  # mj_20260302
    aurora_mask_flat = np.isin(labels, np.where(centroids[:, 0] > INTENSITY_THRESH)[0])  # mj_20260302
    aurora_mask = aurora_mask_flat.reshape(H_img, W_img)  # mj_20260302

    eb = np.full(W_img, SENTINEL, dtype=np.uint8)  # mj_20260302
    pb = np.full(W_img, SENTINEL, dtype=np.uint8)  # mj_20260302

    for j in range(W_img):  # mj_20260302
        col_aurora = np.where(aurora_mask[:, j])[0]  # mj_20260302
        if len(col_aurora) > 0:  # mj_20260302
            eb[j] = col_aurora.max()   # equatorward boundary (largest row = most equatorward)  # mj_20260302
            pb[j] = col_aurora.min()   # poleward boundary (smallest row = most poleward)  # mj_20260302

    return eb, pb  # mj_20260302


def main():  # mj_20260302
    parser = argparse.ArgumentParser(description="Pre-compute oval boundaries for training data")  # mj_20260302
    parser.add_argument("--max_samples", type=int, default=0,  # mj_20260302
                        help="Process only first N samples (0=all, for testing)")  # mj_20260302
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,  # mj_20260302
                        help="Path to training npy directory")  # mj_20260302
    args = parser.parse_args()  # mj_20260302

    data_dir = args.data_dir  # mj_20260302
    npy_files = sorted([f for f in os.listdir(data_dir) if f.lower().endswith(".npy")])  # mj_20260302
    if args.max_samples > 0:  # mj_20260302
        npy_files = npy_files[:args.max_samples]  # mj_20260302
    N = len(npy_files)  # mj_20260302
    print(f"[oval_boundaries] Processing {N} files from {data_dir}")  # mj_20260302

    all_eb = np.zeros((N, W), dtype=np.uint8)  # mj_20260302
    all_pb = np.zeros((N, W), dtype=np.uint8)  # mj_20260302
    all_filenames = []  # mj_20260302

    for i, fname in enumerate(npy_files):  # mj_20260302
        fpath = os.path.join(data_dir, fname)  # mj_20260302
        arr = np.load(fpath).astype(np.float32)  # [3, H, W]  # mj_20260302
        img = arr[CHANNEL_IDX]  # [H, W] — mean energy flux (log10)  # mj_20260302

        eb, pb = compute_boundaries_single(img)  # mj_20260302
        all_eb[i] = eb  # mj_20260302
        all_pb[i] = pb  # mj_20260302
        all_filenames.append(fname)  # mj_20260302

        if (i + 1) % 10000 == 0 or (i + 1) == N:  # mj_20260302
            pct_aurora = np.mean(all_eb[:i+1] != SENTINEL) * 100  # mj_20260302
            print(f"  [{i+1}/{N}] {pct_aurora:.1f}% columns have aurora")  # mj_20260302

    out_path = os.path.join(data_dir, "oval_boundaries.npz")  # mj_20260302
    np.savez_compressed(out_path,  # mj_20260302
                        filenames=np.array(all_filenames),  # mj_20260302
                        eb=all_eb,  # [N, 32] uint8  # mj_20260302
                        pb=all_pb)  # [N, 32] uint8  # mj_20260302
    print(f"[oval_boundaries] Saved to {out_path}")  # mj_20260302
    print(f"  eb shape: {all_eb.shape}, pb shape: {all_pb.shape}")  # mj_20260302
    print(f"  Columns with aurora: {np.mean(all_eb != SENTINEL)*100:.1f}%")  # mj_20260302
    print(f"  File size: {os.path.getsize(out_path)/1e6:.1f} MB")  # mj_20260302


if __name__ == "__main__":  # mj_20260302
    main()  # mj_20260302
