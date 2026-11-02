"""Geometric conditioning channels of the 32 x 32 AMIE grid: sin(2 pi MLT / 24), cos(2 pi MLT / 24), MLAT scaled to
[-1, 1] (90 deg -> +1, 44 deg -> -1). The AMIE grid (24 MLAT values 90..44 deg, 25 MLT values) is read from
assets/amie_grid.npz; MLAT is resampled to 32 rows with PCHIP in index space, MLT sampled at 32 pixels in [0, 24) h.
"""
import os
import numpy as np
from scipy.interpolate import PchipInterpolator

_ASSET = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "amie_grid.npz")


def amie_grid(path=_ASSET):
    g = np.load(path)
    return np.asarray(g["lats"], dtype=np.float32), np.asarray(g["mlts"], dtype=np.float32)


def build_spatial_basis(mlat_deg_orig, mlt_hours_orig, H=32, W=32, include_mlat=True, K=1):
    mlat = np.asarray(mlat_deg_orig, dtype=np.float32)
    if mlat[0] > mlat[-1]:
        mlat_inc, flip_back = mlat[::-1], True
    else:
        mlat_inc, flip_back = mlat, False
    f_lat = PchipInterpolator(np.arange(len(mlat_inc)), mlat_inc, axis=0)
    mlat_res = f_lat(np.linspace(0, len(mlat_inc) - 1, H))
    if flip_back:
        mlat_res = mlat_res[::-1]
    phi = 2.0 * np.pi * (np.linspace(0.0, 24.0, W, endpoint=False) / 24.0)
    chans = []
    for k in range(1, K + 1):
        chans.append(np.sin(k * phi)[None, None, :])
        chans.append(np.cos(k * phi)[None, None, :])
    basis = np.repeat(np.concatenate(chans, axis=0), H, axis=1).astype(np.float32)
    if include_mlat:
        lo, hi = float(mlat_res.min()), float(mlat_res.max())
        mlat_norm = np.clip(2.0 * (mlat_res - lo) / (hi - lo) - 1.0, -1.0, 1.0).astype(np.float32)
        basis = np.concatenate([basis, np.repeat(mlat_norm[:, None], W, axis=1)[None]], axis=0)
    return basis  # [C, H, W] float32


def mlat_rows(H=32, path=_ASSET):
    """MLAT (deg) of the 32 image rows, as used for the area weights of HP / HCE (linear in latitude)."""
    lats, _ = amie_grid(path)
    lat = np.linspace(lats.min(), lats.max(), H)
    return lat[::-1] if lats[0] > lats[-1] else lat


def default_basis(path=_ASSET):
    lats, mlts = amie_grid(path)
    return build_spatial_basis(lats, mlts, 32, 32, include_mlat=True, K=1)


def sml_smu_channels(cond48, W=32, H=32):
    """C_spatial=4 models only: SML / SMU of the 24 MLT sectors (centred at k h) interpolated to the 32 pixel MLTs."""
    centers = np.arange(24, dtype=np.float32)
    pix = np.linspace(0.0, 24.0, W, endpoint=False)
    out = np.zeros((cond48.shape[0], 2, H, W), dtype=np.float32)
    for b in range(cond48.shape[0]):
        out[b, 0] = np.interp(pix, centers, cond48[b, :24], period=24)[None, :]
        out[b, 1] = np.interp(pix, centers, cond48[b, 24:48], period=24)[None, :]
    return out
