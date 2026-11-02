"""Decoding of DHEM outputs and the integrated quantities of the paper.

Model outputs are min-max scaled with the training ranges (assets/amie_range_train80.json; log10 for the two precipitation
channels, linear for the potential). The AMIE test maps use the test ranges (assets/amie_range_test20.json, all channels
linear). Channel 0 = average energy [keV], 1 = particle energy flux [mW m^-2], 2 = potential [kV].
"""
import json
import numpy as np

R_EARTH = 6.371e6


def load_range(path, channel):
    r = json.load(open(path))
    return float(r["vmins"][channel]), float(r["vmaxs"][channel])


def decode_model(arr, kind, vmin, vmax, eps_subtract=0.0, floor=None):
    """DHEM output in [-1, 1] -> physical units (training range)."""
    x = (np.asarray(arr, dtype=np.float64) + 1.0) / 2.0 * (vmax - vmin) + vmin
    if kind == "log10":
        x = 10.0 ** x - eps_subtract
        x = np.maximum(x, 0.0) if floor is None else np.maximum(x, floor)
    elif kind != "linear":
        raise ValueError(kind)
    return x


def decode_amie(npy, channel, test_range_path, floor=None):
    """AMIE test map (linear min-max with the test range) -> physical units."""
    vmin, vmax = load_range(test_range_path, channel)
    x = (np.asarray(npy[channel], dtype=np.float64) + 1.0) / 2.0 * (vmax - vmin) + vmin
    return x if floor is None else np.maximum(x, floor)


def area_weights(lat_rows_deg, W=32):
    """Cell areas [m^2] of the 32 x 32 grid: sin(colat) * dlat * dMLT * R^2, rows = MLAT, columns = MLT."""
    lat = np.asarray(lat_rows_deg, dtype=np.float64)
    dlat = abs(np.deg2rad(lat[1] - lat[0]))
    return (np.sin(np.deg2rad(90.0 - lat)) * dlat * (2.0 * np.pi / W) * R_EARTH ** 2)[:, None] * np.ones((1, W))


def hemispheric_power(flux_mWm2, aw):
    """HP [GW] = sum(F dA); F in mW m^-2, dA in m^2 -> 1e-3 W * ... -> GW factor 1e-12."""
    return float(np.sum(np.maximum(flux_mWm2, 0.0) * aw) * 1e-12)


def cpcp(potential_kV):
    """Cross polar cap potential [kV] = max - min of the map."""
    return float(np.nanmax(potential_kV) - np.nanmin(potential_kV))


def hme(flux_mWm2, energy_keV, aw):
    """Hemispheric mean energy [keV] = sum(F dA) / sum(F/E dA) = HP / hemispheric number flux (number-flux-weighted mean energy)."""
    den = np.sum(flux_mWm2 / energy_keV * aw)
    return float(np.sum(flux_mWm2 * aw) / den) if den > 0 else float("nan")


def bland_altman(a, d):
    """a = reference (AMIE), d = model. Returns dict with r, bias, LoA, MAE."""
    a, d = np.asarray(a, float), np.asarray(d, float)
    m = np.isfinite(a) & np.isfinite(d)
    a, d = a[m], d[m]
    diff = d - a
    sd = diff.std(ddof=1) if len(diff) > 1 else float("nan")
    r = float(np.corrcoef(a, d)[0, 1]) if len(a) > 2 else float("nan")
    return dict(N=int(len(a)), r=r, bias=float(diff.mean()), loa_lo=float(diff.mean() - 1.96 * sd),
                loa_hi=float(diff.mean() + 1.96 * sd), mae=float(np.abs(diff).mean()))
