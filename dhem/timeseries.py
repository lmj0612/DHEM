"""Time-series sampling (Section 3 of the paper; Figures 6-8).

noise_sequence     'iid': independent noise per minute. 'ar1': x(t) = rho x(t - dt) + sqrt(1 - rho^2) eps, rho = exp(-dt / tau),
                   the marginal of every x(t) is N(0, I). 'crn': one field for all minutes. Member k uses default_rng(seed + 1000 k + 17).
smooth_conditions  centred running mean of the condition vectors over runs of consecutive minutes.
crossfade_weights  raised-cosine blend of the two category models around a category switch.
"""
import numpy as np

DEFAULTS = dict(tau_min=120.0, cond_smooth_min=5, xfade_min=5)


def minutes_since(timestamps, origin=None):
    """'YYYY-mm-dd HH:MM[:SS]' strings -> (float minutes since origin [default: first timestamp], datetime64[m] array)."""
    ts = np.array([np.datetime64(str(s)[:16].replace(' ', 'T'), 'm') for s in np.asarray(timestamps).ravel()])
    if origin is None:
        origin = ts.min()
    return (ts - np.datetime64(origin, 'm')).astype('timedelta64[m]').astype(np.float64), ts


def noise_sequence(times_min, K, mode, tau=None, hw=(32, 32), seed=0):
    """x_T noise for N time-ordered minutes and K members: float32 [K, N, 1, H, W].

    times_min : [N] minutes since an arbitrary origin, ascending (gaps allowed; 'ar1' uses the real time step, so
                minutes far apart are effectively independent)
    mode      : 'iid' (independent per minute) | 'ar1' (autoregressive, e-folding time tau [min]) |
                'crn' (one field shared by all minutes)
    Member k uses numpy.random.default_rng(seed + 1000 k + 17).
    """
    times_min = np.asarray(times_min, dtype=np.float64)
    N = len(times_min); H, W = hw
    if N > 1:
        assert np.all(np.diff(times_min) > 0), 'times must be strictly increasing (sort the minutes first)'
    out = np.empty((K, N, 1, H, W), np.float32)
    for k in range(K):
        g = np.random.default_rng(int(seed) + 1000 * k + 17)
        if mode == 'iid':
            out[k] = g.standard_normal((N, 1, H, W), dtype=np.float32)
        elif mode == 'crn':
            out[k] = g.standard_normal((1, 1, H, W), dtype=np.float32)
        elif mode == 'ar1':
            assert tau is not None and tau > 0, "mode 'ar1' needs tau [min] > 0"
            z = g.standard_normal((1, H, W), dtype=np.float32); out[k, 0] = z
            for i in range(1, N):
                rho = float(np.exp(-(times_min[i] - times_min[i - 1]) / tau))
                z = rho * z + np.sqrt(max(1.0 - rho * rho, 0.0)) * g.standard_normal((1, H, W), dtype=np.float32)
                out[k, i] = z
        else:
            raise ValueError(f'unknown noise mode {mode!r}')
    return out


def smooth_conditions(conds, times_min, width):
    """Centred running mean of width `width` minutes (odd) of every column of conds [N, d] over runs of consecutive
    minutes (a gap starts a new run; the window is truncated at run edges). width <= 1 returns a copy."""
    conds = np.asarray(conds, dtype=np.float32)
    out = conds.copy()
    if width is None or int(width) <= 1 or len(conds) == 0:
        return out
    t = np.asarray(times_min, dtype=np.float64); h = int(width) // 2
    seg = np.concatenate([[0], np.cumsum(np.diff(t) != 1.0)])
    for s in np.unique(seg):
        ii = np.where(seg == s)[0]
        cs = np.cumsum(np.concatenate([np.zeros((1, conds.shape[1]), np.float64), conds[ii].astype(np.float64)]), axis=0)
        for j, i in enumerate(ii):
            lo, hi = max(0, j - h), min(len(ii), j + h + 1)
            out[i] = ((cs[hi] - cs[lo]) / (hi - lo)).astype(np.float32)
    return out


def crossfade_weights(cat, times_min, half_width, n_models=4):
    """Blend weights w [n_models + 1, N] (row m = category model m, row 0 unused): one-hot by category, and over the
    2*half_width minutes around every switch between two consecutive minutes a raised-cosine transition from the old
    to the new category model (no blending across data gaps). Columns sum to 1. half_width <= 0 -> one-hot."""
    cat = np.asarray(cat).astype(int); t = np.asarray(times_min, dtype=np.float64); N = len(cat)
    w = np.zeros((n_models + 1, N), np.float32); w[cat, np.arange(N)] = 1.0
    if half_width is None or int(half_width) <= 0 or N < 2:
        return w
    x = int(half_width)
    for b in np.where(np.diff(cat) != 0)[0] + 1:
        if t[b] - t[b - 1] != 1.0:
            continue
        cA, cB = int(cat[b - 1]), int(cat[b])
        for i in range(max(0, b - x), min(N, b + x)):
            f = (i - b + x + 0.5) / (2 * x); wB = 0.5 * (1.0 - np.cos(np.pi * f))
            w[:, i] = 0.0; w[cB, i] = wB; w[cA, i] = 1.0 - wB
    return w / np.maximum(w.sum(axis=0, keepdims=True), 1e-6)
