"""Shared pieces of the figure scripts: data folder, 32 x 32 MLAT x MLT grid, polar-axis helpers, HP integral."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.environ.get('FIG_DATA', os.path.join(ROOT, 'figure_data'))   # the figure-data record, unzipped
OUT = os.environ.get('FIG_OUT', os.path.join(ROOT, 'figures'))
os.makedirs(OUT, exist_ok=True)

SERIF = {'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif']}
CLEAN = {'figure.edgecolor': 'none', 'axes.edgecolor': 'none', 'savefig.edgecolor': 'none',
         'savefig.facecolor': 'white', 'patch.edgecolor': 'none'}
LAT_TICKS = np.array([50, 60, 70, 80])
R_TICKS = 90.0 - LAT_TICKS


def load(name):
    return np.load(os.path.join(DATA, name), allow_pickle=True)


def save(fig, name, **kw):
    for ext in ('png', 'pdf'):
        fig.savefig(os.path.join(OUT, f'{name}.{ext}'), **kw)
    plt.close(fig)
    print('saved', name)


def polar_grid(mlat, mlt, wrap=1):
    """THETA, R meshes for contourf; wrap = number of MLT columns repeated after 24 h."""
    mlt_plot = np.append(mlt, [24.0 + 0.75 * k for k in range(wrap)])
    return np.meshgrid(mlt_plot * (2 * np.pi / 24), 90 - mlat)


def wrap(arr, n=1):
    return np.hstack([arr, arr[:, 0:n]])


def hp_polar_gw(flux_mWm2, mlat, mlt):
    """HP [GW] of the event figures (6-8): cell-centred 31 x 31 integral of the energy flux poleward of 50 deg MLAT."""
    lr = np.radians(mlat)
    dlon = np.radians((mlt[1] - mlt[0]) / 24 * 360)
    area = np.array([[6371e3 ** 2 * dlon * (np.sin(lr[i]) - np.sin(lr[i + 1]))] * 31 for i in range(31)])
    flag = ((lr[1:] > np.radians(50)) & (lr[1:] < np.radians(90))).astype(float)
    f = 0.25 * (flux_mWm2[:-1, :-1] + flux_mWm2[1:, :-1] + flux_mWm2[:-1, 1:] + flux_mWm2[1:, 1:])
    return np.sum(f * area * flag[:, None]) / 1e12


def pearson(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    return (np.corrcoef(x[m], y[m])[0, 1], int(m.sum())) if m.sum() > 2 else (np.nan, 0)


def nice_ticks(vmin, vmax):
    for n in (5, 4, 6, 3):
        t = np.linspace(vmin, vmax, n)
        if np.allclose(t * 10, np.round(t * 10), atol=1e-6):
            return t
    return np.linspace(vmin, vmax, 5)
