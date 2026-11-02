"""Figure 5: mean DHEM-AP particle energy flux (a) and average energy (b) for six SME bins."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from _common import load, save, polar_grid, wrap, SERIF, CLEAN, LAT_TICKS, R_TICKS

plt.rcParams.update({'figure.dpi': 150, 'font.size': 14, **SERIF, **CLEAN})


def two_by_three(maps, labels, vmax, cbar_label, name, mlat, mlt):
    THETA, R = polar_grid(mlat, mlt, wrap=2)
    fig = plt.figure(figsize=(12, 8.5))
    for idx, (d_map, label) in enumerate(zip(maps, labels)):
        ax = fig.add_subplot(2, 3, idx + 1, projection='polar')
        ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
        ax.set_yticks(R_TICKS)
        ax.set_ylim(0, 40); ax.set_axisbelow(False); ax.set_facecolor('white')
        ax.grid(True, ls='--', lw=1.0, alpha=0.7, color='white', zorder=10)
        ax.set_xticklabels(["0", "6", "12", "18"], fontsize=19)
        ax.set_yticklabels([f"{lt:d}°" for lt in LAT_TICKS], fontsize=17, color='white')
        ax.tick_params(axis='x', pad=8)
        ax.contourf(THETA, R, wrap(d_map, 2), levels=np.linspace(0, vmax, 41), cmap='hot', extend='max')
        ax.set_title(f'{label}', fontsize=19, pad=11)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=vmax), cmap='hot'), cax=fig.add_axes([0.91, 0.15, 0.015, 0.7]))
    cb.ax.set_ylabel(cbar_label, fontsize=22)
    cb.ax.tick_params(labelsize=20)
    plt.subplots_adjust(hspace=0.22, wspace=0.55, right=0.84)
    save(fig, name, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')


f = load('flux_bins.npz')
print('Fig 5a N per bin:', f['n'].tolist())
two_by_three(f['mean_dhem_ap'], f['bin_label'], 14, 'Particle Energy Flux [mW/m²]', 'Fig05_flux_climatology_2x3', f['mlat'], f['mlt'])
e = load('energy_bins.npz')
print('Fig 5b bin means [keV]:', np.round(e['mean_dhem_ape'].mean(axis=(1, 2)), 2).tolist())
two_by_three(e['mean_dhem_ape'], e['bin_label'], 8, 'Average Energy [keV]', 'Fig05b_energy_climatology_2x3', e['mlat'], e['mlt'])
