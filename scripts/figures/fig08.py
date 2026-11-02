"""Figure 8: snapshots of the 6 April 2000 storm. (a,b) potential: DHEM-EP, AMIE, Weimer 2005; (c,d) energy flux: DHEM-AP, AMIE, OVATION Prime."""
import numpy as np
import matplotlib.pyplot as plt
from _common import load, save, polar_grid, wrap, hp_polar_gw, SERIF, CLEAN, LAT_TICKS, R_TICKS

plt.rcParams.update({'figure.dpi': 300, 'font.size': 13, **SERIF, 'mathtext.fontset': 'dejavuserif', **CLEAN})


def snapshots(maps, titles, col_labels, panel_labels, levels, cmap, extend, cbar_ticks, cbar_label, label_color, face, name, THETA, R):
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, hspace=0.68, wspace=0.25, left=0.05, right=0.88, top=0.86, bottom=0.06)
    for i in range(2):
        for j in range(3):
            ax = fig.add_subplot(gs[i, j], projection='polar')
            im = ax.contourf(THETA, R, wrap(maps[i][j]), levels=levels, cmap=cmap, extend=extend)
            ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
            ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0", "6", "12", "18"], fontsize=18)
            ax.set_yticks(R_TICKS); ax.set_yticklabels([f"{lt:d}°" for lt in LAT_TICKS], fontsize=16, color=label_color)
            ax.set_ylim(0, 40); ax.set_axisbelow(False)
            ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.7, color=label_color, zorder=10)
            ax.set_facecolor(face)
            ax.set_title(titles[i][j], fontsize=19, fontweight='normal', pad=10 if i == 0 else 15)
            if i == 0:
                ax.text(0.5, 1.5, col_labels[j], transform=ax.transAxes, fontsize=21, ha='center', va='bottom')
            if j == 0:
                ax.text(-0.1, 1.05, panel_labels[i], transform=ax.transAxes, fontsize=19, fontweight='bold', va='top', ha='right', color='black')
    cbar = fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.02, 0.70]))
    cbar.ax.tick_params(labelsize=20)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.set_ylabel(cbar_label, fontsize=22)
    cbar.outline.set_visible(False)
    fig.patch.set_edgecolor('none'); fig.patch.set_linewidth(0)
    for ax in fig.axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
    save(fig, name, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')


# (a, b) electric potential
p = load('fig08ab_potential_snapshots.npz')
THETA, R = polar_grid(p['mlat'], p['mlt'], wrap=1)
maps = [[p['dhem_ep'][i], p['amie'][i], p['weimer'][i]] for i in range(2)]
titles = [[f"{str(p['time'][i])} UT\n" + fr"$\Phi_{{PC}}$ = {m.max() - m.min():.1f} kV" for m in maps[i]] for i in range(2)]
for i in range(2):
    print(p['time'][i], 'CPCP [kV]:', [f'{m.max() - m.min():.1f}' for m in maps[i]])
snapshots(maps, titles, ['DHEM-EP', 'AMIE', 'Weimer (2005)'], ['(a)', '(b)'], np.linspace(-80, 80, 41), 'coolwarm', 'both',
          np.linspace(-80, 80, 5), 'Electric potential [kV]', 'black', 'white', 'Fig08ab_pot_snapshots', THETA, R)

# (c, d) particle energy flux
f = load('fig08cd_flux_snapshots.npz')
THETA, R = polar_grid(f['mlat'], f['mlt'], wrap=1)
maps = [[f['dhem_ap'][i], f['amie'][i], f['ovation_prime'][i]] for i in range(2)]
hp = [[hp_polar_gw(m, f['mlat'], f['mlt']) for m in maps[i]] for i in range(2)]
titles = [[f"{str(f['time'][i])} UT\nHP = {hp[i][j]:.1f} GW" for j in range(3)] for i in range(2)]
for i in range(2):
    print(f['time'][i], 'HP [GW]:', [f'{h:.1f}' for h in hp[i]])
snapshots(maps, titles, ['DHEM-AP', 'AMIE', 'OVATION Prime'], ['(c)', '(d)'], np.linspace(0, 12, 25), 'hot', 'both',
          np.linspace(0, 12, 5), 'Particle Energy Flux (mW/m²)', 'white', 'black', 'Fig08cd_flux_snapshots', THETA, R)
