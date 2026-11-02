"""Figure 3: mean DHEM and AMIE maps and their MAE. (a,b) electric potential, (c,d) particle energy flux, (e,f) average energy."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from _common import load, save, polar_grid, wrap, nice_ticks, SERIF, CLEAN, LAT_TICKS, R_TICKS

# ---------------------------------------------------------------- (a, b) potential: eps_t bins 1 and 5, clock angle 180 deg
plt.rcParams.update({'figure.dpi': 150, 'font.size': 13, 'axes.spines.top': True, 'axes.spines.right': True,
                     **SERIF, 'mathtext.fontset': 'stix', **CLEAN})
p = load('potential_bins.npz')
THETA, R = polar_grid(p['mlat'], p['mlt'], wrap=1)
edges = p['eps_bin_edges']


def polar_panel(ax, pattern, vmin, vmax, title, cmap):
    pattern_plot = np.ma.masked_invalid(wrap(pattern))
    im = ax.contourf(THETA, R, pattern_plot, levels=np.linspace(vmin, vmax, 41), cmap=cmap, extend='max')
    ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0", "6", "12", "18"], fontsize=20)
    ax.set_ylim(0, 40); ax.set_yticks(R_TICKS); ax.set_yticklabels([f"{lt:d}°" for lt in LAT_TICKS], fontsize=18)
    ax.set_axisbelow(False)
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.7, color='black', zorder=10)
    ax.set_facecolor('white')
    ax.set_title(title, fontsize=21, pad=15)
    return im


def three_panels(panels, suptitle, name, cell_mask):
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.03, right=0.93, top=0.70, bottom=0.05)
    for ci, (pattern, vmin, vmax, title, cmap, unit) in enumerate(panels):
        ax = fig.add_subplot(gs[0, ci], projection='polar')
        im = polar_panel(ax, pattern, vmin, vmax, title, cmap)
        cbar = fig.colorbar(im, ax=ax, pad=0.13, shrink=0.75)
        cbar.set_ticks(nice_ticks(vmin, vmax))
        cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cbar.ax.tick_params(labelsize=18)
        cbar.ax.set_ylabel(unit, fontsize=20)
        ax.contour(THETA, R, wrap(cell_mask), levels=[0.5], colors='black', linewidths=2.0, linestyles='-', zorder=15)
    fig.suptitle(suptitle, fontsize=22, y=0.99)
    save(fig, name, dpi=300, bbox_inches='tight', facecolor='white')


for panel, lab, ie in (('a', 'eps1_clk4', 1), ('b', 'eps5_clk4', 5)):
    k = int(np.where(p['bin_label'] == lab)[0][0])
    mae = p['mae'][k]
    print(f'Fig 3{panel}: N={p["n"][k]}, MAE max {mae.max():.2f} kV')
    three_panels([(p['mean_dhem_ep'][k], -35, 35, 'Mean DHEM-EP', 'coolwarm', 'kV'),
                  (p['mean_amie'][k], -35, 35, 'Mean AMIE', 'coolwarm', 'kV'),
                  (mae, 0, 25, f'MAE\nmax = {mae.max():.2f} kV', 'hot_r', 'kV')],
                 fr'$\epsilon_t$: [{edges[ie]:.0f}, {edges[ie + 1]:.0f}), $\theta_c$ = 180° ± 22.5°',
                 f'Fig03{panel}_pot_MAE_eps{edges[ie]:.0f}_{edges[ie + 1]:.0f}', p['cell_mask'][k])

# ---------------------------------------------------------------- (c-f) energy flux and mean energy: SME bins 300-600 and 1000-1500 nT
plt.rcParams.update({'figure.dpi': 150, 'font.size': 14, 'mathtext.fontset': 'dejavusans', **SERIF, **CLEAN})


def setup_polar_ax(ax):
    ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0", "6", "12", "18"], fontsize=16)
    ax.set_yticks(R_TICKS); ax.set_yticklabels([f"{lt:d}°" for lt in LAT_TICKS], fontsize=14)
    ax.set_ylim(0, 40); ax.set_axisbelow(False)
    ax.grid(True, ls='--', lw=1.0, alpha=0.7, color='black', zorder=10)
    ax.set_facecolor('white')


def render_3col(cols, suptitle, name, THETA, R, eb_pb=None):
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, wspace=0.45, left=0.03, right=0.93, top=0.70, bottom=0.05)
    for ci, (d, lvl, cm, title, unit, subtitle, dark) in enumerate(cols):
        ax = fig.add_subplot(gs[0, ci], projection='polar')
        setup_polar_ax(ax)
        ax.set_xticklabels(["0", "6", "12", "18"], fontsize=20)
        ax.tick_params(axis='x', pad=8)
        grid_color = 'white' if dark else 'black'
        ax.grid(True, ls='--', lw=1.0, alpha=0.7, color=grid_color, zorder=10)
        ax.set_yticklabels([f"{lt:d}°" for lt in LAT_TICKS], fontsize=18, color=grid_color)
        im = ax.contourf(THETA, R, wrap(np.nan_to_num(d, nan=0.0), 2), levels=lvl, cmap=cm, extend='max')
        if eb_pb is not None:
            theta_bnd, eb_r, pb_r = eb_pb
            ax.plot(theta_bnd[~np.isnan(eb_r)], eb_r[~np.isnan(eb_r)], color='limegreen', linewidth=2.0, linestyle='-', zorder=15)
            ax.plot(theta_bnd[~np.isnan(pb_r)], pb_r[~np.isnan(pb_r)], color='limegreen', linewidth=2.0, linestyle='--', zorder=15)
        ax.set_title(f'{title}\n{subtitle}' if subtitle else title, fontsize=21, pad=15)
        cb = fig.colorbar(im, ax=ax, pad=0.13, shrink=0.75)
        cb.set_ticks(nice_ticks(lvl[0], lvl[-1]))
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        cb.ax.set_ylabel(unit, fontsize=20)
        cb.ax.tick_params(labelsize=18)
    fig.suptitle(suptitle, fontsize=22, y=0.99)
    save(fig, name, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')


# (c, d) energy flux
f = load('flux_bins.npz')
THETA, R = polar_grid(f['mlat'], f['mlt'], wrap=2)
theta_bnd = np.append(f['mlt'], [24.0, 24.75]) * (2 * np.pi / 24)
gfv = max(np.percentile(m, 99) for m in f['mae']) * 1.1          # MAE colour scale shared by the six SME bins
vmax10 = int(np.ceil(min(gfv, 15) * 10))
while vmax10 % 4 and vmax10 % 5:
    vmax10 += 1
levels_mae = np.linspace(0, vmax10 / 10, 41)
levels_flux = np.linspace(0, 12, 41)
for panel, slug in (('c', 'sme_300_600'), ('d', 'sme_1000_1500')):
    k = int(np.where(f['slug'] == slug)[0][0])
    mae, amie, n = f['mae'][k], f['mean_amie'][k], int(f['n'][k])
    oval = amie >= 0.5
    print(f'Fig 3{panel}: N={n}, oval mean MAE {mae[oval].mean():.2f}, max {mae.max():.2f} mW/m2')
    eb_pb = (theta_bnd, np.append(f['oval_eb_colat'][k], f['oval_eb_colat'][k][:2]), np.append(f['oval_pb_colat'][k], f['oval_pb_colat'][k][:2]))
    render_3col([(f['mean_dhem_ap'][k], levels_flux, 'hot', 'Mean DHEM-AP', 'mW/m²', None, True),
                 (amie, levels_flux, 'hot', 'Mean AMIE', 'mW/m²', None, True),
                 (mae, levels_mae, 'hot_r', 'MAE', 'mW/m²', f'max = {mae.max():.2f} mW/m²', False)],
                f'SME: [{slug[4:].replace("_", ", ")})',
                f'Fig03{panel}_flux_MAE_{slug.replace("sme_", "sme")}', THETA, R, eb_pb)

# (e, f) mean energy
e = load('energy_bins.npz')
THETA, R = polar_grid(e['mlat'], e['mlt'], wrap=2)
gmv = max(np.percentile(m, 99) for m in e['mae']) * 1.1
levels_mae = np.linspace(0, np.ceil(min(gmv, 3) * 10) / 10, 41)
for panel, slug in (('e', 'sme_300_600'), ('f', 'sme_1000_1500')):
    k = int(np.where(e['slug'] == slug)[0][0])
    mae, amie, n = e['mae'][k], e['mean_amie'][k], int(e['n'][k])
    rmae = (mae / np.maximum(amie, 0.05) * 100.0)[amie >= 0.5]
    print(f'Fig 3{panel}: N={n}, oval mean rMAE {rmae.mean():.1f}%, MAE max {mae.max():.2f} keV')
    render_3col([(e['mean_dhem_ape'][k], np.linspace(0, 12, 41), 'hot', 'Mean DHEM-AP(E)', 'keV', None, True),
                 (amie, np.linspace(0, 12, 41), 'hot', 'Mean AMIE', 'keV', None, True),
                 (mae, levels_mae, 'hot_r', 'MAE', 'keV', f'max = {mae.max():.2f} keV', False)],
                f'SME: [{slug[4:].replace("_", ", ")})',
                f'Fig03{panel}_energy_MAE_{slug.replace("sme_", "sme")}', THETA, R)
