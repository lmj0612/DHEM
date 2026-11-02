"""Figure 4: mean DHEM-EP convection pattern per IMF clock-angle sector, eps_t bin [21000, 25500)."""
import numpy as np
import matplotlib.pyplot as plt
from _common import load, save, polar_grid, wrap, SERIF, CLEAN

plt.rcParams.update({'figure.dpi': 150, 'font.size': 13, 'axes.spines.top': True, 'axes.spines.right': True,
                     **SERIF, 'mathtext.fontset': 'dejavuserif', **CLEAN})
p = load('potential_bins.npz')
THETA, R = polar_grid(p['mlat'], p['mlt'], wrap=1)
vmin, vmax = -32, 32
positions = {0: (0, 1), 45: (0, 2), 90: (1, 2), 135: (2, 2), 180: (2, 1), 225: (2, 0), 270: (1, 0), 315: (0, 0)}

fig = plt.figure(figsize=(13, 13))
gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.35, left=0.04, right=0.86, top=0.82, bottom=0.04)
im = None
for k in range(len(p['bin_label'])):
    if p['eps_bin'][k] != 5:
        continue
    clk = int(p['clock_deg'][k])
    pattern = p['mean_dhem_ep'][k]
    cpcp = np.max(pattern) - np.min(pattern)
    print(f'{clk:3d} deg: N={p["n"][k]}, Phi_PC(mean map)={cpcp:.1f} kV, mean sample Phi_PC={p["cpcp_sample_mean_dhem_ep"][k]:.1f} kV')
    ax = fig.add_subplot(gs[positions[clk]], projection='polar')
    im = ax.contourf(THETA, R, wrap(pattern), levels=np.linspace(vmin, vmax, 41), cmap='coolwarm', extend='both')
    ax.set_theta_zero_location('S'); ax.set_theta_direction(1)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(["0", "6", "12", "18"], fontsize=20)
    ax.set_ylim(0, 40)
    lat_ticks = np.array([50, 60, 70, 80])
    ax.set_yticks(90.0 - lat_ticks); ax.set_yticklabels([f"{lt:d}°" for lt in lat_ticks], fontsize=18)
    ax.grid(True, linestyle='--', linewidth=1.0, alpha=0.7, color='black', zorder=5)
    ax.set_axisbelow(False); ax.set_facecolor('white')
    ax.set_title(f"{clk:.0f}°±22.5°\n" + fr'$\Phi_{{PC}}$={cpcp:.0f} kV ($\overline{{\Phi}}_{{PC}}$={p["cpcp_sample_mean_dhem_ep"][k]:.0f} kV)',
                 fontsize=20, pad=10)

# IMF clock-angle diagram in the centre
ax = fig.add_subplot(gs[1, 1])
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect('equal'); ax.axis('off')
for dx, dy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]:
    ax.plot([0, dx * 1.1], [0, dy * 1.1], color='grey', lw=1.5, ls='--', zorder=1)
for xy in [(0, 1.2), (1.2, 0), (0, -1.2), (-1.2, 0)]:
    ax.annotate('', xy=xy, xytext=(0, 0), arrowprops=dict(arrowstyle='->', lw=2, color='black'))
ax.text(0, 1.45, '+Z', ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(1.45, 0, '+Y', ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(0, -1.45, '−Z', ha='center', va='center', fontsize=20, fontweight='bold')
ax.text(-1.45, 0, '−Y', ha='center', va='center', fontsize=20, fontweight='bold')

cbar = fig.colorbar(im, cax=fig.add_axes([0.92, 0.15, 0.02, 0.7]))
cbar.ax.tick_params(labelsize=26)
cbar.set_ticks(np.linspace(vmin, vmax, 5))
cbar.ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FormatStrFormatter('%g'))
cbar.ax.set_ylabel('Electric potential [kV]', fontsize=28)
edges = p['eps_bin_edges']
fig.text(0.5, 0.995, fr'$\epsilon_t$: [{edges[5]:.0f}, {edges[6]:.0f})', ha='center', va='top', fontsize=30)
fig.text(0.5, 0.965, fr'$\langle B_T \rangle$={float(p["sw_bt_mean"]):.1f} nT, $\langle V_{{sw}} \rangle$={float(p["sw_vsw_mean"]):.0f} km/s, '
         fr'$\langle N_{{sw}} \rangle$={float(p["sw_nsw_mean"]):.1f} cm$^{{-3}}$', ha='center', va='top', fontsize=30)
save(fig, 'Fig04_pot_climatology_eps5', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
