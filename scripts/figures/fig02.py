"""Figure 2: Bland-Altman plots of CPCP (DHEM-EP) and HP (DHEM-AP) against AMIE, 2003 test dataset, 10-min cadence."""
import numpy as np
import matplotlib.pyplot as plt
from _common import load, save, SERIF, CLEAN

d = load('fig02_bland_altman.npz')

# (a) cross polar cap potential
plt.rcParams.update({'figure.dpi': 150, 'font.size': 13, 'axes.spines.top': True, 'axes.spines.right': True, **SERIF, **CLEAN})
a, m = d['cpcp_amie'], d['cpcp_dhem_ep']
ba_mean, ba_diff = (a + m) / 2, m - a
bias, sd = ba_diff.mean(), ba_diff.std()
lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
r = np.corrcoef(a, m)[0, 1]
print(f'CPCP: N={len(a)}, r={r:.3f}, bias={bias:+.2f} kV, LoA=[{lo:.2f}, {hi:.2f}], MAE={np.abs(ba_diff).mean():.2f}')
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(ba_mean, ba_diff, s=2, alpha=0.15, color='#555555', rasterized=True)
ax.axhline(bias, color='#d62728', lw=2.0, ls='-')
ax.axhline(hi, color='#1f77b4', lw=1.5, ls='--')
ax.axhline(lo, color='#1f77b4', lw=1.5, ls='--')
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
ax.set_xlim(-2.5, 220); ax.set_ylim(-200, 200)
xlim = ax.get_xlim()
ax.text(xlim[1] * 0.98, hi + 5.0, f'+1.96 SD = {hi:.2f} kV', ha='right', va='bottom', fontsize=13, color='#1f77b4')
ax.text(xlim[1] * 0.98, lo - 5.0, f'−1.96 SD = {lo:.2f} kV', ha='right', va='top', fontsize=13, color='#1f77b4')
ax.text(xlim[1] * 0.98, lo - 60.0, f'Mean diff = {bias:.2f} kV', ha='right', va='top', fontsize=14, color='#d62728')
ax.set_xlabel('Mean CPCP (AMIE + DHEM-EP) / 2 [kV]', fontsize=18)
ax.set_ylabel('Diff. (DHEM-EP − AMIE) [kV]', fontsize=18)
ax.set_title(f'Cross polar cap potential (r = {r:.2f})', fontsize=20)
ax.tick_params(labelsize=16, width=1.2, length=5)
save(fig, 'Fig02a_CPCP_BlandAltman', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')

# (b) hemispheric power
plt.rcParams.update({'figure.dpi': 150, 'font.size': 14, **SERIF, **CLEAN})
XLIM, YLIM = (-2.484866020845498, 180.0), (-130.0, 130.0)
a, m = d['hp_amie'], d['hp_dhem_ap']
mean_hp, diff_hp = (a + m) / 2, m - a
bias, sd = diff_hp.mean(), diff_hp.std()
lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
r = np.corrcoef(a, m)[0, 1]
print(f'HP:   N={len(a)}, r={r:.3f}, bias={bias:+.2f} GW, LoA=[{lo:.2f}, {hi:.2f}], MAE={np.abs(diff_hp).mean():.2f}')
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.scatter(mean_hp, diff_hp, s=2, alpha=0.15, color='#555555', rasterized=True)
ax.axhline(bias, color='#d62728', lw=2.0, ls='-', zorder=4)
ax.axhline(lo, color='#1f77b4', lw=1.5, ls='--', zorder=4)
ax.axhline(hi, color='#1f77b4', lw=1.5, ls='--', zorder=4)
ax.axhline(0, color='k', lw=0.5, alpha=0.3)
x_line = np.linspace(0, XLIM[1] * 1.05, 200)
for pct, ls in [(0.30, '--'), (0.50, ':')]:
    ax.plot(x_line, pct * x_line, ls=ls, color='#2ca02c', lw=1.2, alpha=0.7, zorder=3)
    ax.plot(x_line, -pct * x_line, ls=ls, color='#2ca02c', lw=1.2, alpha=0.7, zorder=3)
ax.text(XLIM[1] * 0.97, 0.30 * XLIM[1] * 0.97 + 2.0, '±30%', fontsize=13, color='#2ca02c', rotation=12, va='bottom', ha='right')
ax.text(XLIM[1] * 0.70, 0.50 * XLIM[1] * 0.70 + 2.0, '±50%', fontsize=13, color='#2ca02c', rotation=18, va='bottom')
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
bb = dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.5)
ax.text(XLIM[1] * 0.98, bias + 1.5, f'Mean diff = {bias:.2f} GW', ha='right', va='bottom', fontsize=14, color='#d62728', zorder=5, bbox=bb)
ax.text(XLIM[1] * 0.98, hi + 1.5, f'+1.96 SD = {hi:.2f} GW', ha='right', va='bottom', fontsize=13, color='#1f77b4', zorder=5, bbox=bb)
ax.text(XLIM[1] * 0.98, lo - 1.5, f'−1.96 SD = {lo:.2f} GW', ha='right', va='top', fontsize=13, color='#1f77b4', zorder=5, bbox=bb)
ax.set_xlabel('Mean HP (AMIE + DHEM-AP) / 2 [GW]', fontsize=18)
ax.set_ylabel('Diff. (DHEM-AP − AMIE) [GW]', fontsize=18)
ax.set_title(f'Hemispheric power (r = {r:.2f})', fontsize=20)
ax.tick_params(labelsize=16, width=1.2, length=5)
save(fig, 'Fig02b_HP_BlandAltman', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
