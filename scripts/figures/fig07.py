"""Figure 7: IMF, SMU/SML and hemispheric power (DHEM-AP, AMIE, OVATION Prime, HP(AE)) during the 6-7 April 2000 storm."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnchoredOffsetbox, DrawingArea
from matplotlib.lines import Line2D
from datetime import datetime
from _common import load, save, pearson, SERIF

plt.rcParams.update({'figure.dpi': 100, 'font.size': 20, 'axes.spines.top': True, 'axes.spines.right': True})
plt.style.use('seaborn-v0_8')
plt.rcParams.update({**SERIF, 'mathtext.fontset': 'stix'})
d = load('fig07_hp_event.npz')
T = lambda key: np.array([datetime.strptime(str(t), '%Y-%m-%d %H:%M') for t in d[key]])
omni_times, times_arr, ovation_times = T('omni_time'), T('time'), T('ovation_time')
ddpm_hp, amie_hp, ovation_hp = d['hp_dhem_ap'], d['hp_amie'], d['ovation_hp']
hp_ae = 4.6 * np.sqrt(np.abs(d['ae'])) - 23          # HP(AE) [GW], Ostgaard et al. (2002)
start_dt, end_dt = datetime(2000, 4, 6, 3, 0, 0), datetime(2000, 4, 8, 0, 0, 0)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.2]})
plt.subplots_adjust(hspace=0.05)

ax = axes[0]
ax.plot(omni_times, d['omni_bz'], '-', color='#e76f51', label='Bz', linewidth=1.5)
ax.plot(omni_times, d['omni_by'], '-', color='#005f73', label='By', linewidth=1.5)
ax.set_ylabel(r'$B_{\mathrm{IMF}}$ [nT]', fontsize=21)
ax.legend(loc='lower right', fontsize=17, ncol=2, frameon=True, facecolor='white', framealpha=0.6, edgecolor='0.85')
ax.grid(True)
ax.text(0.02, 0.85, '(a)', transform=ax.transAxes, fontsize=21, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

ax = axes[1]
ax.plot(times_arr, d['smu'], '-', color='#C44E52', label='SMU', linewidth=1.5)
ax.plot(times_arr, d['sml'], '-', color='#4C72B0', label='SML', linewidth=1.5)
ax.set_ylabel('SMU / SML [nT]', fontsize=21)
ax.legend(loc='lower right', fontsize=17, ncol=2, frameon=True, facecolor='white', framealpha=0.6, edgecolor='0.85')
ax.grid(True)
ax.text(0.02, 0.85, '(b)', transform=ax.transAxes, fontsize=21, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

ax = axes[2]
ax.plot(times_arr, ddpm_hp, '-', color='#1f77b4', linewidth=1.8)
ax.plot(times_arr, amie_hp, '-', color='#ff7f0e', linewidth=1.5, alpha=0.8)
ax.plot(times_arr, hp_ae, '-', color='#d62728', linewidth=1.5, alpha=0.8, label='HP(AE)')
ax.plot(ovation_times, ovation_hp, '-', color='#2ca02c', linewidth=1.5, alpha=0.8)
ax.set_ylim(0, 300)
ax.set_ylabel('HP [GW]', fontsize=21)
ax.grid(True)
ax.text(0.02, 0.90, '(c)', transform=ax.transAxes, fontsize=21, fontweight='bold')

# correlations over the plotted interval
win = (times_arr >= start_dt) & (times_arr <= end_dt)
rd, ra, re = [np.where(win, a, np.nan) for a in (ddpm_hp, amie_hp, hp_ae)]
r_dhem_ae, n_dhem = pearson(rd, re)
r_amie_ae, n_amie = pearson(ra, re)
src_nums = mdates.date2num(times_arr)
ae_at_ovation = np.full(len(ovation_times), np.nan)
for i, q in enumerate(ovation_times):
    j = np.argmin(np.abs(src_nums - mdates.date2num(q)))
    if abs(src_nums[j] - mdates.date2num(q)) * 24 * 60 <= 5:
        ae_at_ovation[i] = hp_ae[j]
r_ovation_ae, n_ovation = pearson(ovation_hp, ae_at_ovation)
r_val_amie, n_val = pearson(rd, ra)
print(f'r(DHEM-AP, HP(AE))={r_dhem_ae:.3f} (n={n_dhem}), r(AMIE, HP(AE))={r_amie_ae:.3f} (n={n_amie}), '
      f'r(OVATION Prime, HP(AE))={r_ovation_ae:.3f} (n={n_ovation}); r(DHEM-AP, AMIE)={r_val_amie:.3f} (n={n_val})')


def legend_row(name, line_color, r, fontsize=15):
    swatch = DrawingArea(1.6 * fontsize, 0.7 * fontsize, 0, 0)
    swatch.add_artist(Line2D([0, 1.6 * fontsize], [0.35 * fontsize, 0.35 * fontsize], color=line_color, linewidth=2))
    parts = [swatch, TextArea(f' {name} ', textprops=dict(color='black', fontsize=fontsize))]
    if r is not None:
        parts.append(TextArea(f'(r={r:.2f})', textprops=dict(color='#d62728', fontsize=fontsize)))
    return HPacker(children=parts, align='center', pad=0, sep=2)


legend_box = VPacker(children=[legend_row('DHEM-AP', '#1f77b4', r_dhem_ae), legend_row('AMIE', '#ff7f0e', r_amie_ae),
                               legend_row('OVATION Prime', '#2ca02c', r_ovation_ae), legend_row('HP(AE)', '#d62728', None)], align='left', pad=4, sep=7)
anchored = AnchoredOffsetbox(loc='upper right', child=legend_box, frameon=True, pad=0.5, borderpad=0.6)
anchored.patch.set_facecolor('white'); anchored.patch.set_alpha(0.6); anchored.patch.set_edgecolor('0.85')
ax.add_artist(anchored)
ax.text(0.02, 0.83, f'r(DHEM-AP, AMIE) = {r_val_amie:.2f}', transform=ax.transAxes, fontsize=17, fontweight='normal', color='black', ha='left', va='top', zorder=10)

for a in axes:
    a.tick_params(axis='both', which='major', direction='in', top=True, right=True, width=1, length=5, labelsize=19)
    if a != axes[-1]:
        plt.setp(a.get_xticklabels(), visible=False)
axes[-1].set_xlabel('Universal Time (Hours)', fontsize=21)


def time_ticks(x, pos):
    t = mdates.num2date(x)
    return t.strftime('%m/%d') if t.hour == 0 and t.minute == 0 else t.strftime('%H')


axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(time_ticks))
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=6))
for a in axes:
    a.set_xlim(start_dt, end_dt)
    for vt in (datetime(2000, 4, 6, 16, 0), datetime(2000, 4, 6, 16, 47)):
        a.axvline(vt, color='black', linestyle='--', linewidth=1.2, zorder=5)
plt.subplots_adjust(left=0.135, right=0.90, top=0.97, bottom=0.08, hspace=0.05)
save(fig, 'Fig07_HP_Event01', dpi=300)
