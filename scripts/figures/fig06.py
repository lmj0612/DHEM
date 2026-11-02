"""Figure 6: IMF, SYM-H and cross polar cap potential (DHEM-EP, AMIE, Weimer 2005, DMSP F13) during the 6-7 April 2000 storm."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.offsetbox import TextArea, HPacker, VPacker, AnchoredOffsetbox, DrawingArea
from matplotlib.lines import Line2D
from datetime import datetime
from _common import load, save, pearson, SERIF

plt.style.use('seaborn-v0_8')
plt.rcParams.update({'font.size': 20, 'figure.dpi': 300, **SERIF, 'mathtext.fontset': 'stix'})
d = load('fig06_cpcp_event.npz')
T = lambda key: np.array([datetime.strptime(str(t), '%Y-%m-%d %H:%M') for t in d[key]])
omni_ts, evt_times, weimer_times = T('omni_time'), T('time'), T('weimer_time')
dmsp_times = np.array([datetime.fromisoformat(str(t)) for t in d['dmsp_time']])   # pass mid-times, with seconds
evt_ddpm, evt_amie = d['cpcp_dhem_ep'], d['cpcp_amie']
start_dt, end_dt = datetime(2000, 4, 6, 3, 0), datetime(2000, 4, 8, 0, 0)

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1.2]})
plt.subplots_adjust(hspace=0.05)

ax = axes[0]
ax.plot(omni_ts, d['omni_bz'], '-', color='#e76f51', label='Bz', linewidth=1.5)
ax.plot(omni_ts, d['omni_by'], '-', color='#005f73', label='By', linewidth=1.5)
ax.set_ylabel(r'$B_{\mathrm{IMF}}$ [nT]', fontsize=21)
ax.legend(loc='lower right', fontsize=17, ncol=2, frameon=True, facecolor='white', framealpha=0.6, edgecolor='0.85')
ax.grid(True)
ax.text(0.02, 0.85, '(a)', transform=ax.transAxes, fontsize=21, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

ax = axes[1]
ax.plot(omni_ts, d['omni_symh'], '-', color='#9467BD', label='SYM-H', linewidth=1.5)
ax.set_ylabel('SYM-H [nT]', fontsize=21)
ax.legend(loc='lower right', fontsize=17, ncol=1, frameon=True, facecolor='white', framealpha=0.6, edgecolor='0.85')
ax.grid(True)
ax.text(0.02, 0.85, '(b)', transform=ax.transAxes, fontsize=21, fontweight='bold')
ax.axhline(0, color='gray', linestyle='--', linewidth=1)

ax = axes[2]
ax.plot(evt_times, evt_ddpm, '-', color='#1f77b4', linewidth=1.8)
ax.plot(evt_times, evt_amie, '-', color='#ff7f0e', linewidth=1.5, alpha=0.8)
w_mask = (weimer_times >= start_dt) & (weimer_times <= end_dt)
ax.plot(weimer_times[w_mask], d['weimer_cpcp'][w_mask], '-', color='#2ca02c', alpha=0.8, linewidth=1.5)
d_mask = (dmsp_times >= start_dt) & (dmsp_times <= end_dt)
dt_dmsp, cpcp_dmsp = dmsp_times[d_mask], d['dmsp_cpcp'][d_mask]
ax.errorbar(mdates.date2num(dt_dmsp), cpcp_dmsp, xerr=d['dmsp_xerr_days'][:, d_mask], fmt='o', markersize=5, color='magenta',
            ecolor='magenta', elinewidth=1.5, capsize=0, label='DMSP F-13', zorder=4)
ax.set_ylabel(r'$\Phi_{PC}$ [kV]', fontsize=21)
ax.set_ylim(0, 300)
ax.grid(True)
ax.text(0.02, 0.90, '(c)', transform=ax.transAxes, fontsize=21, fontweight='bold')


def nearest_match(query_dts, src_dts, src_vals, tol_minutes=30):
    src_nums = mdates.date2num(np.asarray(src_dts))
    out = np.full(len(query_dts), np.nan)
    for i, q in enumerate(query_dts):
        j = np.argmin(np.abs(src_nums - mdates.date2num(q)))
        if abs(src_nums[j] - mdates.date2num(q)) * 24 * 60 <= tol_minutes:
            out[i] = src_vals[j]
    return out


r_dhem, n_dhem = pearson(nearest_match(dt_dmsp, evt_times, evt_ddpm), cpcp_dmsp)
r_amie, n_amie = pearson(nearest_match(dt_dmsp, evt_times, evt_amie), cpcp_dmsp)
r_weimer, n_weimer = pearson(nearest_match(dt_dmsp, weimer_times[w_mask], d['weimer_cpcp'][w_mask]), cpcp_dmsp)
r_val_amie = np.corrcoef(evt_amie, evt_ddpm)[0, 1]
print(f'r(DHEM-EP, DMSP)={r_dhem:.3f} (n={n_dhem}), r(AMIE, DMSP)={r_amie:.3f} (n={n_amie}), r(Weimer, DMSP)={r_weimer:.3f} (n={n_weimer}); '
      f'r(DHEM-EP, AMIE)={r_val_amie:.3f} (N={len(evt_times)})')


def legend_row(name, line_color, r, fontsize=15):
    swatch = DrawingArea(1.6 * fontsize, 0.7 * fontsize, 0, 0)
    swatch.add_artist(Line2D([0, 1.6 * fontsize], [0.35 * fontsize, 0.35 * fontsize], color=line_color, linewidth=2))
    parts = [swatch, TextArea(f' {name} ', textprops=dict(color='black', fontsize=fontsize))]
    if r is not None:
        parts.append(TextArea(f'(r={r:.2f})', textprops=dict(color='magenta', fontsize=fontsize)))
    return HPacker(children=parts, align='center', pad=0, sep=2)


legend_box = VPacker(children=[legend_row('DHEM-EP', '#1f77b4', r_dhem), legend_row('AMIE', '#ff7f0e', r_amie),
                               legend_row('Weimer', '#2ca02c', r_weimer), legend_row('DMSP F-13', 'magenta', None)], align='left', pad=4, sep=7)
anchored = AnchoredOffsetbox(loc='upper right', child=legend_box, frameon=True, pad=0.5, borderpad=0.6)
anchored.patch.set_facecolor('white'); anchored.patch.set_alpha(0.6); anchored.patch.set_edgecolor('0.85')
ax.add_artist(anchored)
ax.text(0.02, 0.83, f'r(DHEM-EP, AMIE) = {r_val_amie:.2f}', transform=ax.transAxes, fontsize=17, fontweight='normal', color='black', ha='left', va='top')

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
    a.axvspan(datetime(2000, 4, 6, 16, 50), datetime(2000, 4, 7, 1, 0), alpha=0.15, color='yellow', zorder=0)
    for vt in (datetime(2000, 4, 6, 8, 6), datetime(2000, 4, 6, 22, 44)):
        a.axvline(vt, color='black', linestyle='--', linewidth=1.5, zorder=0)
plt.subplots_adjust(left=0.135, right=0.90, top=0.97, bottom=0.08, hspace=0.05)
save(fig, 'Fig06_CPCP_Event01', dpi=300)
