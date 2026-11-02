#!/usr/bin/env python
"""Compare a time-series run of scripts/sample.py with the paper maps of the April 2000 storm.

  python scripts/compare_event.py --samples out/ap.npz --paper examples/2000-04-06/paper_outputs/dhem_ap_flux_maps_2000-04-06.npz \
      [--out out/compare_ap.json] [--plot out/compare_ap.png]

With the seeds of scripts/run_example_event.sh the sampler regenerates the paper maps from the same noise fields, so the two
runs agree to within GPU floating-point differences. Reports the max / RMS map difference, the HP (or CPCP) series of both runs,
their correlation and largest difference, and the RMS 1-min change of each series.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dhem import metrics  # noqa: E402
from dhem.spatial_basis import mlat_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True); ap.add_argument("--paper", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--plot", default=None)
    a = ap.parse_args()
    d = np.load(a.samples, allow_pickle=True); P = np.load(a.paper, allow_pickle=True)
    cfg = json.loads(str(d["config"])); dec = cfg["decode"]
    pos = {str(t): i for i, t in enumerate(P["timestamps"])}
    ts = [str(t) for t in d["timestamps"]]; sel = np.array([pos[t] for t in ts])
    m_new = d["samples"][:, 0]; m_pap = P["maps"][sel]
    diff = np.abs(m_new - m_pap); per_map = diff.reshape(len(ts), -1).max(axis=1)
    vmin, vmax = metrics.load_range(os.path.join(ROOT, dec["range_file"]), dec["channel"])
    dn = lambda m: metrics.decode_model(m, dec["kind"], vmin, vmax, eps_subtract=dec.get("eps_subtract", 0.0), floor=dec.get("floor"))
    if dec["channel"] == 1:
        aw = metrics.area_weights(mlat_rows()); q = "HP [GW]"
        f = lambda m: np.array([metrics.hemispheric_power(x, aw) for x in dn(m)])
    elif dec["channel"] == 2:
        q = "CPCP [kV]"; f = lambda m: np.array([metrics.cpcp(x) for x in dn(m)])
    else:
        q = "mean of the map [keV]"; f = lambda m: dn(m).reshape(len(m), -1).mean(axis=1)
    s_new, s_pap = f(m_new), f(m_pap)
    tmin = np.array([np.datetime64(t.replace(' ', 'T'), 'm') for t in ts]); consec = np.diff(tmin) == np.timedelta64(1, 'm')
    jump = lambda s: float(np.sqrt(np.mean(np.diff(s)[consec] ** 2))) if consec.any() else float('nan')
    res = dict(quantity=q, N=len(ts), mode=str(d["mode"]), seed=int(d["seed"]),
               map_absdiff_max=float(diff.max()), map_absdiff_rms=float(np.sqrt((diff ** 2).mean())),
               frac_maps_within_0p01=float((per_map < 1e-2).mean()), series_r=float(np.corrcoef(s_new, s_pap)[0, 1]),
               series_absdiff_max=float(np.abs(s_new - s_pap).max()), series_mean_this=float(s_new.mean()), series_mean_paper=float(s_pap.mean()),
               rms_1min_change_this=jump(s_new), rms_1min_change_paper=jump(s_pap))
    print(json.dumps(res, indent=2))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=2)
    if a.plot:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 3.5)); x = tmin.astype("datetime64[m]").astype(object)
        ax.plot(x, s_pap, lw=2.5, color="0.7", label="paper run"); ax.plot(x, s_new, lw=0.9, color="C0", label=f"this run ({res['mode']})")
        ax.set_ylabel(q); ax.legend(loc="upper right"); ax.set_title(f"{cfg['model_id']}: r = {res['series_r']:.4f}, max |diff| = {res['series_absdiff_max']:.2f}")
        fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(a.plot, dpi=150); print("saved", a.plot)


if __name__ == "__main__":
    main()
