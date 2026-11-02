#!/usr/bin/env python
"""HP / CPCP / HME of generated maps against AMIE maps and/or against the paper run.

  python scripts/evaluate.py --samples out/ap.npz [--amie-dir <AMIE npy of these minutes>] \
      [--paper examples/2003-11-20/paper_outputs/dhem_ap_flux_maps_2003-11-20.npz] --out out/metrics.json

Channel 1 -> HP [GW], channel 2 -> CPCP [kV], channel 0 -> HME [keV] (needs the AMIE flux). Reports Pearson r, bias,
95 % limits of agreement and MAE. A new realization is not expected to match the paper maps minute by minute; the
statistics agree within their sampling uncertainty. The AMIE maps are not part of this archive.
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dhem import metrics  # noqa: E402
from dhem.spatial_basis import mlat_rows  # noqa: E402

TEST_RANGE = os.path.join(ROOT, "assets", "amie_range_test20.json")


def load_samples(paths):
    phys, names, ts, mid, ch = [], [], [], set(), set()
    for p in paths:
        d = np.load(p, allow_pickle=True)
        cfg = json.loads(str(d["config"]))
        phys.append(d["mean_physical"]); names.append(d["npy_names"]); ts.append(d["timestamps"]); mid.add(cfg["model_id"]); ch.add(cfg["decode"]["channel"])
    assert len(ch) == 1, f"mixed channels {ch}"
    return np.concatenate(phys), np.concatenate(names), np.concatenate(ts), sorted(mid), ch.pop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", nargs="+", required=True)
    ap.add_argument("--amie-dir", default=None, help="folder with the AMIE maps of these minutes (not included in the archive); without it only the paper run is compared")
    ap.add_argument("--paper", default=None, help="paper_outputs npz with 'maps' (normalised) and 'timestamps'")
    ap.add_argument("--flux-samples", nargs="*", default=None, help="HME only: DHEM flux sample files (else AMIE flux weights both)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    phys, names, ts, models, ch = load_samples(a.samples)
    order = np.argsort(ts); phys, names, ts = phys[order], names[order], ts[order]
    aw = metrics.area_weights(mlat_rows())
    qty = {1: "HP [GW]", 2: "CPCP [kV]", 0: "HME [keV]"}[ch]
    assert a.amie_dir or a.paper, "give --amie-dir and/or --paper"
    assert a.amie_dir or ch != 0, "HME needs the AMIE flux maps (--amie-dir)"
    ref, mod = np.full(len(ts), np.nan), np.zeros(len(ts))
    for k, (n, m) in enumerate(zip(names, phys)):
        npy = np.load(os.path.join(a.amie_dir, str(n))) if a.amie_dir else None
        if ch == 1:
            mod[k] = metrics.hemispheric_power(m, aw)
            if npy is not None: ref[k] = metrics.hemispheric_power(metrics.decode_amie(npy, 1, TEST_RANGE), aw)
        elif ch == 2:
            mod[k] = metrics.cpcp(m)
            if npy is not None: ref[k] = metrics.cpcp(metrics.decode_amie(npy, 2, TEST_RANGE))
        else:
            F = np.maximum(metrics.decode_amie(npy, 1, TEST_RANGE), 0.0)
            ref[k] = metrics.hme(F, metrics.decode_amie(npy, 0, TEST_RANGE, floor=1e-3), aw); mod[k] = metrics.hme(F, np.maximum(m, 1e-3), aw)
    res = {"quantity": qty, "models": models, "N": int(len(ts)), "model_mean": float(mod.mean())}
    if a.amie_dir:
        res.update({"this_run": metrics.bland_altman(ref, mod), "amie_mean": float(ref.mean())})
    if a.paper:
        P = np.load(a.paper, allow_pickle=True)
        pts = {str(t): i for i, t in enumerate(P["timestamps"])}
        sel = np.array([pts[str(t)] for t in ts])
        # paper maps are normalised; decode with the training range of this channel the same way as sample.py
        cfg = json.loads(str(np.load(a.samples[0], allow_pickle=True)["config"]))
        dec = cfg["decode"]; vmin, vmax = metrics.load_range(os.path.join(ROOT, dec["range_file"]), dec["channel"])
        pm = metrics.decode_model(P["maps"][sel], dec["kind"], vmin, vmax, eps_subtract=dec.get("eps_subtract", 0.0), floor=dec.get("floor"))
        pap = np.zeros(len(ts))
        for k, m in enumerate(pm):
            if ch == 1: pap[k] = metrics.hemispheric_power(m, aw)
            elif ch == 2: pap[k] = metrics.cpcp(m)
            else:
                npy = np.load(os.path.join(a.amie_dir, str(names[k]))); F = np.maximum(metrics.decode_amie(npy, 1, TEST_RANGE), 0.0); pap[k] = metrics.hme(F, np.maximum(m, 1e-3), aw)
        if a.amie_dir: res["paper_run"] = metrics.bland_altman(ref, pap)
        res["this_vs_paper"] = metrics.bland_altman(pap, mod)
    print(f"\n{qty}  N={res['N']}  models={models}")
    print(f"{'set':14s} {'r':>6s} {'bias':>8s} {'LoA':>20s} {'MAE':>7s}")
    for key in ("this_run", "paper_run", "this_vs_paper"):
        if key in res:
            b = res[key]; print(f"{key:14s} {b['r']:6.3f} {b['bias']:8.2f} [{b['loa_lo']:8.2f},{b['loa_hi']:8.2f}] {b['mae']:7.2f}")
    print((f"AMIE mean {res['amie_mean']:.2f}, " if "amie_mean" in res else "") + f"model mean {res['model_mean']:.2f}")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True); json.dump(res, open(a.out, "w"), indent=2); print("saved", a.out)


if __name__ == "__main__":
    main()
