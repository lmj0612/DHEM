#!/usr/bin/env python
"""Generate DHEM maps.

  python scripts/sample.py --family dhem_ap  --conditions <npz> --out out/ap.npz    # particle energy flux, four category models
  python scripts/sample.py --family dhem_ape --conditions <npz> --out out/ape.npz   # average energy
  python scripts/sample.py --family dhem_ep  --conditions <npz> --out out/ep.npz    # electric potential
  python scripts/sample.py --config configs/<model>.json --conditions <npz> --out out/x.npz   # one model

--mode timeseries (default): AR(1)-correlated noise (--tau 120 min), condition vectors averaged over --cond-smooth 5 min,
  raised-cosine blend of the two category models over +-(--xfade 5) min at a category switch (Figures 6-8 of the paper).
--mode independent: new noise and raw 1-min conditions for every minute, each minute by its own category model (Figures 2-5).
--members R realizations per minute; --seed (member k uses numpy default_rng(seed + 1000 k + 17)); --shard i/n, merged
  with scripts/merge_shards.py.

Conditions npz: family files hold timestamps, forsyth_category and cond_cat1 .. cond_cat4 (dhem_ap, dhem_ape) or conditions
(dhem_ep); see scripts/data_prep/make_family_conditions.py. With --config and a category model only the minutes of that
category are generated (--no-cat-filter to disable).
Output npz: samples [N, R, 32, 32] (normalised), physical [N, R, 32, 32] (keV | mW/m2 | kV), mean_physical, timestamps,
npy_names, forsyth_category, rows (index in the conditions file), model_weights [N, 5], and the run metadata.
DDIM, 1000 steps, eta = 0, single process.
"""
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
import torch as th

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from dhem.script_util import model_and_diffusion_defaults, create_model_and_diffusion  # noqa: E402
from dhem.spatial_basis import default_basis, sml_smu_channels  # noqa: E402
from dhem import metrics  # noqa: E402
from dhem import timeseries as ts_mod  # noqa: E402

ARCH_KEYS = ["image_size", "num_channels", "num_res_blocks", "num_heads", "num_heads_upsample", "attention_resolutions",
             "dropout", "learn_sigma", "use_scale_shift_norm", "diffusion_steps", "noise_schedule", "rescale_timesteps",
             "rescale_learned_sigmas", "use_circ_conv_cond", "use_cross_attn", "cond_drop", "C_spatial",
             "xattn_heads", "sector_drop"]


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def build_model(cfg, weights, device):
    d = model_and_diffusion_defaults()
    d.update({k: cfg["architecture"][k] for k in ARCH_KEYS if k in cfg["architecture"]})
    d.update(cond_dim=int(cfg["cond_dim"]), cond_spatial=None, class_cond=False, timestep_respacing="")
    model, diffusion = create_model_and_diffusion(**d)
    sd = th.load(weights, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    return model, diffusion


def family_configs(family):
    """{model key: config dict}: category 1..4 for dhem_ap / dhem_ape, 0 for dhem_ep."""
    if family == "dhem_ep":
        f = sorted(glob.glob(os.path.join(ROOT, "configs", "dhem_ep_*.json")))
        assert len(f) == 1, f"expected one dhem_ep config, found {f}"
        return {0: json.load(open(f[0]))}
    out = {}
    for k in (1, 2, 3, 4):
        f = sorted(glob.glob(os.path.join(ROOT, "configs", f"{family}_cat{k}_*.json")))
        assert len(f) == 1, f"expected one {family} cat{k} config, found {f}"
        out[k] = json.load(open(f[0]))
    return out


def spatial_channels(basis, C, cb, H, W):
    B = cb.shape[0]
    if C == 2:
        return np.tile(basis[:2][None], (B, 1, 1, 1))
    if C == 3:
        return np.tile(basis[:3][None], (B, 1, 1, 1))
    if C == 4:
        return np.concatenate([np.tile(basis[:2][None], (B, 1, 1, 1)), sml_smu_channels(cb[:, :48], W, H)], axis=1)
    raise ValueError(C)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--family", choices=["dhem_ap", "dhem_ape", "dhem_ep"])
    g.add_argument("--config")
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default=None, help="--config only: weights path (default: from the config)")
    ap.add_argument("--mode", choices=["timeseries", "independent"], default="timeseries")
    ap.add_argument("--tau", type=float, default=ts_mod.DEFAULTS["tau_min"], help="timeseries: AR(1) e-folding time [min]")
    ap.add_argument("--cond-smooth", type=int, default=ts_mod.DEFAULTS["cond_smooth_min"], help="timeseries: centred driver average [min]")
    ap.add_argument("--xfade", type=int, default=ts_mod.DEFAULTS["xfade_min"], help="timeseries, category families: blend half-width [min]")
    ap.add_argument("--members", "--n-repeats", dest="members", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if th.cuda.is_available() else "cpu")
    ap.add_argument("--shard", default="0/1", help="i/n: process minutes i::n")
    ap.add_argument("--no-cat-filter", action="store_true")
    ap.add_argument("--max-rows", type=int, default=0, help="debug: only the first k minutes")
    a = ap.parse_args()
    dev = th.device(a.device); assets = os.path.join(ROOT, "assets")

    # ---- models and condition blocks ----
    c = np.load(a.conditions, allow_pickle=True)
    if a.family:
        cfgs = family_configs(a.family)
        if a.family == "dhem_ep":
            conds = {0: np.asarray(c["conditions"], np.float32)}
        else:
            missing = [k for k in (1, 2, 3, 4) if f"cond_cat{k}" not in c.files]
            assert not missing, f"--family {a.family} needs a family conditions file with cond_cat1..cond_cat4 (missing {missing}); see scripts/data_prep/make_family_conditions.py"
            conds = {k: np.asarray(c[f"cond_cat{k}"], np.float32) for k in (1, 2, 3, 4)}
        weights = {k: os.path.join(ROOT, cfg["weights"]) for k, cfg in cfgs.items()}
        model_id = a.family
    else:
        cfg = json.load(open(a.config)); k0 = int(cfg.get("category_filter") or 0)
        key = "conditions" if "conditions" in c.files else f"cond_cat{k0}"
        cfgs = {k0: cfg}; conds = {k0: np.asarray(c[key], np.float32)}
        weights = {k0: a.weights or os.path.join(ROOT, cfg["weights"])}
        model_id = cfg["model_id"]
    N0 = next(iter(conds.values())).shape[0]
    for k, cfg in cfgs.items():
        assert conds[k].shape == (N0, cfg["cond_dim"]), f"model {cfg['model_id']}: conditions {conds[k].shape}, expected ({N0}, {cfg['cond_dim']})"
    decs = {json.dumps(cfg["decode"], sort_keys=True) for cfg in cfgs.values()}; assert len(decs) == 1, "models decode differently"
    meta = {k: np.asarray(c[k]) for k in ("timestamps", "npy_names", "forsyth_category") if k in c.files}
    cat = meta["forsyth_category"].astype(int) if "forsyth_category" in meta else np.zeros(N0, int)

    # ---- time axis, noise, driver average, blend weights (computed over ALL rows of the file) ----
    mode = a.mode
    if "timestamps" in meta:
        tmin, _ = ts_mod.minutes_since(meta["timestamps"])
        assert len(np.unique(tmin)) == N0, "duplicate timestamps in the conditions file"
        order = np.argsort(tmin, kind="stable")
    else:
        if mode == "timeseries": print("[sample] no timestamps in the conditions file -> independent mode")
        mode = "independent"; tmin = np.arange(N0, dtype=np.float64); order = np.arange(N0)
    tmin = tmin[order]; cat = cat[order]; conds = {k: v[order] for k, v in conds.items()}; meta = {k: v[order] for k, v in meta.items()}
    R = int(a.members); H = W = 32
    multi = len(cfgs) > 1
    if mode == "timeseries":
        noise = ts_mod.noise_sequence(tmin, R, "ar1", tau=a.tau, hw=(H, W), seed=a.seed)
        conds = {k: ts_mod.smooth_conditions(v, tmin, a.cond_smooth) for k, v in conds.items()}
        wts = ts_mod.crossfade_weights(cat, tmin, a.xfade) if multi else None
    else:
        noise = ts_mod.noise_sequence(tmin, R, "iid", hw=(H, W), seed=a.seed)
        wts = ts_mod.crossfade_weights(cat, tmin, 0) if multi else None
    if wts is None:
        k0 = next(iter(cfgs)); wts = np.zeros((5, N0), np.float32); wts[k0] = 1.0

    # ---- rows to generate (category restriction for a single category model; shard; debug cap) ----
    rows = np.arange(N0)
    one = cfgs[next(iter(cfgs))]
    if not a.family and one.get("category_filter") and not a.no_cat_filter and "forsyth_category" in meta:
        rows = np.where(cat == int(one["category_filter"]))[0]
    i_sh, n_sh = (int(x) for x in a.shard.split("/"))
    rows = rows[i_sh::n_sh]
    if a.max_rows:
        rows = rows[: a.max_rows]
    N = len(rows)
    print(f"[sample] {model_id}: {N} minutes x {R} member(s), mode {mode}"
          + (f" (tau {a.tau:g} min, driver average {a.cond_smooth} min" + (f", phase blend +-{a.xfade} min" if multi else "") + ")" if mode == "timeseries" else "")
          + f", device {dev}, shard {a.shard}, {N0} rows in file", flush=True)

    basis = default_basis(os.path.join(assets, "amie_grid.npz"))
    maps = np.zeros((R, N, H, W), np.float32); wsum = np.zeros(N, np.float32)
    t0 = time.time(); done = 0; total = int(sum((wts[k, rows] > 0).sum() for k in cfgs)) * R
    for k, cfg in cfgs.items():
        loc = np.where(wts[k, rows] > 0)[0]
        if len(loc) == 0:
            continue
        model, diffusion = build_model(cfg, weights[k], dev)
        arch = cfg["architecture"]; C = int(arch["C_spatial"]); clip = bool(arch.get("clip_denoised", True)); eta = float(arch.get("ddim_eta", 0.0))
        assert arch.get("sampler", "ddim") == "ddim", "DHEM sampling is DDIM (deterministic map from the noise field)"
        pairs = [(m, j) for m in range(R) for j in loc]
        for s in range(0, len(pairs), a.batch_size):
            pb = pairs[s: s + a.batch_size]; mm = np.array([p[0] for p in pb]); jj = np.array([p[1] for p in pb])
            src = rows[jj]
            cb = conds[k][src]
            kw = {"cond": th.as_tensor(cb, device=dev), "cond_spatial": th.as_tensor(spatial_channels(basis, C, cb, H, W), device=dev)}
            x_T = th.as_tensor(noise[mm, src], device=dev)
            with th.no_grad():
                x = diffusion.ddim_sample_loop(model, (len(pb), 1, H, W), noise=x_T, clip_denoised=clip, model_kwargs=kw, eta=eta)
            maps[mm, jj] += wts[k, src][:, None, None] * x[:, 0].float().cpu().numpy()
            done += len(pb)
            print(f"  {cfg['model_id']}: {done}/{total} maps  {time.time() - t0:.0f}s", flush=True)
        wsum[loc] += wts[k, rows[loc]]
        del model
        if dev.type == "cuda":
            th.cuda.empty_cache()
    assert np.allclose(wsum, 1.0, atol=1e-4), "blend weights do not sum to 1"

    samples = maps.transpose(1, 0, 2, 3)
    dec = one["decode"]
    vmin, vmax = metrics.load_range(os.path.join(ROOT, dec["range_file"]), dec["channel"])
    phys = metrics.decode_model(samples, dec["kind"], vmin, vmax, eps_subtract=dec.get("eps_subtract", 0.0), floor=dec.get("floor")).astype(np.float32)
    cfg_out = dict(model_id=model_id, decode=dec, members={str(k): v for k, v in cfgs.items()})
    extra = {"conditions": conds[next(iter(cfgs))][rows]} if len(cfgs) == 1 else {}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez_compressed(a.out, samples=samples, physical=phys, mean_physical=phys.mean(axis=1), rows=order[rows],
                        model_weights=wts[:, rows].T.astype(np.float32), model_id=model_id, config=json.dumps(cfg_out),
                        weights=json.dumps({str(k): os.path.basename(v) for k, v in weights.items()}),
                        weights_sha256=json.dumps({str(k): sha256(v) for k, v in weights.items()}),
                        mode=mode, tau_min=a.tau if mode == "timeseries" else 0.0,
                        cond_smooth_min=a.cond_smooth if mode == "timeseries" else 0,
                        xfade_min=a.xfade if (multi and mode == "timeseries") else 0,
                        n_repeats=R, members=R, seed=a.seed, shard=a.shard, sampler="ddim", elapsed_s=time.time() - t0,
                        device=str(dev), torch_version=th.__version__, **extra, **{k: v[rows] for k, v in meta.items()})
    print(f"[sample] saved {a.out}  ({N} minutes x {R} member(s), {time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
