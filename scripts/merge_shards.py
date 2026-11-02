#!/usr/bin/env python
"""Merge the --shard i/n outputs of scripts/sample.py into one file in time order.

  python scripts/merge_shards.py --out out/ap.npz out/ap_shard0.npz out/ap_shard1.npz ...
"""
import argparse, json
import numpy as np

PER_RUN = {"elapsed_s", "shard", "device"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    zs = [np.load(p, allow_pickle=True) for p in a.shards]
    n = [len(z["rows"]) for z in zs]; keys = zs[0].files
    out = {}
    for k in keys:
        v0 = zs[0][k]
        if getattr(v0, "ndim", 0) >= 1 and v0.shape[0] == n[0] and all(z[k].shape[0] == m for z, m in zip(zs, n)):
            out[k] = np.concatenate([z[k] for z in zs])
        else:
            if k not in PER_RUN:
                assert all(np.array_equal(z[k], v0) for z in zs), f"field {k} differs between shards"
            out[k] = v0
    order = np.argsort(out["rows"], kind="stable"); N = len(order)
    assert len(np.unique(out["rows"])) == N, "overlapping shards"
    for k, v in out.items():
        if getattr(v, "ndim", 0) >= 1 and v.shape[0] == N and k != "rows":
            out[k] = v[order]
    out["rows"] = out["rows"][order]; out["shard"] = np.array(f"merged {len(zs)} shards"); out["elapsed_s"] = float(sum(float(z["elapsed_s"]) for z in zs))
    np.savez_compressed(a.out, **out)
    print(f"saved {a.out}: {N} minutes from {len(zs)} shards ({json.loads(str(out['config']))['model_id']})")


if __name__ == "__main__":
    main()
