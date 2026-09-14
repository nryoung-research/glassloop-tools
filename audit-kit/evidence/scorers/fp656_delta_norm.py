#!/usr/bin/env python3
"""FP-656: realized weight-delta norms of an exported bf16 state vs the base snapshot on the written window (layers 28-35 MLP gate/up/down),
read from the safetensors shards without loading the models. Writes OUT/delta_norm.json (no-clobber) with per-matrix and total Frobenius
norms (bf16 values cast to fp32), and the total over the whole model for completeness. Usage: fp656_delta_norm.py BASE_DIR STATE_DIR OUT_DIR"""
import glob
import hashlib
import json
import os
import sys

import torch
from safetensors import safe_open

base_dir, state_dir, out_dir = sys.argv[1:4]
_LO, _HI = [int(x) for x in os.environ.get("FP656_LAYERS", "28-35").split("-")]   # FP-680: window configurable for other depths (default unchanged: 28-35)
LAYERS = range(_LO, _HI + 1)
WIN = {"model.layers.%d.mlp.%s.weight" % (L, m) for L in LAYERS for m in ("gate_proj", "up_proj", "down_proj")}


def tensors(d):
    out = {}
    for shard in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                out[k] = (shard, f)
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


def load_all(d, keys):
    res = {}
    for shard in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(shard, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k in keys:
                    res[k] = f.get_tensor(k).float()
    return res


def main():
    op = os.path.join(out_dir, "delta_norm.json")
    if os.path.exists(op):
        raise SystemExit("NO-CLOBBER %s" % op)
    b = load_all(base_dir, WIN)
    s = load_all(state_dir, WIN)
    missing = sorted(WIN - set(b)) + sorted(WIN - set(s))
    if missing:
        raise SystemExit("missing keys: %s" % missing[:4])
    per = {k: float((s[k] - b[k]).norm()) for k in sorted(WIN)}
    total = float(sum(v * v for v in per.values()) ** 0.5)
    # whole-model check: any change outside the window?
    keys_all = set()
    for shard in glob.glob(os.path.join(state_dir, "*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as f:
            keys_all |= set(f.keys())
    outside = 0.0
    bo = load_all(base_dir, keys_all - WIN)
    so = load_all(state_dir, keys_all - WIN)
    for k in so:
        if k in bo and bo[k].shape == so[k].shape:
            outside += float((so[k] - bo[k]).norm()) ** 2
    rec = {"kind": "fp656_delta_norm", "base_dir": base_dir, "state_dir": state_dir, "window_fro_total": total, "per_matrix": per, "outside_window_fro": outside ** 0.5,
           "manifest_sha256": (hashlib.sha256(open(os.path.join(state_dir, "MANIFEST-SHA256.json"), "rb").read()).hexdigest() if os.path.exists(os.path.join(state_dir, "MANIFEST-SHA256.json")) else None)}
    json.dump(rec, open(op, "w"), indent=1, sort_keys=True)
    print("DELTA-NORM window %.5f outside %.5f -> %s" % (total, outside ** 0.5, op))


if __name__ == "__main__":
    main()
