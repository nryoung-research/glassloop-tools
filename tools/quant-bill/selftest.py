# -*- coding: utf-8 -*-
"""quant-bill selftest: pure python, no torch, no numpy, no downloads.

Verifies the grid arithmetic against hand-computed values (including the
round-half-to-even ties that torch.round produces), leaf preservation (the
b_in rule), GQA-shape handling with cross-head shared per-channel scales,
and the RTN error bound.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quant_bill import (DEFAULT_LEAVES, is_quant_target,  # noqa: E402
                        quantize_state_dict, quantize_tensor)

CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def close(a, b, tol=1e-9):
    return abs(a - b) <= tol


def deep_eq(a, b, tol=1e-9):
    if isinstance(a, list):
        return len(a) == len(b) and all(deep_eq(x, y, tol)
                                        for x, y in zip(a, b))
    return close(a, b, tol)


# hand-check tensor: 4 rows x 2 channels
W_IN = [[7.0, 12.7],
        [2.5, -1.5],
        [-3.5, 0.3],
        [1.0, 14.0]]
# bits=4 (qmax=7): col0 scale 7/7=1, col1 scale 14/7=2
# col0: round(7)=7, round(2.5)=2 (tie->even), round(-3.5)=-4 (tie->even),
#       round(1)=1
# col1: 12.7/2=6.35->6->12.0; -1.5/2=-0.75->-1->-2.0; 0.3/2=0.15->0->0.0;
#       14/2=7->14.0
W_IN_Q4 = [[7.0, 12.0],
           [2.0, -2.0],
           [-4.0, 0.0],
           [1.0, 14.0]]

# GQA-shaped K: [n_kv_heads=1, d_model=4, d_head=3]
W_K_GQA = [[[3.0, -6.0, 0.9],
            [1.5, 2.0, -0.3],
            [-0.75, 4.0, 0.6],
            [0.375, -8.0, -1.2]]]

# Q: [n_heads=2, d_model=4, d_head=3]; col-0 max (7.0) lives in head 1, so
# head 0's 3.5 must land on the SHARED cross-head grid (scale 1 -> 4.0),
# not its own per-head grid (which would keep 3.5 exact)
W_Q = [[[3.5, 1.0, 0.5], [0.0, 0.0, 0.0],
        [1.0, -1.0, 2.0], [2.0, 0.5, -0.25]],
       [[7.0, 2.0, 1.0], [0.0, 0.0, 0.0],
        [-1.0, 1.0, -4.0], [3.0, 0.25, 0.125]]]

SD = {
    "embed.W_E": [[0.1, 0.2], [0.3, 0.4]],
    "blocks.0.attn.W_Q": W_Q,
    "blocks.0.attn._W_K": W_K_GQA,
    "blocks.0.attn._W_V": [[[1.0, -2.0, 0.5], [0.25, 0.75, -1.0],
                            [2.0, 1.0, 0.0], [-4.0, 0.5, 0.25]]],
    "blocks.0.attn.W_K": [[6.0, 0.5], [3.0, -0.25]],
    "blocks.0.attn.b_Q": [0.01, 0.02],
    "blocks.0.mlp.W_in": W_IN,
    "blocks.0.mlp.b_in": [0.1, -0.2, 0.3, 0.0, 5.5, -7.25],
    "blocks.0.mlp.b_out": [0.5, 0.5],
    "blocks.0.ln1.w": [1.0, 1.0],
    "blocks.1.mlp.W_gate": [[1.0, -2.0], [0.5, 2.0]],
    "blocks.1.mlp.W_out": [[0.25, 0.5], [1.0, -1.0]],
    "unembed.W_U": [[9.0, 9.0], [9.0, 9.0]],
}
EXPECT_QUANTIZED = {
    "blocks.0.attn.W_Q", "blocks.0.attn._W_K", "blocks.0.attn._W_V",
    "blocks.0.attn.W_K", "blocks.0.mlp.W_in", "blocks.1.mlp.W_gate",
    "blocks.1.mlp.W_out",
}


def flat_rows(t):
    if t and isinstance(t[0], list):
        for sub in t:
            yield from flat_rows(sub)
    else:
        yield t


def grid_ok(W, Q, qmax, tol=1e-6):
    """Independent grid check: per channel (last dim, flattened over all
    leading dims), q/scale must be an integer with |int| <= qmax, and the
    max-abs element must reconstruct exactly."""
    rw, rq = list(flat_rows(W)), list(flat_rows(Q))
    cols = len(rw[0])
    for j in range(cols):
        amax = max(abs(r[j]) for r in rw)
        s = max(amax, 1e-12) / qmax
        for r, q in zip(rw, rq):
            k = q[j] / s
            if abs(k - round(k)) > tol or abs(round(k)) > qmax:
                return False
        # the channel's max-abs element sits exactly on the grid edge
        for r, q in zip(rw, rq):
            if abs(abs(r[j]) - amax) < 1e-15 and not close(q[j], r[j], tol):
                return False
    return True


def main():
    print("leaf filter (source verbatim, incl. GQA 4omv rule):")
    check(is_quant_target("blocks.0.attn._W_K"),
          "GQA '_W_K' IS a target (a W_K-only filter silently skips it)")
    check(is_quant_target("blocks.7.mlp.W_in"), "'W_in' is a target")
    check(not is_quant_target("blocks.0.mlp.b_in"), "'b_in' is NOT quantized")
    check(not is_quant_target("unembed.W_U"),
          "non-blocks tensors are never touched")

    print("grid arithmetic, hand-computed (int4, qmax=7):")
    q4, st4 = quantize_tensor(W_IN, 4)
    check(deep_eq(q4, W_IN_Q4),
          "W_in int4 matches hand values incl. half-to-even ties "
          "(2.5->2, -3.5->-4)")
    check(close(st4["max_abs_err"], 0.7, 1e-6),
          "per-tensor max_abs_err = 0.7 (the 12.7->12.0 cell)")
    check(st4["max_abs_err"] <= max(1.0, 2.0) / 2 + 1e-9,
          "RTN bound holds: max error <= max_scale/2")
    q8, st8 = quantize_tensor(W_IN, 8)
    check(st8["max_abs_err"] < st4["max_abs_err"],
          f"int8 error < int4 error ({st8['max_abs_err']:.4f} < 0.7)")
    exp = round(2.5 * 127 / 7.0) * 7.0 / 127
    check(close(q8[1][0], exp),
          f"int8 cell check: 2.5 -> {exp:.6f} (45/127 grid)")

    print("degenerate grids:")
    qz, _ = quantize_tensor([[0.0, 0.0], [0.0, 0.0]], 8)
    check(deep_eq(qz, [[0.0, 0.0], [0.0, 0.0]]),
          "all-zero channel survives the 1e-12 scale clamp (no div-by-zero)")
    q2, _ = quantize_tensor([[-1.0], [0.4], [0.6], [-0.2], [1.0]], 2)
    check(set(v[0] for v in q2) <= {-1.0, 0.0, 1.0},
          "bits=2 grid has at most 2*qmax+1 = 3 levels")

    print("GQA shapes and cross-head scales:")
    sd_q4, rep = quantize_state_dict(SD, 4)
    check([len(x) for x in [sd_q4["blocks.0.attn._W_K"],
                            sd_q4["blocks.0.attn._W_K"][0],
                            sd_q4["blocks.0.attn._W_K"][0][0]]] == [1, 4, 3],
          "_W_K keeps its [n_kv_heads, d_model, d_head] = [1,4,3] shape")
    check(grid_ok(W_K_GQA, sd_q4["blocks.0.attn._W_K"], 7),
          "_W_K on-grid per channel; channel maxes (3.0/-8.0/-1.2) exact")
    check(grid_ok(W_Q, sd_q4["blocks.0.attn.W_Q"], 7),
          "W_Q on-grid per channel across both heads")
    check(close(sd_q4["blocks.0.attn.W_Q"][0][0][0], 4.0),
          "head-0 3.5 lands on the SHARED cross-head grid (scale from "
          "head-1's 7.0): 3.5 -> 4.0, not exact 3.5")
    check(close(sd_q4["blocks.0.attn.W_Q"][1][0][0], 7.0),
          "head-1 7.0 (the channel max) reconstructs exactly")

    print("leaf preservation (the b_in rule):")
    check(set(sd_q4.keys()) == set(SD.keys()),
          "output checkpoint carries EVERY input leaf")
    check(set(r["name"] for r in rep["per_tensor"]) == EXPECT_QUANTIZED,
          f"exactly {len(EXPECT_QUANTIZED)} tensors quantized")
    for k in sorted(set(SD) - EXPECT_QUANTIZED):
        check(deep_eq(sd_q4[k], SD[k], 0.0),
              f"passthrough leaf bit-unchanged: {k}")
    check(rep["n_quantized"] == 7 and rep["n_passthrough"] == 6,
          "report counts: 7 quantized, 6 preserved")
    check("b_in" in rep["note"], "report carries the b_in defect-class note")

    print("determinism:")
    sd_q4b, _ = quantize_state_dict(SD, 4)
    check(all(deep_eq(sd_q4[k], sd_q4b[k], 0.0) for k in sd_q4),
          "two runs produce identical checkpoints")

    print(f"SELFTEST PASS (quant-bill): {CHECKS} checks")


if __name__ == "__main__":
    main()
