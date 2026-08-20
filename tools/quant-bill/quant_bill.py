# -*- coding: utf-8 -*-
"""quant-bill: what does quantization cost YOUR model?

Simulated-lattice quantizer ported from the Glass Loop program's
quantization-bill instrument (fp540_quantbill.py / fp543_i8screen.py):
round-to-nearest, per-channel symmetric int8/int4 over the attention + MLP
linear weights, applied in model space and saved as an ordinary state-dict
checkpoint — designed to chain straight into the sibling `panel-bill` tool
so the damage gets billed in nats, per item, by name.

The grid (source arithmetic, verbatim):

    qmax = 2**(bits-1) - 1
    flat = W.reshape(-1, W.shape[-1])          # channels = last dim
    s    = flat.abs().amax(dim=0).clamp_min(1e-12) / qmax
    q    = round(flat / s) * s                 # round half-to-even

Two rules carried from the source program's defect record:

  GQA RULE (4omv)   transformer_lens names grouped-query K/V tensors
                    `_W_K` / `_W_V`. A leaf filter that only matches
                    `W_K`/`W_V` silently leaves the K/V projections of a
                    GQA model UNQUANTIZED. Both spellings are in the
                    default leaf set.

  b_in RULE         every non-target leaf is PRESERVED in the output
                    checkpoint. In the source program, a quantizer that
                    saved only the quantized W tensors silently dropped a
                    trained `b_in` when chained after an edited checkpoint
                    — a documented defect ("b_in defect class"). This tool
                    therefore saves EVERY leaf of the input state dict:
                    quantized where targeted, passed through bit-unchanged
                    otherwise.

No telemetry. Deterministic. Selftest runs on CPU in pure python (no torch,
no numpy, no downloads).
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

# Attention + MLP linear weights, transformer_lens naming, GQA spellings
# included (4omv refuter rule).
DEFAULT_LEAVES: Tuple[str, ...] = ("W_Q", "W_K", "W_V", "_W_K", "_W_V",
                                   "W_O", "W_in", "W_gate", "W_out")


def is_quant_target(name: str, leaves: Sequence[str] = DEFAULT_LEAVES) -> bool:
    """Source filter, verbatim: a `blocks.*` parameter whose final name
    component is in the leaf set."""
    return name.startswith("blocks.") and name.split(".")[-1] in leaves


# ------------------------------------------------ pure-python tensor backend

def _shape(t: Any) -> List[int]:
    s = []
    while isinstance(t, list):
        s.append(len(t))
        t = t[0]
    return s


def _leaf_rows(t: list) -> Iterator[list]:
    """Innermost lists in order == reshape(-1, last_dim) rows."""
    if t and isinstance(t[0], list):
        for sub in t:
            yield from _leaf_rows(sub)
    else:
        yield t


def _rebuild(shape: List[int], rows: Iterator[list]) -> list:
    if len(shape) == 1:
        return next(rows)
    return [_rebuild(shape[1:], rows) for _ in range(shape[0])]


def _quantize_pylist(W: list, qmax: float) -> Tuple[list, Dict[str, Any]]:
    shape = _shape(W)
    rows = list(_leaf_rows(W))
    cols = shape[-1]
    scales = [max(max(abs(r[j]) for r in rows), 1e-12) / qmax
              for j in range(cols)]
    qrows = [[round(r[j] / scales[j]) * scales[j] for j in range(cols)]
             for r in rows]
    errs = [qr[j] - r[j] for r, qr in zip(rows, qrows) for j in range(cols)]
    n = len(errs)
    st = dict(shape=shape, numel=n,
              max_abs_err=max(abs(e) for e in errs),
              rmse=(sum(e * e for e in errs) / n) ** 0.5,
              max_scale=max(scales))
    return _rebuild(shape, iter(qrows)), st


# ------------------------------------------------------- torch/numpy backends

def _quantize_torch(W, qmax: float):
    import torch
    W = W.detach()
    flat = W.reshape(-1, W.shape[-1])
    s = flat.abs().amax(dim=0).clamp_min(1e-12) / qmax
    q = torch.round(flat / s) * s
    Q = q.reshape(W.shape).to(torch.float32).cpu()
    err = (Q - W.to(torch.float32).cpu()).reshape(-1)
    st = dict(shape=list(W.shape), numel=int(err.numel()),
              max_abs_err=float(err.abs().max()),
              rmse=float(err.pow(2).mean().sqrt()),
              max_scale=float(s.max()))
    return Q, st


def _quantize_numpy(W, qmax: float):
    import numpy as np
    flat = W.reshape(-1, W.shape[-1])
    s = np.maximum(np.abs(flat).max(axis=0), 1e-12) / qmax
    q = np.round(flat / s) * s  # np.round = half-to-even, same as torch
    Q = q.reshape(W.shape).astype(np.float32)
    err = (Q.astype(np.float64) - W.astype(np.float64)).ravel()
    st = dict(shape=list(W.shape), numel=int(err.size),
              max_abs_err=float(np.abs(err).max()),
              rmse=float(np.sqrt((err ** 2).mean())),
              max_scale=float(s.max()))
    return Q, st


def quantize_tensor(W: Any, bits: int):
    """Per-channel symmetric round-to-nearest on one tensor; backend chosen
    by the tensor's type (torch tensor / numpy array / nested python list)."""
    qmax = float(2 ** (int(bits) - 1) - 1)
    root = type(W).__module__.split(".")[0]
    if root == "torch":
        return _quantize_torch(W, qmax)
    if root == "numpy":
        return _quantize_numpy(W, qmax)
    if isinstance(W, list):
        return _quantize_pylist(W, qmax)
    raise TypeError(f"unsupported tensor type: {type(W)!r}")


# ---------------------------------------------------------------- state dict

def quantize_state_dict(sd: Dict[str, Any], bits: int,
                        leaves: Sequence[str] = DEFAULT_LEAVES):
    """Quantize the targeted leaves; pass EVERY other leaf through unchanged
    (b_in rule — see module docstring). Returns (out_sd, report)."""
    out: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    passthrough: List[str] = []
    for name, v in sd.items():
        if is_quant_target(name, leaves):
            q, st = quantize_tensor(v, bits)
            out[name] = q
            rows.append(dict(name=name, bits=int(bits), **st))
        else:
            out[name] = v
            passthrough.append(name)
    report = dict(bits=int(bits), leaves=list(leaves),
                  n_quantized=len(rows), n_passthrough=len(passthrough),
                  per_tensor=rows, passthrough_keys=passthrough,
                  note=("every non-target leaf is preserved in the output "
                        "checkpoint (b_in defect class: quantizers that "
                        "save only W tensors silently drop trained biases "
                        "when chained after an edited checkpoint)"))
    return out, report


# --------------------------------------------------------------------- CLI

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Per-channel symmetric int8/int4 RTN quantizer that "
                    "preserves every non-target leaf; chains into "
                    "panel-bill for the damage bill.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("quantize", help="model/ckpt -> quantized ckpt")
    src = q.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="input state-dict .pt (torch)")
    src.add_argument("--model", help="transformer_lens model name "
                                     "(loaded fp32)")
    q.add_argument("--overlay", default=None,
                   help="with --model: edited checkpoint overlaid before "
                        "quantizing (e.g. a taught head)")
    q.add_argument("--bits", type=int, required=True,
                   help="8 (int8) or 4 (int4); any 2..8 accepted")
    q.add_argument("--out", required=True, help="quantized checkpoint .pt")
    q.add_argument("--report", default=None,
                   help="per-tensor error stats json")
    q.add_argument("--leaves", default=None,
                   help="comma list overriding the default leaf set "
                        f"{','.join(DEFAULT_LEAVES)}")
    q.add_argument("--device", default=None)
    A = ap.parse_args(argv)

    if not (2 <= A.bits <= 8):
        raise SystemExit("--bits must be in 2..8 (8=int8, 4=int4)")
    leaves = tuple(A.leaves.split(",")) if A.leaves else DEFAULT_LEAVES

    import torch  # heavy import, CLI path only
    torch.set_grad_enabled(False)
    if A.ckpt:
        sd = torch.load(A.ckpt, map_location="cpu")
        if not isinstance(sd, dict):
            raise SystemExit("--ckpt must contain a state dict")
    else:
        from transformer_lens import HookedTransformer
        device = A.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = HookedTransformer.from_pretrained(A.model, device=device,
                                                  dtype=torch.float32)
        if A.overlay:
            osd = torch.load(A.overlay, map_location=device)
            missing = model.load_state_dict(osd, strict=False)
            del osd
            if len(missing.unexpected_keys) > 0:
                raise SystemExit("REFUSED: overlay has unexpected keys")
        sd = model.state_dict()

    out, report = quantize_state_dict(sd, A.bits, leaves)
    # save every leaf on cpu (quantized tensors are already cpu fp32)
    out = {k: (v.detach().cpu() if hasattr(v, "detach") else v)
           for k, v in out.items()}
    torch.save(out, A.out)
    print(f"QUANT-DONE bits={A.bits} quantized={report['n_quantized']} "
          f"preserved={report['n_passthrough']} -> {A.out}")
    worst = sorted(report["per_tensor"], key=lambda r: -r["max_abs_err"])[:5]
    for r in worst:
        print(f"  {r['name']}: shape {r['shape']} "
              f"max_abs_err {r['max_abs_err']:.6g} rmse {r['rmse']:.6g}")
    if A.report:
        json.dump(report, open(A.report, "w", encoding="utf-8"), indent=1)
        print(f"report -> {A.report}")
    print("next: bill the damage with the sibling panel-bill tool "
          f"(measure --checkpoint {A.out}, then bill vs your base "
          "measurement)")


if __name__ == "__main__":
    main()
