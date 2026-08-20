# quant-bill — what does quantization cost YOUR model?

Simulated-lattice quantizer from the Glass Loop program's quantization-bill
instrument (fp540/fp543): **round-to-nearest, per-channel symmetric int8 or
int4** over the attention + MLP linear weights, saved as an ordinary
state-dict checkpoint — designed to chain straight into the sibling
[`panel-bill`](../panel-bill/) tool so the cost is billed the way that
program bills weight edits: a casualty list by name, deficits in nats,
open-set rank-1 losses. Not an abstract perplexity delta — *which* of your
facts died.

## The grid

```
qmax = 2**(bits-1) - 1                       # 127 for int8, 7 for int4
flat = W.reshape(-1, W.shape[-1])            # channels = last dim
s    = flat.abs().amax(dim=0).clamp_min(1e-12) / qmax
q    = round(flat / s) * s                   # round half-to-even
```

This is the source program's arithmetic verbatim. It is *fake quant*
(weights land on the int grid but stay stored in float): it prices the
lattice, it does not produce a deployable int8 runtime artifact, and it is
deliberately less sophisticated than calibration-based methods (GPTQ/AWQ) —
an honest RTN baseline for "what does the grid alone cost".

## Two rules from the source program's defect record

**GQA rule (4omv).** transformer_lens names grouped-query K/V tensors
`_W_K` / `_W_V`. A leaf filter matching only `W_K`/`W_V` silently leaves the
K/V projections of every GQA model **unquantized** — your bill reads low and
you never know. Both spellings are in the default leaf set:
`W_Q, W_K, W_V, _W_K, _W_V, W_O, W_in, W_gate, W_out` (on `blocks.*` only).

**b_in rule — why this tool saves every leaf.** In the source program, a
quantizer that wrote only the quantized W tensors to its output checkpoint
**silently dropped a trained `b_in`** when chained after an edited
checkpoint: loading the output over a pristine base reverted the bias part
of the edit, and the damage bill quietly measured a model nobody had built.
This is a documented defect ("b_in defect class"). The output checkpoint
here therefore carries **every** leaf of the input state dict — quantized
where targeted, passed through bit-unchanged otherwise (biases, layer norms,
embeddings, buffers, all of it).

## Usage

```bash
# quantize a full state-dict checkpoint (torch required):
python quant_bill.py quantize --ckpt model_sd.pt --bits 8 \
    --out model_i8.pt --report i8_stats.json

# or load a transformer_lens model, optionally with a taught head on top:
python quant_bill.py quantize --model Qwen/Qwen2.5-3B-Instruct \
    --overlay taught_head.pt --bits 4 --out taught_i4.pt

# then bill the damage with panel-bill:
python ../panel-bill/panel_bill.py measure --panel panel.json --arm base \
    --model Qwen/Qwen2.5-3B-Instruct --out meas_base.json
python ../panel-bill/panel_bill.py measure --panel panel.json --arm i8 \
    --model Qwen/Qwen2.5-3B-Instruct --checkpoint model_i8.pt --out meas_i8.json
python ../panel-bill/panel_bill.py bill --base meas_base.json --current meas_i8.json
```

`--report` writes per-tensor error stats (shape, max abs error, RMSE, max
channel scale). `--leaves` overrides the leaf set for non-TL naming schemes.
`--bits` accepts any of 2..8 (the source program also ran 3-bit arms).

## Caveats — read before citing

- **Simulated lattice only.** The bill prices round-to-nearest onto the int
  grid; a deployment stack with different scales, group sizes, activation
  quantization, or calibration will cost differently (usually less than RTN
  int4, comparably at int8).
- **Naming scheme matters.** The default leaf filter is transformer_lens
  naming. For HF-native or other state dicts, pass `--leaves` and *verify
  the quantized-tensor count in the report* — a filter that matches nothing
  produces a bit-perfect (and worthless) "quantized" checkpoint.
- **Channels = last dim.** For TL weight layouts (`[..., d_model, d_head]`,
  `[d_model, d_mlp]`) this matches the source program. Other layouts change
  which axis gets per-channel scales; check your shapes.
- The overlay path refuses checkpoints with unexpected keys (loading a
  wrong file with `strict=False` otherwise quantizes the pristine base —
  same failure family as the b_in defect).
- Deterministic: RTN with half-to-even rounding (torch/numpy/python `round`
  all agree), no data, no calibration, no randomness.

## Selftest

```bash
python selftest.py
```

Pure python, no torch/numpy, no downloads: hand-computed grid values
including the half-to-even ties, GQA-shape handling with cross-head shared
channel scales, degenerate all-zero channels, every-leaf preservation, and
determinism.

No telemetry. MIT license.
