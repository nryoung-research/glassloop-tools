# panel-bill — damage billing for weight edits

Measure a model over a fixed probe panel, then bill the damage between any
two measurements — a base and an edited/quantized/fine-tuned arm — as a
**casualty list by name with deficits in nats**. Ported from the Glass Loop
program's sealed panel instrument (`fp467_measure_panel.py`) and the billing
arithmetic used throughout that program (fp540/fp543/fp544/fp547).

## What `measure` records per item

A panel item is `{id, templates: [...], candidates: [...], answer}`;
`templates[0]` contains an `{answer}` slot.

**Closed channel (margin).** For each candidate, substitute it into the
template and sum the model's log-probs over the *full* token sequence
(prompt tokens included — they cancel in the margin up to tokenization
shifts; this is the source instrument's exact arithmetic, kept byte-identical
so measurements stay comparable). Margin = `lp(answer) − max(lp(distractor))`
in nats. Per-candidate log-probs are also recorded so flip *destinations*
stay readable (which wrong answer won — the intrusion diagnostic).

**Open-set channel.** One extra forward at the position before the answer
slot, read over the FULL vocabulary: `{lp_ans, rank_ans, top1_outside,
mass_in_set}`. This channel is standard in the source program because the
closed margin reads only the candidates' logits — 16 of 151,665 in the
source setup — and was **measured to under-report damage ~1.79x**; on some
frames the closed read finds zero damage where the open read finds real
loss. Items whose template has no terminal answer slot, or whose candidates
share a first token, are skipped **with a counted reason** — never silently
dropped.

## What `bill` computes

Given a reference measurement and a current one:

- **base-positives** — items the reference got right (margin > 0)
- **casualties** — base-positives whose current margin < 0, listed by id
  with per-item **deficit = max(0, −margin)** nats. Strict inequalities: an
  exactly-zero margin is neither correct nor a casualty and carries zero
  deficit.
- **deficit sum** — total nats below zero across base-positives
- **open-set rank-1 losses** — items at rank 1 in the reference that are
  rank > 1 now (items missing from the current open-set read are not
  counted)
- **field stats** — mean and p95 of |Δmargin| over shared items

## Usage

```bash
# measure the base and an edited checkpoint (torch + transformer_lens):
python panel_bill.py measure --panel panel.json --arm base \
    --model Qwen/Qwen2.5-3B-Instruct --out meas_base.json
python panel_bill.py measure --panel panel.json --arm edited \
    --model Qwen/Qwen2.5-3B-Instruct --checkpoint edited.pt --out meas_edited.json

# bill the damage:
python panel_bill.py bill --base meas_base.json --current meas_edited.json --out bill.json
```

`--panel` may contain full items, or id-rows resolved against a separate
`--bank` file (the source program's layout). A `.sha256` sidecar is written
next to each measurement; the panel file's sha256 is recorded inside it.

Custom backends: pass `--adapter module:callable` returning a `ModelAdapter`
(four methods; see the docstring — a pure-python `MockModel` reference
implementation is included).

### Loading large models: the `bf16_np` option

`--loader bf16_np` uses `from_pretrained_no_processing` + `bfloat16`.
**Use it for large models.** The naive fp32 `from_pretrained` path
materializes roughly 4x the weight bytes during load and has OOM-killed a
host in the source program — a real incident, not a hypothetical. Note that
fp32 and bf16_np measurements are *not* interchangeable: compare arms only
within one loader.

### Checkpoint safety

Checkpoints load with `strict=False` but **fail loud on unexpected keys**:
with the return value discarded, `strict=False` can silently score the
pristine base model instead of your edit (a documented finding in the source
program).

## Caveats — read before citing

- **Closed-set numbers are lower bounds.** The forced-choice margin sees
  only the candidate logits. Always read the open-set channel beside it.
- **Compare margins in nats, not flip rates.** A flip count confounds damage
  with headroom: high-margin items absorb the same Δ without flipping. The
  deficit/field numbers are the damage measure; casualty counts are the
  headline, not the evidence.
- **Comparability requires identity of instrument**: same panel, same
  tokenizer, same loader/dtype, same backend. The margin arithmetic here is
  the source instrument's verbatim so that holds within this tool, but do
  not compare against numbers produced by a different scorer.
- Adapter-path log-prob sums accumulate in python float64; the source
  program summed in torch fp32 (differences ~1e-6 nats, verdict-neutral).
- Determinism: greedy, no sampling; on GPU pin your backend's determinism
  flags (the source program set `CUBLAS_WORKSPACE_CONFIG=:4096:8`) if you
  need bit-stable reruns.

## Selftest

```bash
python selftest.py
```

Pure python, no downloads, no torch/numpy: hand-computed margin and open-set
arithmetic through `measure`, a damaged-mock round trip through `bill`, and
synthetic measurement jsons covering the edge cases (exactly-zero margins,
one-sided items, non-rank-1 baselines).

Chains with the sibling `quant-bill` tool: quantize a checkpoint there, then
`measure` + `bill` it here.

No telemetry. MIT license.
