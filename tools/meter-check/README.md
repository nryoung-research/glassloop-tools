# meter-check — is your eval's scorer lying to you?

Three-channel corrected-meter rescore for language-model evals, ported from
the Glass Loop program's FP-555 full population rescore. It answers one
question: **when your scorer says "fail", did the model actually not know the
answer — or did your meter just not look?**

## The three channels

For every item `{prompt, answer, answer_set}` the same model is scored three
ways:

| channel | definition | character |
|---|---|---|
| `FIRST-TOKEN` | argmax of the next-token distribution after the prompt equals the first token of `" " + answer` | legacy channel; known artifact-prone |
| `FC` | forced choice by **full-answer** log-prob: sum of per-token log-probs of each candidate continuation; hit iff the true answer scores highest over `answer_set` | closed-set, robust to first-token quirks |
| `OPEN` | greedy decode of N tokens (default 12); hit iff the answer string appears, case-insensitive, in the generation | open-ended containment |

**The artifact signature** is `FIRST-TOKEN = fail` while `FC` and/or `OPEN` =
pass: the model knows the answer but the first-token meter refuses it (a
preamble token, a tokenizer boundary, a chat-template tic). In the source
program, applying this corrected meter to a sealed population of historical
refused verdicts flipped **193 of 227** cells. If your eval pipeline scores by
first-token argmax, run this before trusting a single refusal.

## Usage

```bash
# with any model, via a small adapter you write (see protocol below):
python meter_check.py --items items.jsonl --adapter mymod:make_adapter --out report.json

# with a transformer_lens model (torch + transformer_lens required):
python meter_check.py --items items.jsonl --model Qwen/Qwen2.5-3B-Instruct \
    --checkpoint edited.pt --out report.json
```

`items.jsonl`: one JSON object per line, `{"prompt": ..., "answer": ...,
"answer_set": [...], "id": optional}`. The answer must be a member of the
answer_set (the tool fails loud otherwise). In the source program the
answer_set was the sorted set of all lesson targets.

The report contains per-item rows (all three hits, the FC scores, the actual
generation) and a summary: per-channel pass counts, disagreement ids, and the
artifact-signature list.

## Adapter protocol

Five methods; see `ModelAdapter` in `meter_check.py`:

- `to_tokens(text) -> list[int]` — full-context tokenization (BOS included
  here if your model uses one)
- `to_string(ids) -> str`
- `first_token_id(text) -> int` — first token of `" " + text`, tokenized
  without special tokens
- `sequence_logprobs(ids) -> list[float]` — element `t` = log-prob of
  `ids[t+1]` given `ids[:t+1]`
- `next_logprobs(ids)` — full-vocab log-softmax at the last position

A pure-python `MockModel` (hand-built logits) is included and doubles as the
reference implementation; a `TransformerLensAdapter` (lazy torch import) is
included for real models.

## Caveats — read before citing

- **FC is closed-set.** It only sees the candidates you gave it. A model can
  fail FC and still "know" the answer under a different frame; a model can
  pass FC by eliminating distractors. FC verdicts are relative to the
  answer_set.
- **OPEN containment can false-positive**: if the answer string occurs
  incidentally in the first N greedy tokens (short answers, common words),
  OPEN passes. Prefer distinctive answer strings; inspect the `generation`
  field in the report.
- **The corrected channels are not oracles.** meter-check tells you the
  channels *disagree* and which pattern the disagreement has; the 193/227
  flip figure is from one program's population (one model family, one task
  family) and is not a universal rate.
- **Tokenizer-dependent.** All three channels depend on your adapter's
  tokenization conventions (BOS, leading spaces). Keep them identical between
  runs you compare.
- Determinism: greedy decode, no sampling. On GPU, kernel nondeterminism can
  wiggle low-order logit bits; for bit-stable runs pin your backend's
  determinism flags (the source program set `CUBLAS_WORKSPACE_CONFIG`).
- FC sums are accumulated in python float64 from per-token log-probs; the
  source program summed in torch fp32. Differences are ~1e-6 nats and do not
  change verdicts.

## Selftest

```bash
python selftest.py
```

Pure python, no downloads, no torch/numpy. Hand-built logits prove each
channel's arithmetic (including the multi-token FC sum), and a synthetic
artifact case (model emits a preamble token before the correct answer) is
caught by the summary.

No telemetry. MIT license.
