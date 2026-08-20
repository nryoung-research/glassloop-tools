# freshknow — measure your model's knowledge decay and fresh-knowledge floor

Score a frozen LLM against time-stamped real-world questions and read out
its **knowledge decay curve**: how accuracy falls as facts get newer than
the model's training data. Also provides a **keyed-hash teach/hidden
split** so you can teach half the facts into the weights and measure the
other half as an untouched control. Extracted from a continual-learning
research program where it established the fresh-knowledge floor before any
weight edits.

## Get the data (not bundled)

This tool ships **no dataset**. Download FreshQA yourself from
[github.com/freshllms/freshqa](https://github.com/freshllms/freshqa) —
it is updated regularly; take the newest dated CSV. The expected format is
the FreshQA distribution layout:

- two junk rows, header on **row 3**
- 20 columns: `id, split, question, effective_year, next_review,
  false_premise, num_hops, fact_type, source, answer_0..answer_9, note`
- quoted multiline fields present (parsed with Python's `csv` module)

## The keyed-hash split (recomputable by anyone)

`fq_split.py` partitions the CSV with zero discretion. The exact rule:

```
key = sha256(raw_csv_file_bytes + b"fp556-split-v1")        # 32-byte digest
for each HEADLINE item:
    h = sha256(key + item_id_utf8)     # id = exact string from the 'id' column
    first byte of h EVEN  -> TEACH-ELIGIBLE
    first byte of h ODD   -> HIDDEN
```

Populations:

- **HEADLINE** = `fact_type == 'fast-changing'` and
  `false_premise == 'FALSE'` — split teach/hidden by the rule above.
- **SIDE-CHANNEL** = fast-changing items with a false premise — scored and
  printed, never headline.

Because the key is derived from the CSV bytes themselves, anyone holding
the same CSV recomputes the identical split; nobody (including you) can
cherry-pick which facts land in the hidden half. The output
`fq_split.json` records `csv_sha256` and a `key_hash_prefix` so runs are
checkable against each other.

```
python fq_split.py path/to/freshqa.csv [--out fq_split.json]
```

## The deterministic dual-channel scorer

`fq_score.py` is a pure-stdlib scorer — no LLM judge, no network, fully
deterministic:

- **CONTAINS channel:** normalized containment of any non-empty
  `answer_0..answer_9` in the full response. Normalization: casefold,
  punctuation→space (so "Jean-Pierre" and "Jean Pierre" agree), collapsed
  whitespace, one leading article stripped from the answer side only.
  Containment is **token-boundary-safe** (contiguous token subsequence): a
  bare "2" can never match inside "2026".
- **STRICT channel:** the same test restricted to the response's first
  sentence — a proxy for "answered directly" vs "mentioned it somewhere".
- **Failure taxonomy:** every non-hit is classified `EMPTY`, `REFUSAL`
  (deterministic pattern family: "I don't know" / "I cannot" / "as of my
  knowledge...", etc.), or `WRONG`. Separating refusal from confabulation
  matters: a model that says "I don't know" about 2026 is in a very
  different state from one that asserts a stale answer.

Known deterministic limitation (accepted, documented): abbreviations
ending "&lt;letter&gt;. " can shorten the first sentence for the STRICT
channel.

## The decay readout

`fq_baseline.py` runs the closed-book baseline: greedy zero-shot
generation over all three populations, scored by `fq_score`, aggregated
per population **and per `effective_year`**. The by-year table is the
headline readout. In the source program, a 7B instruct model (frozen
weights, 2024-era training data) scored **0.50 CONTAINS on facts effective
in 2022, decaying to 0.013 on facts effective in 2026** — a frozen model's
knowledge of the world falls off a cliff at its data horizon. That curve
is the baseline any teaching/editing intervention has to beat, and the
hidden half of the split is the control that keeps the comparison honest.

```
python fq_split.py freshqa_2026-08-18.csv --out fq_split.json
python fq_baseline.py --split fq_split.json --store fq_runs
python fq_baseline.py --split fq_split.json --ckpt edited.pt --label E1
```

Requires `torch` + `transformer_lens` (imported lazily — the selftest and
splitter run without them). Defaults target `Qwen/Qwen2.5-7B-Instruct` on
CUDA; `--model`/`--eos-token`/`--device` adapt it to other setups (the
default end-of-turn token `<|im_end|>` is Qwen's chat template — change it
for other families). Runs are resumable (`.partial` file); exact prompts
and generations are logged per item with prompt hashes.

## The bf16_np loader (read this before loading a 7B)

`--loader bf16_np` (default) uses transformer_lens
`from_pretrained_no_processing` with `bfloat16`. The naive alternative
(`--loader fp32`, plain `from_pretrained` in fp32) spikes to roughly **4x
model size in host RAM during weight processing**; in the source program
it OOM-killed a 7B load at 113GB resident on a 128GB box (kernel OOM
logs, 2026-08-18). Use `fp32` only for small models on machines with
plenty of headroom, and keep the loader **identical across every arm you
compare** — it is an instrument choice, and it is recorded in the report
for exactly that reason.

## Selftest (no model, no download, CPU-only)

`fq_selftest.py` synthesizes its own tiny FreshQA-format fixture CSV (same
layout: junk rows, row-3 header, 20 columns, quoted multiline field) and
runs five gates against the real scorer and splitter:

- **(a) ORACLE** — each true answer, wrapped in three prose forms, must
  score CONTAINS=1 (the meter can find a correct answer however phrased);
- **(b) WRONG-ENTITY** — a response built from a different item's answer
  must score 0/0 (the meter cannot be fooled by fluent wrong answers);
- **(c) TAXONOMY** — empty → EMPTY, refusal phrasings → REFUSAL;
- **(d) WORD-BOUNDARY** — superstring numbers must not match ("2" never
  hits inside "2026");
- **(e) SPLIT** — determinism (two builds byte-identical), teach/hidden
  disjoint, every fast-changing item in exactly one bucket.

Verified output of this exact copy (`python fq_selftest.py`):

```
synthesized fixture: C:\...\fq_fixture_b5mpeu3p\fq_fixture.csv
headline population n=40 (teach 19 / hidden 21)
sampled 25 items, seed 556: ['fx023', 'fx025', 'fx033', 'fx021', 'fx029', 'fx038', 'fx003', 'fx002', 'fx015', 'fx020', 'fx001', 'fx013', 'fx034', 'fx005', 'fx010', 'fx004', 'fx006', 'fx011', 'fx032', 'fx017', 'fx000', 'fx014', 'fx026', 'fx008', 'fx024']
[a] oracle prose wrappings
[b] wrong-entity distractors
[c] failure taxonomy
[d] word-boundary / superstring numbers
  numeric <=2-token answers checked: 8
[e] split determinism and coverage
checks run: 178, failures: 0
FQ-SELFTEST-PASS
```

Once you have downloaded the real CSV, run the same gates against it:
`python fq_selftest.py --csv freshqa_YYYY-MM-DD.csv` (against the source
program's real CSV this executes the original 296 checks, all passing).
Run the selftest to PASS **before** believing any model number out of
`fq_baseline.py`.

## Provenance notes

Faithful copies from the source program with three deliberate changes:
`fq_split.py` takes the CSV path as a CLI argument instead of assuming a
bundled file; `fq_baseline.py` takes `--split/--store/--model/--device/
--eos-token` arguments instead of hardcoded internal paths (and imports
torch lazily); `fq_selftest.py` synthesizes its fixture instead of
requiring the real CSV. The scorer `fq_score.py` is byte-identical to the
sealed original. The split salt string `fp556-split-v1` is kept verbatim —
changing it would silently change everyone's split.
