# Replicate this work — start here

Every claim in this repo is designed to be checked by a stranger.
Pick your tier by hardware. Report anything that fails — a failed
replication report is worth more to us than a compliment.

## Tier 0 — no GPU, ~10 minutes: verify the instruments

Every tool ships a selftest that proves its arithmetic on synthetic
data with known answers. Run them all:

    cd tools/meter-check  && python selftest.py
    cd tools/panel-bill   && python selftest.py
    cd tools/quant-bill   && python selftest.py
    cd tools/skillbench   && python gen_tasks.py && python selftest.py
    cd tools/freshknow    && python fq_selftest.py

Expected: every suite prints PASS (293 checks total). If any check
fails on your machine, that is a finding — please open an issue with
your OS/Python/output.

## Tier 1 — any GPU with ~8 GB, ~1 hour: measure a real model

- **Knowledge decay curve** (freshknow): download the FreshQA sheet
  (see tools/freshknow/README.md), run the split + baseline on any
  chat model you can load. You should see the signature: strong on
  older facts, near-zero on the newest year, mostly confident wrong
  answers rather than refusals.
- **Quantization bill** (quant-bill + panel-bill): quantize any small
  model to int8, bill the damage against its own baseline. Every
  casualty comes out named.

## Tier 2 — ~16 GB GPU, an afternoon: replicate the skills result

The headline claim: a skill no model has ever seen can be installed by
an audited weight-write, verified by executing the model's code on
unseen problems. skillbench contains everything needed:

1. `python gen_tasks.py` — mints the invented-library benchmark
   (or fork quirklib.py and mint your OWN library nobody has seen).
2. Baseline the untaught model on `tasks/holdout_*.jsonl` — expect ~0
   on the skill tier (score with score_exec, execution-only).
3. Optional ceiling check: prepend the library docs to the prompt —
   this measures whether your model CAN do the tasks with the book
   open (our 3B: 0.675; 14B: 0.975).
4. `python gen_lessons.py` then train the lesson corpus into the model
   (any LoRA/finetune stack; we used rank-64 attention+MLP over all
   layers — MLP-only writes carry facts but not skills, which is
   itself a replicable finding).
5. Re-run the holdout. Our result: 0.000 -> 1.000 at 3B.
6. The control that makes it real: `python gen_deranged.py` and train
   a second model on the scrambled corpus (same dose, wrong mapping) —
   it should score ~0 on the real holdout. Ours did: 0.000 at
   identical training loss.

## Tier 3 — the genealogy tool (sibling repo)

`weight-genealogy` detects model family relationships from weight
deltas alone, with calibrated verdicts. The atlas contains five
verified rows; rerun any of them, or point it at model pairs we have
never seen and tell us what it says.

## What "success" and "failure" both look like

Success: your numbers land within the tolerances each tool's README
states. Failure: they don't — in which case you have either found an
instrument defect (we have found 17 of our own; the catalog is part of
this project) or a real limit of the claims. Both are wanted. The one
thing this project asks of replicators: report what you MEASURED, not
what you expected.

## AI-assisted replication is welcome

This work was itself built by AI hands under human arbitration, so
using an AI to run your replication is fully in the spirit of the
project. One rule keeps it honest, and it applies to humans and AIs
equally: **attach the raw outputs** — the actual console prints,
report JSONs, and logs from YOUR machine — not a summary of them. An
AI can accidentally "replicate" by reading the docs and reporting the
expected numbers as if it ran them (a failure mode we have caught in
our own instruments more than once). Artifacts, not narratives.
Disclose AI assistance in your report; it costs nothing and is the
norm here.
