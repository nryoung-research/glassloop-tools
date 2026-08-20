# skillbench — build a skill benchmark no model can have memorized

Mint a fact-and-skill benchmark from an **invented library with novel
semantics**, so that base-model performance is zero by construction and any
score above zero is attributable to what *you* taught the model. Extracted
from a continual-learning research program where it was used to measure
whether procedural skills (not just facts) can be written into LLM weights.

## Why an invented library

If you benchmark on real APIs or public puzzles, you can never separate
"the model learned this from my training intervention" from "the model saw
this in pretraining." `quirklib.py` sidesteps the problem: ten small
functions (`snerp`, `gribble`, `quangle`, ...) with made-up names and
deliberately arbitrary — but exact, deterministic, pure — semantics over
ints and lists of ints. The semantics are canonical *as written in the
docstrings*; they are intentionally not the "natural" operation a model
would guess. A model that has never been taught quirklib scores zero. That
is the point.

**Fork it.** `quirklib.py` is ~150 lines. Replace the function bodies and
docstrings with your own invented semantics (keep the docstring format:
prose semantics, a unique `Nickname: "the ..."` line, an `Example:` block)
and you have minted a fresh, never-memorized benchmark. Do this especially
if you suspect this repo's default instance has leaked into training
corpora — the design makes re-minting cheap.

## The two tiers, and what nicknames buy you

`gen_tasks.py` emits two task tiers (40 train + 40 holdout each, from
disjoint seeds, with construction-level dedup):

- **TIER-F (facts):** recall the exact value of single quirklib calls.
  The grader **blocks** `import quirklib` for these tasks, so the model
  must answer from internalized semantics — it cannot look anything up.
- **TIER-S (skills):** compose 1–3 quirklib functions plus plain-language
  plumbing steps. Here `import quirklib` is **allowed**, but the task
  instructions refer to functions **only by their docstring nicknames**
  ("the parity weave", "the sieve stamp"), never by real name. A model that
  can call the library but does not know *which function each nickname
  denotes* cannot solve the task. This makes TIER-S a test of **mapping
  knowledge** (nickname → function → correct use), not retrieval or
  pattern-matching on identifiers.

Task instructions never contain reference solutions or specs;
`solutions/*.jsonl` is grader-side only and must never reach the model.

## Execution-only scoring

`score_exec.py` scores a raw model response with **no string match against
reference code and no LLM judge** anywhere:

1. **Extraction:** fenced ```` ```python ```` block, bare fence, or a
   largest-parseable-def-block heuristic fallback (the path that fired is
   logged per item).
2. **Sandboxed execution:** every unit test runs in its own `python -I`
   subprocess with a hard timeout (Windows-safe, no signals). The child
   builds quirklib from source with docstrings stripped, registers it
   (TIER-S) or blocks its import (TIER-F), then execs the model code under
   restricted builtins (no `open`/`eval`/`exec`/`input`; imports limited to
   a stdlib allowlist) and calls the entry point.
3. **Strict typed equality:** `bool` is not `int`, `tuple` is not `list`.

Score = tests passed / tests total, with a per-item JSON log (extraction
path, per-test pass/fail, exception text).

## Lessons and the deranged control

`gen_lessons.py` turns the train split into a teaching corpus
(`lessons_sft.jsonl`: API semantics/nickname/worked-example records plus
train tasks with reference solutions, chat SFT format) and an editor spec.
It never reads holdout artifacts; worked examples are drawn only from
train-side cases, and even the docstring `Example:` blocks are excluded
from lesson payloads because they live in the same small argument space as
holdout draws and cannot be deduped against them.

`gen_deranged.py` emits the **semantic control**: a cyclic derangement of
the name→function mapping (every occurrence of function name *f_i* renamed
to *f_(i+1) mod 10*, word-boundary safe, in both user and assistant text).
The deranged corpus is token/dose-matched by construction and teaches a
coherent but *wrong* mapping. If a model trained on it still scores on the
real holdout, your meter is passing on something other than the taught
mapping — stop and debug the meter.

## The calibration protocol (run this BEFORE any training claim)

Before attributing any post-training score to your intervention, run both
calibration arms:

1. **Base-zero arm:** evaluate the untouched model on the holdout. It must
   score ~0. If it does not, your library instance has leaked or your
   semantics are guessable — fork quirklib and re-mint.
2. **In-context-ceiling arm:** evaluate the same model with the full
   quirklib source (docstrings included) in the prompt. This is the
   skill-ceiling of the model given perfect information; your
   weights-teaching result lives between the two arms.

In the source program this protocol caught a would-be false negative on a
different battery: a low post-training score that was about to be read as
a "skills wall" turned out to sit at the model's own in-context ceiling —
the model could not do the tasks even with the library in front of it.
Without the ceiling arm that would have been misattributed to the teaching
method. Do not skip it.

## Usage

```
python gen_tasks.py                    # tasks/ + solutions/ + manifest.json
python gen_lessons.py                  # lessons_sft.jsonl + specs/
python gen_deranged.py                 # lessons_sft_deranged.jsonl
python score_exec.py --task t.json --response r.txt   # score one response
python selftest.py                     # prove the scorer cannot lie
python selftest_lessons.py             # prove the lesson payloads are clean
```

All generation is deterministic (seeded `random.Random`, no wall clock);
`--seed-train/--seed-holdout` mint fresh task instances from the same
library. Stdlib only — no third-party dependencies. Python 3.10+.

Note: generated artifacts (`tasks/`, `solutions/`, lesson files) are
deliberately not shipped in this repo — publishing a solved instance is how
benchmarks get memorized. Generate your own, ideally from your own fork of
`quirklib.py` with your own seeds.

## Selftests (run them yourself)

`selftest.py` proves the scorer cannot lie in either direction: for 10
sampled tasks, a correct reference solution wrapped in three response
formats (fenced / prose+bare fence / messy no-fence) must score 100% via
every extraction path, and a plausible wrong solution (TIER-F: an
import-cheat that the import block must defeat; TIER-S: the reference
pipeline with a subtle extra step) plus a garbage response must score 0%
with a logged failure reason and no scorer crash.

Verified output of this exact copy (`python selftest.py`, tail):

```
[PASS] S-train-007    fenced   score=1.00 path=fenced_python
[PASS] S-train-007    prose    score=1.00 path=fenced_bare
[PASS] S-train-007    messy    score=1.00 path=heuristic_defblock
[PASS] S-train-007    wrong    score=0.00 path=fenced_python reason='result mismatch: got [9, -5, -3, -8, 2, 7] expected [9, -5, -3, -8, 2]'
[PASS] S-train-007    garbage  score=0.00 path=none reason='no code extracted: bare fence present but contents not parseable; no par'
------------------------------------------------------------
selftest: 50/50 cases passed
```

`selftest_lessons.py` (after `gen_tasks.py` + `gen_lessons.py`) scans the
emitted lesson files for holdout leakage (928 needles: ids, expressions,
instruction blocks, test JSON), re-executes every embedded reference
solution through the real scorer, and re-verifies the spec. Verified
output of this exact copy:

```
[PASS] files present  110 lesson records, specs: ['quirk_full.json']
[PASS] leak scan: no holdout string in lessons/specs  928 needles scanned
[PASS] task record count  80 task records
[PASS] all embedded references pass their tests via score_exec  80 scored
[PASS] spec task passages embed identical reference code  80 passages
[PASS] spec counts
[PASS] items/probes expression-disjoint
[PASS] all 40 spec evaluations re-verify against real quirklib
[PASS] dose fields verbatim from template
[PASS] templates use {n}/{t} placeholders (gl_edit2m .format(n=,t=))
[PASS] fact_text_perm is a permutation of fact_texts
[PASS] every item rendered in an 'evaluates to' passage
[PASS] no holdout-probe evaluation in any non-task passage
[PASS] code-context identifier fraction >= 0.5  70/110 = 0.636
------------------------------------------------------------
selftest_lessons: all checks passed
```

## Provenance notes

These files are faithful copies from the source research program.
Docstrings retain internal experiment-registry references (FP-558,
METER-REDTEAM D19, a template-spec path, `gl_edit2m` — the program's
weight-editing harness, not included here); they document where the
defaults came from and do not affect execution. The editor spec emitted by
`gen_lessons.py` (`specs/quirk_full.json`) is only useful if you have your
own editing harness; the SFT corpus (`lessons_sft.jsonl`) is
harness-agnostic chat format.
