# glassloop-tools

Self-contained instruments from an **auditable continual learning**
research program: teaching facts and skills into LLM weights with the
collateral damage priced in nats, casualty-directed repair, and verified
revert — treating weight updates the way CI/CD treats code changes
(measure, gate, roll back). Every tool in this repo was built and
battle-tested inside that program before being packaged here. Each tool is
a standalone directory: Python 3.10+, argparse CLIs, heavy dependencies
imported lazily, and a selftest that runs without downloading any model.

## Start here: the auditable continual learning result, and how to check it yourself

If you arrived from the paper *Auditable Continual Learning: Measured Transactions for
Sequential Weight Edits* ([10.5281/zenodo.21881995](https://doi.org/10.5281/zenodo.21881995)),
this is the door.

- **The claim, in one sentence.** A weight change can be priced (its damage measured in nats
  on 641 held facts and on a held coding capability), gated by a pre-registered rule,
  receipted, and reverted — and new skills can be installed under that contract. The paper
  shows it for facts on Qwen2.5-3B; the program has since shown two invented skills installed
  inside the same screens at 3B, 7B and 14B (Qwen2.5) and on Llama-3.1-8B, one seed each.
- **Evidence you can verify on a laptop, with no GPU and no trust in us:** [`audit-kit/`](audit-kit/)
  (24 MB). `cd audit-kit && python verify.py --static-only` checks every file against its hash,
  re-derives every bill and crossing count from the stored log-probabilities, and rebuilds the
  result tables (618 checks, about one second, any OS). `python verify.py` additionally
  re-executes the saved HumanEval+ programs through evalplus; `--regrade` re-executes the
  stored skill programs through their graders (Linux, macOS or WSL). The evidence root digest
  is the sha256 of `audit-kit/evidence/SHA256SUMS`:
  `fa270b6786e2c6c6d08c3865168ba2631c3643e897e7dc4571303e31f42f9ee0`; `verify.py` prints it at
  step 0 so a run can be bound to exact files.
- **What the kit establishes and what it does not** is stated in
  [`audit-kit/README.md`](audit-kit/README.md) under "What the checks do and do not establish".
  The kit never runs a model; provenance rests on receipts that carry weights hashes, software
  stack fields and timestamps, cross-checked against each other. (The same pattern, stored
  certificates re-checked by a portable script plus a hashed manifest binding the heavy
  inputs, with the line between "computer-assisted" and "formalized" stated, appears
  independently in the September 2026 deletion-channel capacity certificate at
  github.com/anadim/binary-deletion-channel-capacity; no priority is claimed either way.) The kit's tables are the 3B
  results; the 7B, 14B and Llama-8B ladders are recorded in the program's channel and enter the
  kit in its next version.
- **The paper's own verification code and pre-registration record** (FP-471 to FP-516, seals,
  drivers, reports, gate policies, ledger slice): [10.5281/zenodo.21881977](https://doi.org/10.5281/zenodo.21881977).
- **Not claimed:** frontier scale; a second seed; a mechanism for the coding cost that some
  model families pay under this writer; anything beyond the numbers in the kit's tables and
  the disclosures beside them.

## Watch it work (live demos — three pages, in reading order)

1. **[Watch it grow](https://nryoung-research.github.io/glassloop-tools/demo/loop-watch-it-grow.html)** — the original: five 2026 facts learned, priced, healed, and hash-verified back to the starting state.
2. **[The First Accept](https://nryoung-research.github.io/glassloop-tools/demo/the-first-accept.html)** — a sealed campaign's first contract-admitted weight transaction, with its disclosed limits.
3. **[The Full Cycle](https://nryoung-research.github.io/glassloop-tools/demo/loop-full-cycle.html)** — teach, repair to a clean panel, press to int8, teach the compressed master, then unwind everything byte-exact.

**https://nryoung-research.github.io/glassloop-tools/demo/loop-watch-it-grow.html**

One unbroken sitting: a 3B model whose knowledge ends in 2023 learns five
things from 2026, pays a measured price for each, heals every named
casualty back to zero, survives a simulated int8 press, keeps learning on
the fp32 master — and is then hash-verified back to its exact starting
state. The page is generated mechanically from the run's signed state
ledger; every quoted model answer passed a byte-exact quote audit; the
served page bytes hash to the independently audited value. Verify all of
it yourself from [`demo/`](demo/) (state ledger, console log, generator,
quote-gate script, instructions).

## Tool census

| Tool | What it does | Status |
|---|---|---|
| [`tools/meter-check`](tools/meter-check/) | Adversarial checks on a measurement meter itself — prove the instrument cannot be gamed before trusting numbers from it | packaged from the source program; verify via its selftest |
| [`tools/panel-bill`](tools/panel-bill/) | Price the collateral damage of a weight edit across a fixed probe panel, in nats | packaged from the source program; verify via its selftest |
| [`tools/quant-bill`](tools/quant-bill/) | Price the damage of quantization with the same panel-bill units, so precision changes and weight edits are comparable | packaged from the source program; verify via its selftest |
| [`tools/skillbench`](tools/skillbench/) | Mint a skill benchmark no model can have memorized: invented library, nickname-mapped tasks, execution-only scoring, deranged control | selftests passing in this packaging (50/50 scorer cases; all lesson checks) |
| [`tools/freshknow`](tools/freshknow/) | Measure a frozen model's knowledge decay by effective year and its fresh-knowledge floor; keyed-hash teach/hidden split over FreshQA | selftest passing in this packaging (178 fixture checks; 296 against the real CSV) |
| [`audit-kit`](audit-kit/) | Verify the program's priced, receipted weight changes from hashed evidence: bills recomputed from stored log-probabilities, tables rebuilt, saved programs re-executed | `verify.py --static-only` 618 checks / 0 failures in this packaging; two cold tests by strangers-as-agents recorded in the kit README |

Sibling repo: **weight-genealogy** — weight-space lineage detection
(which released checkpoints descend from which), maintained separately.

## Limitations (read before citing)

- **Provenance is narrow.** Many defaults, thresholds, and design choices
  in these tools were validated on a single model family at one or two
  sizes (7B-class instruct models, mostly one vendor). The instruments
  are general; the *default numbers* inherit that provenance. Recalibrate
  on your own model before treating any default as meaningful.
- **Tools are instruments, not verdicts.** A passing gate means "this
  specific check, on this specific probe set, did not fire" — it is a
  lower bound on problems, never proof of absence. Closed-set and
  fixed-panel measurements systematically under-report damage.
- **Cite the numbers, not the vibes.** Every claim these tools support
  should be quoted with its measured value, its population, and its
  instrument settings (loader, seeds, split hash, and the device and
  software stack the model was trained and read on: Paper 12's September
  2026 erratum measured a 3.6–4.0× change in edit cost from the training
  device alone, at matched seed, data order and endpoint loss). If a
  statement about a model can't be traced to a logged number from a
  selftested meter, it is not supported by this repo.

## Verify before trust

Each tool ships a selftest designed to prove the instrument cannot lie in
either direction (correct inputs must pass, plausible-wrong inputs must
fail with a logged reason). Run them before believing any output:

```
python tools/skillbench/selftest.py           # scorer: 50 adversarial cases
python tools/skillbench/selftest_lessons.py   # after gen_tasks + gen_lessons: leak scan + re-execution
python tools/freshknow/fq_selftest.py         # meter + split gates, synthetic fixture, no download
python tools/meter-check/...                  # see tools/meter-check/README.md
python tools/panel-bill/...                   # see tools/panel-bill/README.md
python tools/quant-bill/...                   # see tools/quant-bill/README.md
```

A selftest that fails on your machine is a result: do not proceed to model
measurements until it passes.

## The research program behind the tools

Every instrument here — and the three demo pages above — was built inside a 32-paper
research program on the metrology of language-model internals (papers numbered 1–33;
13 unused), in two arcs: the **state arc** (papers 1–28 — the part of a transformer's
residual stream its own output layer cannot read), and the **change arc** (papers 29–33 —
weight updates as measured, auditable, revertible transactions), with a campaign-record
paper on contract-monotone continual learning in preparation.

- **[PAPERS.md](PAPERS.md)** — the map: what each paper shows and where to start by interest.
- **[CORPUS.md](CORPUS.md)** — the canonical DOI register, every entry verified against the
  live Zenodo record.

Every result is pre-registered and sealed before the run; negative results are banked at the
same weight as positive ones. Author: Nathan Ryan Young, ORCID
[0009-0003-2840-7726](https://orcid.org/0009-0003-2840-7726).

## Corrections (2026-09-28)

Ten papers of the state arc carry September 2026 errata sections. As of 2026-09-28 the Zenodo
versions that carry them are not yet live; the links below are concept DOIs, which resolve to
each paper's newest version. None of the tools, the audit kit or the demo pages rests on a
corrected result; [PAPERS.md](PAPERS.md) is corrected where a summary did. Most of these errata
apply one standard: a claim that the complement of a readable frame (the "dark" part, the
"halo") carries energy, load or decodable content is scored against a random subspace of the
same rank, with the rank printed.

- **Paper 1**, [*The Rectified Frame Potential*](https://doi.org/10.5281/zenodo.21126359): A(9,3) ≤ 5 by an explicit construction (the annealing value 6 is withdrawn); the harmonic-sign corollaries are classical in their integer-power content and now attributed; the two-point bound at m = 7, d = 3 saturates at 0.5783, so the remark that a higher degree improves it is withdrawn (a three-point certificate gives 0.7001); the pentagon is not a universal building block.
- **Paper 8**, [*Model Surgery*](https://doi.org/10.5281/zenodo.21214477): "healed to below healthy" was a training-budget confound. Against the equally trained sham the lesion healed to within 0.010 nats; nothing healed past health.
- **Paper 12**, [*The Price of Dust*](https://doi.org/10.5281/zenodo.21234072): predicting the loss change of a given edit from Taylor terms is prior art (Molchanov et al., 2019); the output-head fit of 1.0000 follows from Taylor's theorem and checks the machinery, not the certificate's predictive value. The cross-device edit-cost gap, previously n = 1, replicated in three seeds (3.6–4.0×): device ≫ software stack ≫ seed.
- **Paper 14**, [*What the Lens Deletes*](https://doi.org/10.5281/zenodo.21274591); **Paper 18**, [*The Dark Atlas*](https://doi.org/10.5281/zenodo.21349363); **Paper 19**, [*The Work Meter*](https://doi.org/10.5281/zenodo.21367316): the size-matched control is owed, and the affected comparisons are read as rank effects until it exists. The Qwen readable frame is one direction in 3,584, so Qwen "dark" rows describe the residual stream. Paper 18's contrast with the Anthropic workspace result is withdrawn pending a J-space-matched control; Paper 19's autograft is a diagnosis that needs the gold answer, not a usable repair.
- **Paper 15**, [*Weighing and Reading the Dark Matter*](https://doi.org/10.5281/zenodo.21274615): the control, run (pythia-410m; readable rank 209–437 of 1,024). The "dark" energy share is the rank share; the readable frame is load-bearing (0.333 against 0.252 ± 0.021 nats) and readable (+0.07) beyond a random frame of its rank; the halo is not denser than a random complement and reads the copied token no better (0.831 against 0.887). "Darkness is a property of the instrument, not the information" is withdrawn.
- **Paper 17**, [*The Coupling Code*](https://doi.org/10.5281/zenodo.21349350): below a matched random subspace in three of the four linear frames tested, not every one; the deep-anatomical consumer frame sits above the floor (0.401 against 0.314).
- **Paper 23**, [*The Implanted Thought*](https://doi.org/10.5281/zenodo.21367358): gates G1 and G2 are retracted. A random scramble of the same size does the same damage as the dark scramble (Qwen retention 0.656 against 0.625; Mistral 0.909 against 0.909); swapping ~3,300 dimensions from another item, dark or random, moves the chain to the donor in the same 22 of 32 items, and 256-dimensional swaps, readout frame or random, change nothing. The variable is the size of the component, not the frame. The CONFESS/UNFAITHFUL count is built in under greedy decoding, the "~90% dark" rider is withdrawn, and concept injection with self-report (Lindsey, 2026) is prior art.
- **Paper 25**, [*The Doors of the Two-Hop*](https://doi.org/10.5281/zenodo.21881892): the "collision" with Gurnee et al. (2026) inherited Paper 18's withdrawn contrast; the reconciliation is scoped to "consistent with it, pending the matched control". The two doors, the family-compiled selectivity and the write-gain law on the paper's own frame stand.
- **Research ledger (not a paper):** the reading of FP-382b as "identifiability is a phase" is retracted. Random Gaussian dictionaries show the same step at the same widths (Wendel's theorem on the atom count). Paper 27 does not assert it and is unchanged.

## License and author

MIT License (see [LICENSE](LICENSE)). Author: Nathan Ryan Young.
