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
  software stack each step ran on: in Paper 12's September 2026 erratum
  the same seeds gave a 3.6–4.0× different edit cost on two devices'
  train-and-measure pipelines, a gap not yet attributed to the hardware,
  the software or the measurement). If a statement about a model can't be
  traced to a logged number from a selftested meter, it is not supported
  by this repo.

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

Sixteen papers of the state arc carry September 2026 errata sections. The corrected versions are
being deposited on Zenodo; until they are live, the links below (concept DOIs, which resolve to
each paper's newest version) show the earlier text. None of the tools, the audit kit or the demo
pages rests on a corrected result; [PAPERS.md](PAPERS.md) is corrected where a summary did. Seven
of these errata (Papers 14, 15, 17, 18, 19, 23, 25) apply one standard: a claim that the
complement of a readable frame (the "dark" part, the "halo") carries energy, load or decodable
content is scored against a random subspace of the same rank, with the rank printed. Where that
control has not been run, darkness, rank and frame selection remain confounded.

- **Paper 1**, [*The Rectified Frame Potential*](https://doi.org/10.5281/zenodo.21126359): A(9,3) ≤ 5, and A(2d+3,d) ≤ 5 for every d ≥ 2, by an explicit construction (the annealing value 6 is withdrawn); the harmonic-sign corollaries are classical in their integer-power content (Bach's derivation covers integer exponents; the extension to real p rests on the paper's own argument); the two-point bound at m = 7, d = 3 saturates at 0.5783, so the remark that a higher degree improves it is withdrawn (a three-point certificate gives 0.7001); the pentagon is not a universal building block (at m = 10, d = 4 the rectified 5-cell, 5/3 in the paper's ordered convention, lies below two orthogonal pentagons; an earlier annealing comparison is withdrawn).
- **Papers 5, 6, 7**, [*Interference Modes in Real Language Models*](https://doi.org/10.5281/zenodo.21137825), [*Scale, Depth and Downstream*](https://doi.org/10.5281/zenodo.21213148), [*The Native MLP Interference Law*](https://doi.org/10.5281/zenodo.21213238): the r ≥ 0.99 shape gate does not discriminate (the measured mean curves of two unrelated models, GPT-2 layer 8 and Pythia-70m layer 3, correlate at r = 0.9930), so shape passes are not counted as evidence and the results rest on median relative errors (one control pair: the gate's general false-pass rate is not estimated, and values near 0.9999 have no evaluated null). In Paper 5, copying an older slice's measured curve predicts the virgin curve as well as or better than the sealed summaries (0.6–1.3%), so transfer to unseen text is not diagnostic; the modes show compression, and their reading as input physics is withdrawn pending a random-mode null. Papers 6 and 7 have not yet been scored against that null. Paper 7's registered count is 17 of 18 layers (GPT-2 MLP layer 9 is a shape-gate miss, r = 0.9803); 18 of 18 holds only on a retrospective error-only reading.
- **Paper 8**, [*Model Surgery*](https://doi.org/10.5281/zenodo.21214477): "healed to below healthy" was a training-budget confound, and the nearest sham trains an untied output head where the patient's is tied, so it is not matched. The claims that the lesion healed past health and that it closed its gap are withdrawn, as is the 84% closure figure for the early block. A tied, equally trained sham, since run (FP-788), narrows the early block's gap by 56% between separate from-start 500- and 2,000-step schedules (0.266 to 0.116 nats, an aggregate cross-entropy gap) and leaves a 0.116-nat residual; that re-run failed its pre-registered 10⁻³ pipeline check (3–6 millinat drift).
- **Papers 9, 10**, [*The Router is a Ruler*](https://doi.org/10.5281/zenodo.21214526), [*One Ruler for Every Selector*](https://doi.org/10.5281/zenodo.21214543): under the isotropic Gaussian input noise used to measure flips, each term of the margin budget is the exact reversal probability of one pair, but the budget (a truncated, clipped sum of correlated pair events) is neither guaranteed to match the measured flips nor guaranteed to be an upper bound. Its agreement across routers, the vocabulary head and retrieval is calibration of that approximation on the models' measured geometry, not evidence of a router mechanism (an untrained random selector passes the same bars). The cascade's survival under transient noise is exactly the product of (1 − q_t), q_t the chance that any challenger overtakes the winner; the implemented product of budgets approximates it, and tightening with horizon is not guaranteed. The front-loaded hazard is withdrawn as a finding: the registered comparison cannot separate it from a stationary mix of easy and hard prompts. Concentration ratios above 10⁷ are set by a division guard. The model evidence is in the rounding and systematic-perturbation experiments; a full-stack INT8 test (Paper 11, FP-129) found the budget over-prices flips by about 2×.
- **Paper 11**, [*Why d-squared, Derived and Controlled*](https://doi.org/10.5281/zenodo.21214570): the weight-space lever recomputes and subtracts each layer's leading accumulated-drift component after every training step, so it shows that repeated drift suppression during training keeps the meter near its floor, not that one shared direction carries a finished model's meter to the exclusion of cross-talk (endpoint mediation is unmeasured); its learning cost is inside the 3% tolerance (+1.55% cross-entropy), not zero; the blind width predictions establish no incremental value over a log-log line through the measured neighbours; the d² exponent follows approximately from the measured growth of the update norm (about d^½), which is measured, not derived.
- **Paper 12**, [*The Price of Dust*](https://doi.org/10.5281/zenodo.21234072): predicting the loss change of a given edit from Taylor terms is prior art (Molchanov et al., 2019); the output-head fit of 1.0000 is an observed small-dose result that checks the machinery, not the certificate's predictive value; "Lemma D2 closes" means agreement on the measured ray, not a bound. The cross-device edit-cost gap recurs in three seeds (3.6–4.0×) but depends on dose (about 4× at full excision, 2.2–2.3× at a quarter of the dust), and it is not a hardware attribution: each run trained, extracted and measured on one device, so a crossed design is owed. Two pre-registered predictions were missed: matched endpoint loss (T1) and a software-stack effect under 5% (T5, 7.4% in one seed). Two GB10s agree bit for bit on the same stack.
- **Paper 14**, [*What the Lens Deletes*](https://doi.org/10.5281/zenodo.21274591); **Paper 18**, [*The Dark Atlas*](https://doi.org/10.5281/zenodo.21349363); **Paper 19**, [*The Work Meter*](https://doi.org/10.5281/zenodo.21367316): the size-matched control is owed; until it exists the affected comparisons carry an unresolved rank and selection confound (they are not shown to be rank effects either). On Qwen the Jacobian-lens readable frame is one direction at each layer 9–17, so its "dark" complement is nearly the whole stream there (Paper 14's Qwen rows are read at layer 20, where the frame rank is owed); Paper 18's two-hop writes use a different, 256-direction frame. Paper 18's contrast with the Anthropic workspace result is withdrawn as not established (different model, task, intervention and metric; their workspace is sparse and input-dependent, ours a fixed frame), and its localisation of the read and the write to the readout-invisible component is not established. Paper 19's autograft needs the gold answer and is not a usable repair; its success alone does not show the answer was present (it flips 5 of 24 never-gold errors against 19 of 24 transient-gold ones).
- **Paper 15**, [*Weighing and Reading the Dark Matter*](https://doi.org/10.5281/zenodo.21274615): the control, run (pythia-410m, 12 prompts; readable rank 209–437 of 1,024). The unread energy share mostly reflects the complement's size (it exceeds its share of dimensions by 0.06–0.12); the readable frame is load-bearing (0.333 against 0.252 ± 0.021 nats) and readable (+0.07) beyond a random frame of its rank; a random same-rank complement reads the copied token better than the halo (0.887 against 0.831). The halo-versus-random density comparison is withdrawn: rerun at a common calibration budget (FP-787), the halo's removal costs 0.886 of a random complement's (95% CI 0.750–1.060), so no difference is detected, which is not equivalence (that run missed its pre-registered 10⁻⁴ pipeline bar, 3.6 × 10⁻⁴ at worst; its 10⁻³ cross-device bar is a disclosed post-hoc deviation). The readable frame's load excess over a random frame of its rank survives budget matching (+0.162, CI 0.106–0.221). On density, what survives is a within-frame ratio (the readable frame costs 2.74× its complement, random frames 1.00–1.42× theirs). "Darkness is a property of the instrument, not the information" is withdrawn.
- **Paper 17**, [*The Coupling Code*](https://doi.org/10.5281/zenodo.21349350): below a matched random subspace in three of the four linear frames tested, not every one (the deep-anatomical consumer frame sits above the floor, 0.401 against 0.314); nonlinear action is consistent with the measurements but not implied by them.
- **Paper 23**, [*The Implanted Thought*](https://doi.org/10.5281/zenodo.21367358): gates G1 and G2 are retracted. A random subspace of the same dimension does indistinguishable damage (Qwen retention 21/32 for both on the original host; Mistral 0.909 for both), a failure to detect a difference at n = 32, not a demonstration of equivalence. Swapping in another item's 3,328-dimensional component, readout-invisible or random, makes the chain name the donor in the same 22 of 32 items (the donor's capital is the answer in 20 and 21), and 256-dimensional swaps change nothing. The effect is not specific to the readout-invisible frame, and the carrier's width is not identified: any two such subspaces share at least 3,072 of the 3,584 dimensions. CONFESS records a donor mention, not an endorsement; the "~90% dark" rider is withdrawn; concept injection with self-report (Lindsey, 2026) is prior art.
- **Paper 25**, [*The Doors of the Two-Hop*](https://doi.org/10.5281/zenodo.21881892): the "collision" with Gurnee et al. (2026) is not established and the reconciliation is withdrawn (the rates and the log-odds shifts differ in model, intervention and metric); no claim about the workspace result is made. The two doors, the family-compiled selectivity and the write-gain law on the paper's own frame stand.
- **Research ledger (not a paper):** the reading of FP-382b as "identifiability is a phase" is retracted. Random Gaussian dictionaries show the same step at the same widths (Wendel's theorem on the atom count). Paper 27 does not assert it and is unchanged.

## License and author

MIT License (see [LICENSE](LICENSE)). Author: Nathan Ryan Young.
