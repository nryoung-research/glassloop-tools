# The papers — a map of the program

Thirty-two papers (numbered 1–33; number 13 is unused), two objects of study. The first: **the part of a transformer's residual
stream that its own output layer cannot read, and the computation that lives there** (papers
1–28). The second, which grew out of the first: **what a weight update does to a model — what
it costs, whether it can be audited, and whether it can be undone** (papers 29–33). Every
result is pre-registered and sealed before the run; negative results are banked at the same
weight as positive ones, and at least one paper is entirely a kill record.

The papers are cross-braced and share a vocabulary, which makes them hard to enter at a random
point. This is the map. Every paper's DOI, live-verified, is in **[CORPUS.md](CORPUS.md)**; the
tools packaged from the change arc, and three live demonstration pages, are in this
repository's [README](README.md).

---

## Where to start, depending on why you are here

**"Just tell me what this is."** → Paper **00, *Reading the Unspoken — A Primer*** for the
state arc, or **00b, *Reading the Change — A Primer*** for the editing/continual-learning
arc. Both are written as entry points rather than results (both are in preparation for
deposit; see [CORPUS.md](CORPUS.md) for what carries a DOI today).

**"I care about superposition and interference."** → **2** (the law), then **7** (the same law
measured natively in a real transformer's MLP), then **11** (why the scaling is d²).

**"I care about what interpretability instruments miss."** → **15** (*Weighing and Reading the
Dark Matter*), then **14** (what the Jacobian lens deletes and why), then **17** (why the missing
part resisted every linear instrument — it is couplings, not directions).

**"I care about editing and control."** → **8** (*Model Surgery*), then **22** (*The Forced-Light
Editor*), then **26** (*Trained in the Light* — what it costs to constrain a model's writes to be
readable, in from-scratch pretraining runs).

**"I care about continual learning, and whether weight edits can be made safe."** → **00b**
(the arc's primer), then **29** (*Auditable Continual Learning* — the transaction
architecture), then **30** (the damage laws), then **31** (what repair costs, honestly
priced).

**"I care about introspection and self-report."** → **21** (*The Self-Knowledge Map*), then **25**
(*The Doors of the Two-Hop*).

**"I want the mathematics."** → **1**, then **27**. See the note on that pair below.

---

## The arc, in six movements

### I. The law (papers 1–7, 11)

How features interfere when a model packs more of them than it has dimensions.

| | | |
|---|---|---|
| **1** | *Rectified Frame Potential* | The mathematical object the program is built on: a one-sided frame potential penalizing only acute inner products. |
| **2** | *Co-occurrence Law* | Derives a two-term variational principle for learned feature geometry from the reconstruction loss of a tied ReLU autoencoder. |
| **3** | *Designed Feature Codes* | What you gain by *designing* a sparse dictionary instead of learning or randomizing one — Walsh–Hadamard cosets with exact algebraic decodability. |
| **4** | *Sign Not Magnitude and the Resummed Law* | Why the sign structure of the Gram matrix, not its magnitudes, governs the loss. |
| **5** | *Interference Modes in Real Language Models* | Takes the law out of designed worlds and into real models. |
| **6** | *Scale, Depth and Downstream* | Blind-predictive across regimes: an eight-regime summary with sixty-four interference modes. |
| **7** | *The Native MLP Interference Law* | Superposition's cost structure, measured in a real transformer's own MLP rather than in an SAE. |
| **11** | *Why d-squared, Derived and Controlled* | Derives the empirical scaling rather than fitting it. |

### II. The ruler (papers 8–10, 12)

Every decision a transformer makes is a thresholded selection, and its fragility is calculable
*in advance*.

| | | |
|---|---|---|
| **9** | *The Router is a Ruler* | MoE routing is fragile exactly when the k-th and (k+1)-th scores are close — and that is predictable. |
| **10** | *One Ruler for Every Selector* | The same ruler governs MoE routers, the vocabulary head, and nearest-neighbour retrieval. One instrument, three organs. |
| **8** | *Model Surgery* | A complete loop: diagnose localized weight damage read-only, treat it, and calculate the treatment beforehand. |
| **12** | *The Price of Dust* | The tied embedding/readout matrix — the most legible organ a model has — and the training "wake" left in it. |

### III. The dark (papers 14–19)

The program's centre of gravity: most of a model's mid-computation is invisible to standard
readouts, and it is load-bearing.

| | | |
|---|---|---|
| **14** | *What the Lens Deletes* | The averaging that makes a Jacobian lens work is exactly what makes it blind. |
| **15** | *Weighing and Reading the Dark Matter* | What linear readouts structurally discard — and it is the majority of mid-layer computation. |
| **16** | *Reading What a Model Is Doing* | The complementary question: the *contents* may be dark, but the **operation** is legible. |
| **17** | *The Coupling Code* | Why the dark centre resisted every linear instrument — it acts through nonlinear couplings, not directions. |
| **18** | *The Dark Atlas* | Mapping it. |
| **19** | *The Work Meter* | Pricing the work the dark computation performs. |

### IV. Writing, and what the model knows (papers 20–25)

| | | |
|---|---|---|
| **24** | *The Capacity Knee* | Three priceable quantities in the residual stream: stored knowledge, ongoing computation, and **working state**. This paper prices the third. |
| **21** | *The Self-Knowledge Map* | Does a model know whether its own answer is correct? Measured as a map, not a number — eight models, four families, 0.5B–12B. |
| **22** | *The Forced-Light Editor* | Reading and writing the dark state from outside. |
| **23** | *The Implanted Thought* | Putting an unspoken intermediate conclusion into a model and watching it act. |
| **20** | *The Interaction Reader* | **A kill record.** Published because the program banks negative results at the weight of positive ones. |
| **25** | *The Doors of the Two-Hop* | Reconciles an apparent collision with Gurnee et al. (2026) on whether content outside the verbalizable workspace re-enters behaviour. |

### V. The synthesis (papers 26–28)

| | | |
|---|---|---|
| **26** | *Trained in the Light* | The alignment between a model's residual writes and its readout is unconstrained by the training objective. This asks what constraining it **costs**, in paired from-scratch pretraining runs. |
| **27** | *The Rectifier Is the Reader* | For a linear code the codebook is a gauge choice — there is no fact of the matter about which dictionary a data stream uses. The rectifier is what breaks that gauge freedom. |
| **28** | *A Causal Map of a Decision* | A multi-scale, intervention-verified causal map of a simple agent decision (observation + goal → yes/no action) inside a language model. |

### VI. The change (papers 29–33)

The second arc. Papers 1–28 measure what is *in* a model at a moment; these papers measure what
an *update* to a model did — the transaction, its collateral damage, its repair cost, and the
audit trail that makes any of it checkable.

| | | |
|---|---|---|
| **29** | *Auditable Continual Learning* | Sequential weight edits as measured transactions: sandbox candidate, independent panel, mechanical admission rule, verified rollback, append-only ledger. Includes the arc's founding negative result: forecast gating misses 63.2% of realized margin displacement — you must measure, not predict. |
| **30** | *The Price of a Lesson* | The damage laws for sequential weight edits: damage is trajectory-governed, not update-governed; placement works where allocation does not; understanding costs about twice what memorization does. |
| **31** | *Casualty-Directed Repair* | Closing the damage loop: repair reaches zero observed casualties over seven rounds — non-monotonically; one round repaired six items and minted seven — and mints fresh damage throughout. The gross-healing tax is measured (2.14×), and the closed-set "zero" is reported beside its open-set complement. |
| **32** | *The Three Levers of QAT* | A mechanism taxonomy for quantization-aware robustness, with the popular margin explanation refuted for the measured pair. An instrument paper for intervention accounting. |
| **33** | *Weight Endpoint Continuity* | Lineage-consistent correlation and aligned update geometry in public tensors: an instrument for endpoint continuity, CPU-only, minutes per pair — with its own pre-registered miss reported at full weight. |

---

## A note on papers 1 and 27

These are the two ends of a single thread, and they are the natural pair to read together.

Paper **1** introduces and prices the rectified frame potential as a geometric object. Paper **27**
proves that the same rectifier is what makes a codebook *identifiable at all* — without it, the
dictionary is a gauge choice and no reading of a model's internal code can be said to be correct.
The first paper builds the instrument; the last one shows the instrument is the reason reading is
possible.

If you arrived at 27 first, 1 is the paper it stands on.

---

## What is not here yet

The change arc's next installment is in preparation: a campaign record on **contract-monotone
continual learning** — a transaction system whose behavioral contract is held invariant while
the model changes, demonstrated in both verdict directions on sealed, independently audited
campaigns: refuse-when-bad under deliberate overdose, and accept-when-good under knee dosing
(one admitted transaction, durable through the rest of its campaign, with its three held-out
paraphrases scoring 0/3 — the generalization boundary disclosed beside the success). The
claim it defends is scoped: an auditable continual-repair architecture with bounded evidence of
repeated local improvement, plus instruments for intervention accounting and endpoint
continuity.

---

## Verification

Each paper's experiments are sealed and pre-registered before the run. The verification code and
pre-registration records are deposited alongside the papers as numbered archives (`0`, `0b`–`0p`,
a paper-21 artifact deposit, plus per-paper deposits for the continual-learning arc:
`deposit-CL`, `deposit-31`, `deposit-32`, `deposit-33`), each with its own DOI, covering the
experiment ranges cited in each paper's title.

Contact: Nathan Ryan Young · ORCID [0009-0003-2840-7726](https://orcid.org/0009-0003-2840-7726)
