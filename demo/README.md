# Watch It Grow — demo v3, the performance take (2026-08-20; page rev 2, 2026-08-21)

`loop-watch-it-grow.html` is the demo page, generated MECHANICALLY
from the run's state ledger (`demo_v3_state.json`) and lesson manifest
(`demo_manifest.json`) by `gen_demo_page.py`, using the style/JS
templates in `demo-pages/` and the repair-round edit certificates in
`demo_v3/`. The full console log of the one-sitting `--fresh` run is
`demo_v3_fresh.log`; the run driver is `fp_demo_v3.py`.

Verify it yourself — all of this works from THIS DIRECTORY alone:
1. `sha256sum loop-watch-it-grow.html` ->
   `d378e69bbb6ae1c9b605f85d0d4ebfde85abd928d0f51a1b1e720a2a60cf5bdc`
2. Regenerate: `python gen_demo_page.py demo_v3_state.json demo_manifest.json out.html`
   -> byte-identical to the published page.
3. Quote gate: `python demo_page_check.py loop-watch-it-grow.html demo_v3_state.json`
   -> 14/14 quotes byte-exact against the ledger's recorded generations.

REPRODUCTION SCOPE, honestly stated: steps 1-3 (the page and every
number on it) reproduce from this directory. Re-running the MODEL RUN
itself (`fp_demo_v3.py`) additionally requires the FreshQA snapshot
CSV, the lesson specs, and a GPU with the model; the manifest's
absolute paths are the original run's provenance record, not portable
inputs. The run is one-command reproducible on the original rig; it is
not yet packaged for stranger reproduction.

PAGE REV 2 (2026-08-21, after an external stranger-audit — every one
of its numbers verified against the ledger before adoption):
- MEASUREMENT DISCLOSURE added to Act 5: the repair trains on the
  watch-list's own items, so post-repair DAMAGE 0 means the meter
  reads clean, NOT healed-to-baseline (net +270 nats vs base across
  the 40 items; 37/40 above their base margins; repaired statements
  generalize to unseen phrasings at 11% -> 7% -> 6%). Acts 6-7 are
  scoped as read on that pinned meter.
- Instrumentation named on the page: Qwen2.5-3B-Instruct (fp32),
  margins in nats, the casualty rule, dose lr 2e-5 x 200 steps, MLP
  layers 28-35.
- The two page templates, the run driver, and the Act-5 edit
  certificates were added so regeneration works from this directory
  (they were missing in rev 1 — the audit's reproducibility finding).

The generator FAILS CLOSED: if the ledger's revert verdicts are not
both MATCH, no page is produced. The sealed 641-item panel is the
instrument of record for the papers; the sealed hidden split is never
read by the run. The int8 press is a simulated lattice (RTN in
place), not a packed artifact.

Prior page rev 1 sha256 (superseded by the rev-2 disclosure, retained
for the record):
`38ff86e3371ae9165558c55f9535e14ff181c4e82c880f5a037f776968e3fe96`
