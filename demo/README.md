# Watch It Grow — demo v3, the performance take (2026-08-20)

`loop-watch-it-grow.html` is the demo page, generated MECHANICALLY
from the run's state ledger (`demo_v3_state.json`) and lesson manifest
(`demo_manifest.json`) by `gen_demo_page.py`. The full console log of
the one-sitting `--fresh` run is `demo_v3_fresh.log`.

Verify it yourself:
1. `sha256sum loop-watch-it-grow.html` ->
   `38ff86e3371ae9165558c55f9535e14ff181c4e82c880f5a037f776968e3fe96`
2. Regenerate: `python gen_demo_page.py demo_v3_state.json demo_manifest.json out.html`
   -> byte-identical to the published page.
3. Quote gate: `python demo_page_check.py loop-watch-it-grow.html demo_v3_state.json`
   -> 14/14 quotes byte-exact against the ledger's recorded generations.

The generator FAILS CLOSED: if the ledger's revert verdicts are not
both MATCH, no page is produced. Damage numbers are observed
casualties on the declared 40-item watch-list; the sealed 641-item
panel is the instrument of record (see the papers). The int8 press is
a simulated lattice (RTN in place), not a packed artifact.

Pre-publish audit: independently delta-verified 2026-08-20
(regeneration byte-identity, quote gate, fail-closed tamper test,
hash match) before this commit.
