# FP-682b THE PANEL-SPAN PROJECTION LADDER AT 3B — can the in-write projection hold the protected panel without blocking the second family? (prereg v0; frozen at launch; FP number provisional, Nathan renames)

## Question
C-860: with the vex-only pool the two-family ladder keeps coding (109 / 104) but the panel bill rises with every pass (77.7 at the second, 86.4 at the third; the screen is 53.12) and the damage is DIFFUSE (the twenty worst items lose 0.6-2.7 nats each; the total is spread over hundreds of facts), so a casualty-repair pool of a few facts cannot fix it. The coding-span projection held bills (sq: 29 nats) but blocked quill outright (FP-667 / FP-677). This run asks whether a projection onto the PANEL's own activation span (the 641 protected statements) holds the bill while letting quill install - the span of what we promised not to touch, rather than the span of a capability the skill shares structure with. Nathan 2026-09-12: "Proceed with 682".

## Design (box 2 writes and reads; the FP-679b chain with a projection added)
Panel-span bases: fp658_coding_bases.py 6e0646da (the FP-658 builder, unchanged) on the 641 completed statements (bank templates[0] with the gold answer; fp682b_panel_pairs.json 7f56c759, manifest 3e2271c4, pool PANEL641), layers 28-35, rule 1e-2, seed 20260909, built on box 2 at the start of the run (receipt kept; the random control basis is produced but not used). Start artifact rq ed2f987f; writer dba17398 in R mode, quill family, pool VEX60 (cc95a4f2), FG_PROJECT_BASES = the panel-span bases (sha pinned in the writer receipt and checked by the scorer against the builder receipt). Arms: pq2 = quill lessons from rq; pq3 = from pq2. Reads per arm as FP-679b (panel -> price vs base@box2 -> delta norm -> vex teach + paired -> vex held-out -> quill teach + held-out -> anchor). Comparators: vq2 / vq3 (C-860, same chain without the projection).

## Predictions (frozen at launch)
- P1 the panel span holds the bill: bill(pq2) <= 60 (vq2 77.7). Credence 0.5.
- P2 the panel span does not block quill: teach_quill(pq2) >= 45 (vq2 54). Credence 0.5. Falsifier: < 15 -> the panel span overlaps the skill the way the coding span does.
- P3 vex held: teach_vex(pq2) >= 55 AND teach_vex(pq3) >= 55. Credence 0.6.
- P4 coding kept: plus(pq2) >= 105 (vq2 109). Credence 0.5.
- P5 crossings <= 2 at both passes. Credence 0.5.
- P6 the third pass is inside ALL three screens with both families >= 54 (quill, vex >= 54; plus >= 108; bill <= 53.12; crossings <= 2). Credence 0.25. The clean two-family artifact.
- Reported: projected fraction per pass; held-out per family; dose per pass.

## Gates
G0 per arm: writer, mode R, family pins, pool VEX60 with the pinned manifest, projection PRESENT with bases sha == the builder receipt's coding-bases sha and projected steps == accepted, model_root chain (pq2 <- rq, pq3 <- pq2), reload, ARRIVED == manifest, same-stack base, anchor bindings, ARM DONE ledger; SCORER_READY == scorer file. Scorer fp682b_score.py exits UNSCORED for any unbound arm.

## Not claimed
One start artifact, one seed, one rule for the span; the panel span is built from the same 641 facts the bill is measured on (a projection, not a training target: no panel fact enters any loss); the prior-art fence C-802 applies.

## FROZEN at launch (2026-09-12 08:52:38 local) - pins and disclosure
scorer flagship/fp616/tools/pilot/fp682b_score.py 6c9cae0a1341bcc1 (derived from fp679b_score.py; dry run before any read: all UNSCORED); chain C:/fp43/fp682b_box2_chain.sh 81b26768116ee4d1 (as ~/fp682b/chain.sh 5268000b after CRLF strip; copy out/box2/fp682b/chain.sh); fetch C:/fp43/fp682b_fetch.sh; panel pairs 7f56c759 / manifest 3e2271c4 (shipped, shas verified); builder 6e0646da (shipped); writer / pools / start artifact as FP-679b. No smoke of the projected write on box 2 (the same writer ran projected writes on the 5090 in FP-677 / FP-680; the chain fails closed if the bases sha mismatches). Standing word: Nathan 2026-09-12 "Proceed with 682".
