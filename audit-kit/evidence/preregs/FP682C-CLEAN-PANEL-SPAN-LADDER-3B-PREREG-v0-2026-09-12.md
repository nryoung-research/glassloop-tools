# FP-682c THE CLEAN LADDER WITH THE PANEL-SPAN PROJECTION AT 3B — the two remedies together from S1 (prereg v0; frozen at launch; FP number provisional, Nathan renames)

## Question
FP-682a (C-861): from S1 under the vex pool alone, the second pass holds both families (60 / 55) with coding at base (112) and fails only the panel bill (80.9). FP-682b (box 2, in progress): from rq, the projection onto the 641-statement panel span holds the bill (pq2: 50.6 / 0 crossings) at quill 47 and coding 107. This run combines them from the clean start: S1, vex pool, panel-span projection, three passes. It is the direct test of P6 (a two-family artifact inside all three screens). Nathan 2026-09-12: "Proceed with 682".

## Design (5090, one stack = FP-678 base; the FP-682a chain with FG_PROJECT_BASES added)
Panel-span bases built locally by fp658_coding_bases.py (unchanged) on fp682b_panel_pairs.json 7f56c759 / manifest 3e2271c4 (PANEL641), layers 28-35, rule 1e-2, seed 20260909: out/fp682c/bases/fp658_coding_bases.pt 75641c46 (ranks resid 827-1160 / mlp 29-2163 per layer, matching the box-2 build to the unit; receipt kept). Start S1 5e8b5d28; writer dba17398 in R mode, quill family 30057895, pool VEX60 (cc95a4f2), FG_PROJECT_BASES = the panel span (sha pinned in every writer receipt and checked by the scorer against the builder receipt). Arms px1 <- S1, px2 <- px1, px3 <- px2. Reads: the FP-678 runner e570bce2 with FP678_FAMILIES=vex,quill against the FP-678 base panel; HumanEval+ in WSL; fp656 window norms.

## Screens on this stack
teach >= 54; plus >= 106 (base 112 - 6); bill <= 53.12 with <= 2 crossings.

## Predictions (frozen at launch)
- P1 the panel span holds the bill at the second pass: bill(px2) <= 53.12 with <= 2 crossings (x2 unprojected: 80.9 / 1). Credence 0.5.
- P2 quill installs by the second pass: teach_quill(px2) >= 45 (x2: 55; pq2 under the span from rq: 47). Credence 0.5.
- P3 vex held: teach_vex(px3) >= 55. Credence 0.6.
- P4 coding inside at the second pass: plus(px2) >= 106 (x2: 112). Credence 0.5.
- P5 quill monotone in passes. Credence 0.6.
- P6 some pass is inside ALL three screens with both families >= 54. Credence 0.35. The clean two-family artifact.
- Reported: projected fraction per pass; held-out per family; bills, crossings, dose per pass.

## Gates
As FP-682a plus: projection PRESENT in every writer receipt with bases sha == the local builder receipt's coding-bases sha and projected steps == accepted steps. Scorer fp682c_score.py exits nonzero unless G0.

## Not claimed
One seed; the span is built from the same 641 facts the bill is measured on (a projection, never a training target: no panel fact enters any loss); nothing about a fourth screen; the prior-art fence C-802 applies.

## FROZEN at launch (2026-09-12 11:06:20 local) - pins and disclosure
scorer flagship/fp616/tools/pilot/fp682c_score.py 12ace764c5a68f07 before any read (derived from fp682a_score.py; dry run: all MISSING / UNSCORED); chain C:/fp43/fp682c_chain.sh a7831d87e175173a (copy out/fp682c/chain.sh; derived from fp682a_chain.sh with the projection env); bases 75641c46 (built 11:05 local, out/fp682c/bases/); runner e570bce2; writer dba17398; start S1 5e8b5d28. No smoke (the projected write ran in FP-680 on this machine and in FP-682b on box 2 with this writer; the chain fails closed on a bases-sha mismatch). Standing word: Nathan 2026-09-12 "Proceed with 682".
