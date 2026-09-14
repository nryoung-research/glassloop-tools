# FP-688 THE FOURTH SCREEN — held-out language-modelling loss of the audited artifacts (prereg v0; frozen at launch; FP number provisional, Nathan renames)

## Question
Nathan 2026-09-12 (from the Kimi K2 loss curve and Kaplan et al. 2001.08361): the method applied at the plateau might drive general loss down further. The three screens measure a taught skill, a held capability and 641 promised facts; none measures the general language-modelling loss. This read adds it as a fourth screen on artifacts that already exist, with no writes: does an audited write past the plateau leave general loss unchanged, raise it (a price the three screens miss), or lower it (the strong form of the theory)? "Bank fp-688" is the word.

## Design (reads only; base and candidates under one stack per box; tool fp688_lmloss.py 5a73dc34a7f90159)
Sample: the first 200 documents of NeelNanda/pile-10k (hub snapshot 127bfedc), each truncated to its first 512 tokens under the candidate's own tokenizer; teacher-forced mean per-token NLL in nats, bf16, no chat template; per-document NLLs kept in the receipt. Delta = candidate NLL minus its base NLL on the same box and stack. Artifacts: 5090 stack - base 3B, px3 (C-865), base 1.5B, q3 (C-859); box 2 stack - base 3B (the box-2 snapshot), pq3 (C-862), rq3 (C-858), vq3 (C-860), u2 (C-858). LoRA / full fine-tune and the x-ladder exports were deleted after their reads and cannot be scored (disclosed; re-derivable from receipts on request).

## Predictions (frozen at launch)
- P1 loss-neutral at the operating point: |dNLL(px3)| <= 0.02 nats. Credence 0.4.
- P2 the pq lineage inside 0.05: dNLL(pq3) <= 0.05. Credence 0.5.
- P3 the smaller model pays more: dNLL(q3 at 1.5B) > dNLL(px3 at 3B). Credence 0.6.
- P4 no audited artifact LOWERS general loss by more than 0.01 nats (the cheap falsifier of the strong theory). Credence 0.7.
- Reported: the rq lineage ordering u2 (no pool) vs vq3 (vex pool) vs rq3 (vex + coding pool); per-doc distributions; tokens scored.

## Gates
Same text sha and token cap for every read on a box; base read on the same box and stack as each candidate; candidate model sha in the receipt == the export manifest; tool sha == the frozen one. The comparison never crosses boxes (5090 deltas vs 5090 base; box-2 deltas vs box-2 base).

## Not claimed
One text sample; two sizes; general loss as a screen is descriptive here (no threshold set before the read); the prior-art fence C-802 applies.

## FROZEN at launch (2026-09-12 14:29:18 local)
tool flagship/fp616/tools/pilot/fp688_lmloss.py 5a73dc34a7f90159 (smoke: 5 docs x 128 tokens on the 3B base, NLL 2.518, receipt deleted); reads launched on both boxes after this line. Standing word: Nathan 2026-09-12 "Bank fp-688".
