# FP-700 - THE OUT-OF-SPAN PANEL: is the bill a price or a receipt for a constraint honoured? (pre-registration v0, SEALED 2026-09-13 01:09:30 local, before any read)

Trigger: audit-kit cold test 3 (2026-09-13, a fresh agent): "the bill screen is graded against the write's own input: the projection bases were built from those exact 641 prompts ... no arm in the kit reads facts outside the projected span ... a same-artifact read of 641 unprojected facts drawn from the 6,956-item bank is the cheapest way to show the price is a price." Nathan (2026-09-13): keep all boxes busy; 5090 interruptible.

## Design (frozen)
- Panel: flagship/glassloop/fp700_panel_oos.json 10610c826da0baf7 = 641 bank items NOT in fp467_panel_v2 (overlap 0), same source composition as the parent (BEAR 388, PopQA 142, CounterFact 90, ARC 14, MMLU 7), seed 700, sorted by id. Reader: panel_read_hf.py (unchanged), bf16, 5090 venv_ee (torch 2.7.1+cu128, tf 5.5.4), the same stack as every FP-678/682c read; price_state.py vs the base OOS read.
- Reads: base (Qwen2.5-3B-Instruct aa8e7253), S1 (out/fp662/P_coding/A_written_bf16, the lineage root: coding-span-projected vex install), px3 (out/fp682c/px3/A_written_bf16, the two-family artifact: S1 + three panel-span-projected quill passes). No writes. ~40 min.
- Comparison numbers (in-span, vs the same base, same stack): px3 bill 33.76 / 1 crossing (C-865); S1's in-span bill vs the FP-678 base is not on record (S1 was priced against the FP-662 base under a different stack), so S1 is read in-span too (L2_S1 vs L2_fp678_base) in this chain, for a like-for-like pair.

## Predictions (frozen)
- P1 (the price is a price): bill_oos(px3) <= 2.0 x bill_in(px3) = 67.5 nats. Credence 0.5. (A receipt-for-a-constraint would show a much larger out-of-span bill; a price would show the same order.)
- P2 crossings_oos(px3) <= 5. Credence 0.5.
- P3 the lineage order holds out of span: bill_oos(px3) >= bill_oos(S1) (the three quill passes do not repair S1's damage on unseen facts). Credence 0.7.
- P4 the in-span/out-of-span ratio is similar for the two artifacts: |ratio(px3) - ratio(S1)| <= 0.5 where ratio = bill_oos / bill_in. Credence 0.4.
- P5 out-of-span damage is relation-local in the same way (memory: 92 percent of high-margin flips same-relation): of the OOS facts that lose >= 2 nats under px3, at least 60 percent share a source with an in-span fact that lost >= 2 nats. Credence 0.4 (a weak proxy for relation locality; sources are coarse).

## Readings
- P1 MISS (bill_oos far above 2x): the span projection buys fidelity to the constraint, not preservation of knowledge; every "641 facts held" sentence in the kit must add "on the protected panel; out-of-span damage X"; FP-688 remains the general-damage measure. A random-span control (FP-682d) then becomes mandatory.
- P1 HOLD: the bill screen transfers to unseen facts at this size within a factor of two; the README's "kept" gains a second panel.

## Not claimed
Anything about the 1.5B/7B artifacts (not on this box); a threshold for out-of-span damage (none frozen); anything about the fine-tune baselines (exports deleted).
