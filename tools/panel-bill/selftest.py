# -*- coding: utf-8 -*-
"""panel-bill selftest: pure python, no downloads, no heavy imports.

Part 1 drives `measure` with a hand-built MockModel and checks the margin
and open-set arithmetic against independently computed values (raw math, not
the library helpers). Part 2 bills a damaged mock against the base mock.
Part 3 pushes fully synthetic measurement jsons through the bill arithmetic
with known answers, covering the edge cases (exactly-zero margins, items
missing from one arm, non-rank-1 open-set baselines).
"""
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from panel_bill import TAIL, MockModel, bill, measure_items  # noqa: E402

VOCAB = ["the", "capital", "of", "france", "is", "paris", "london", "sky",
         "color", "blue", "biggest", "city", "new", "york", "jersey", "red"]
V = len(VOCAB)

CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def logits(**kw):
    out = [0.0] * V
    for w, x in kw.items():
        out[VOCAB.index(w)] = float(x)
    return out


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


ITEMS = [
    dict(id="cap_france", templates=["the capital of france is {answer}"],
         candidates=["paris", "london"], answer="paris"),
    dict(id="sky_color", templates=["the {answer} is blue"],
         candidates=["sky", "color"], answer="sky"),          # non-terminal
    dict(id="big_city", templates=["the biggest city is {answer}"],
         candidates=["new york", "new jersey"], answer="new york"),  # collision
]


def make_model(france_logits, new_logits):
    return MockModel(VOCAB, table={
        ("the", "capital", "of", "france", "is"): france_logits,
        ("the",): logits(sky=1),
        ("new",): new_logits,
    })


def main():
    print("TAIL regex (terminal-slot test), verbatim from the source:")
    check(TAIL.fullmatch("") is not None, "empty tail is terminal")
    check(TAIL.fullmatch(".") is not None, "trailing '.' is terminal")
    check(TAIL.fullmatch(" is blue") is None, "' is blue' is non-terminal")

    print("measure on the base mock:")
    L_FR = logits(paris=2, london=0)
    base = measure_items(make_model(L_FR, logits(york=3)), ITEMS, arm="base")
    m = base["margins"]
    # margins: shared-prefix positions cancel, only the differing answer
    # tokens contribute -- hand values are exact logit differences
    check(abs(m["cap_france"] - 2.0) < 1e-9,
          "margin(cap_france) = lp(paris|ctx) - lp(london|ctx) = 2.0")
    check(abs(m["sky_color"] - 1.0) < 1e-9,
          "margin(sky_color) = 1.0 (only the slot position differs)")
    check(abs(m["big_city"] - 3.0) < 1e-9,
          "margin(big_city) = lp(york|new) - lp(jersey|new) = 3.0")
    check(base["open_set_skipped"] == {"non_terminal": 1, "collision": 1},
          "open-set skips counted by reason: 1 non-terminal, 1 collision")
    check(set(base["open_set"]) == {"cap_france"},
          "only the terminal, collision-free item gets an open-set read")
    rec = base["open_set"]["cap_france"]
    Z = lse(L_FR)
    check(abs(rec["lp_ans"] - (2.0 - Z)) < 1e-6,
          f"open-set lp_ans = 2 - lse = {2.0 - Z:.6f}")
    check(rec["rank_ans"] == 1 and rec["top1_outside"] is False,
          "answer is rank 1, top-1 inside the candidate set")
    want_mass = (math.exp(2.0) + 1.0) / math.exp(Z)
    check(abs(rec["mass_in_set"] - want_mass) < 1e-6,
          f"mass_in_set = (e^2 + e^0)/Z = {want_mass:.6f}")

    print("bill: damaged mock vs base mock:")
    cur = measure_items(
        make_model(logits(paris=-1, london=1), logits(york=-2)),
        ITEMS, arm="damaged")
    rep = bill(base, cur)
    check(rep["n_base_positives"] == 3, "3 base-positives")
    check([r["id"] for r in rep["casualties"]] == ["big_city", "cap_france"],
          "casualty list by name (sorted): big_city, cap_france")
    d = {r["id"]: r["deficit_nats"] for r in rep["casualties"]}
    check(d == {"big_city": 2.0, "cap_france": 2.0},
          "per-casualty deficits = max(0, -margin) = 2.0 nats each")
    check(rep["total_deficit_nats"] == 4.0, "total deficit 4.0 nats")
    check(rep["openset_rank1_losses"] == [dict(id="cap_france", new_rank=16)],
          "open-set rank-1 loss: cap_france fell to rank 16 (below every "
          "zero-logit token)")

    print("bill arithmetic on synthetic measurement jsons:")
    base_doc = dict(arm="base", margins=dict(
        a=2.0, b=0.5, c=-1.0, d=1.0, e=0.0, f=3.0),
        open_set=dict(a=dict(rank_ans=1), b=dict(rank_ans=1),
                      d=dict(rank_ans=3), e=dict(rank_ans=1)))
    cur_doc = dict(arm="arm1", margins=dict(
        a=1.5, b=-0.7, d=0.0, e=-1.0, f=-0.25, z=9.0),
        open_set=dict(a=dict(rank_ans=2), b=dict(rank_ans=1),
                      d=dict(rank_ans=9)))
    # exercise the file round-trip exactly as the CLI does
    with tempfile.TemporaryDirectory() as td:
        pb = os.path.join(td, "base.json")
        pc = os.path.join(td, "cur.json")
        json.dump(base_doc, open(pb, "w"))
        json.dump(cur_doc, open(pc, "w"))
        rep = bill(json.load(open(pb)), json.load(open(pc)))
    check(rep["n_shared"] == 5 and rep["n_only_base"] == 1
          and rep["n_only_current"] == 1,
          "coverage: 5 shared, 1 base-only (c), 1 current-only (z)")
    check(rep["n_base_positives"] == 4,
          "base-positives = 4 (margin 0.0 is NOT positive)")
    check([r["id"] for r in rep["casualties"]] == ["b", "f"],
          "casualties = b, f (d landed exactly at 0.0: not a casualty; "
          "e was never base-positive)")
    check(rep["total_deficit_nats"] == 0.95,
          "deficit sum = 0.7 + 0.25 = 0.95 nats")
    check(rep["field"]["mean_abs_dm"] == 1.39,
          "mean |dm| = (0.5+1.2+1.0+1.0+3.25)/5 = 1.39")
    check(rep["field"]["p95_abs_dm"] == 3.25,
          "p95 |dm| = sorted|dm|[int(5*0.95)] = 3.25")
    check(rep["openset_rank1_losses"] == [dict(id="a", new_rank=2)],
          "open-set losses: only a (b held rank 1; d was never rank 1; "
          "e missing from current arm is not counted)")

    print(f"SELFTEST PASS (panel-bill): {CHECKS} checks")


if __name__ == "__main__":
    main()
