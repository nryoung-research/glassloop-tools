# -*- coding: utf-8 -*-
"""meter-check selftest: pure python, no downloads, no heavy imports.

Proves each channel's arithmetic against independently hand-computed values
(log-sum-exp done here with raw math, not the library helper) and shows the
artifact signature being caught: a model that emits a preamble token before
the correct answer fails FIRST-TOKEN while FC and OPEN pass.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from meter_check import MockModel, run_items, score_item  # noqa: E402

VOCAB = ["the", "capital", "of", "france", "is", "paris", "london", "well",
         ",", "big", "city", "new", "york", "england", "japan", "tokyo"]
V = len(VOCAB)


def logits(**kw):
    """Raw logits vector: zeros except the named words."""
    out = [0.0] * V
    for w, x in kw.items():
        out[VOCAB.index(w)] = float(x)
    return out


def lse(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


CHECKS = 0


def check(cond, msg):
    global CHECKS
    CHECKS += 1
    if not cond:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"  ok: {msg}")


def main():
    L_A = logits(paris=5, london=1)                 # clean knowledge
    L_B = logits(well=6, london=5, paris=1)         # preamble artifact
    L_C = logits(new=3, paris=2)                    # multi-token answer
    L_D = logits(london=5, tokyo=1)                 # genuinely wrong
    model = MockModel(VOCAB, table={
        ("the", "capital", "of", "france", "is"): L_A,
        ("the", "capital", "of", "england", "is"): L_B,
        ("the", "big", "city", "is"): L_C,
        ("the", "capital", "of", "japan", "is"): L_D,
        ("new",): logits(york=4),
        ("well",): logits(**{",": 4}),
        (",",): logits(london=4),
    })

    items = [
        dict(id="caseA", prompt="the capital of france is",
             answer="paris", answer_set=["london", "paris"]),
        dict(id="caseB", prompt="the capital of england is",
             answer="london", answer_set=["london", "paris"]),
        dict(id="caseC", prompt="the big city is",
             answer="new york", answer_set=["new york", "paris"]),
        dict(id="caseD", prompt="the capital of japan is",
             answer="tokyo", answer_set=["tokyo", "london"]),
    ]
    rep = run_items(model, items)
    rows = {r["id"]: r for r in rep["rows"]}

    print("channel verdicts (FIRST-TOKEN / FC / OPEN):")
    check(rows["caseA"]["pattern"] == "1/1/1", "caseA clean: 1/1/1")
    check(rows["caseB"]["pattern"] == "0/1/1",
          "caseB artifact: first-token refused, FC+OPEN passed (0/1/1)")
    check(rows["caseC"]["pattern"] == "1/1/1",
          "caseC multi-token answer: 1/1/1")
    check(rows["caseD"]["pattern"] == "0/0/0", "caseD truly wrong: 0/0/0")

    print("FC arithmetic vs hand-computed full-answer log-probs:")
    lseA = lse(L_A)
    fa = rows["caseA"]["fc_scores"]
    check(abs(fa["paris"] - (5 - lseA)) < 1e-6,
          f"caseA fc(paris) = 5 - lse = {5 - lseA:.6f}")
    check(abs(fa["london"] - (1 - lseA)) < 1e-6,
          f"caseA fc(london) = 1 - lse = {1 - lseA:.6f}")
    lse_is = lse(L_C)
    lse_new = lse(logits(york=4))
    fc = rows["caseC"]["fc_scores"]
    want_ny = (3 - lse_is) + (4 - lse_new)   # lp(new|is) + lp(york|new)
    check(abs(fc["new york"] - want_ny) < 1e-6,
          f"caseC fc(new york) sums BOTH answer tokens = {want_ny:.6f}")
    check(abs(fc["paris"] - (2 - lse_is)) < 1e-6,
          f"caseC fc(paris) = {2 - lse_is:.6f}")
    check(fc["new york"] > fc["paris"],
          "caseC forced choice picks the multi-token answer")

    print("OPEN channel greedy decode:")
    check(rows["caseB"]["generation"].startswith("well , london"),
          f"caseB generation = {rows['caseB']['generation']!r} "
          "(preamble then the true answer)")
    check(len(rows["caseA"]["generation"].split()) == 12,
          "OPEN decodes exactly n_open=12 tokens")

    print("summary / artifact census:")
    check(rep["n"] == 4 and rep["first_token_pass"] == 2
          and rep["fc_pass"] == 3 and rep["open_pass"] == 3,
          "pass counts: FIRST-TOKEN 2/4, FC 3/4, OPEN 3/4")
    check(rep["artifact_signature_ids"] == ["caseB"],
          "artifact signature flags exactly caseB")
    check(rep["disagreement_ids"] == ["caseB"],
          "channel disagreement flags exactly caseB")

    print("fail-loud on malformed items:")
    try:
        score_item(model, "the capital of france is", "paris", ["london"])
        check(False, "answer outside answer_set must raise")
    except ValueError:
        check(True, "answer outside answer_set raises ValueError")

    print(f"SELFTEST PASS (meter-check): {CHECKS} checks")


if __name__ == "__main__":
    main()
