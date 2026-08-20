# -*- coding: utf-8 -*-
"""Oracle selftest for the freshknow meter (fq_score + fq_split).
CPU only, no model, no network, no real FreshQA download: by default it
synthesizes a small FreshQA-FORMAT fixture CSV (same layout: 2 junk
rows, header row 3, 20 columns, quoted multiline fields) and runs every
gate against it. Pass --csv to run the same gates against a real
FreshQA CSV instead.

Gates:
 (a) ORACLE: for 25 deterministically sampled headline items, each true
     answer wrapped in three prose forms must score CONTAINS=1.
 (b) WRONG-ENTITY: a response built from a DIFFERENT item's answer must
     score 0 on both channels (donor chosen so its answer does not
     legitimately overlap the item's own answer set).
 (c) TAXONOMY: empty -> EMPTY; refusal phrasings -> REFUSAL.
 (d) WORD-BOUNDARY: for numeric answers of <= 2 tokens, a constructed
     superstring number must NOT match (plus the canonical '2' vs
     '2026' check).
 (e) SPLIT: build_split() twice -> byte-identical JSON; teach/hidden
     disjoint; every fast-changing item in exactly one bucket.
Exit 0 + 'FQ-SELFTEST-PASS' only if every gate passes.
"""
import argparse
import csv
import json
import os
import random
import sys
import tempfile

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import fq_score  # noqa: E402
import fq_split  # noqa: E402

FAILS = []
CHECKS = [0]

HEADER = (["id", "split", "question", "effective_year", "next_review",
           "false_premise", "num_hops", "fact_type", "source"]
          + [f"answer_{i}" for i in range(10)] + ["note"])


def make_fixture(path):
    """Write a synthetic FreshQA-format CSV: 40 headline items
    (fast-changing, false_premise FALSE) with mutually non-overlapping
    answers, 8 of them carrying a short numeric secondary answer; 5
    side-channel items (false_premise TRUE); 3 non-fast-changing items;
    one quoted multiline note field. Deterministic content."""
    def tag(i):
        return chr(ord("a") + i // 26) + chr(ord("a") + i % 26)

    rows = [["FreshQA synthetic fixture (freshknow selftest)", "", ""],
            ["junk row 2", "", ""],
            HEADER]
    years = ["2022", "2023", "2024", "2025", "2026"]
    for i in range(40):
        t = tag(i)
        answers = [f"kalvor{t} dresset{t}"] + [""] * 9
        if i < 8:
            answers[1] = str(113 + 17 * i)  # short numeric answer
        note = "line one\nline two (quoted multiline)" if i == 0 else ""
        rows.append([f"fx{i:03d}", "TEST",
                     f"Who holds the synthetic office {t} right now?",
                     years[i % 5], "2027-01-01", "FALSE", "one-hop",
                     "fast-changing", "synthetic"] + answers + [note])
    for i in range(5):
        t = tag(100 + i)
        rows.append([f"sc{i:03d}", "TEST",
                     f"Why did the fictional {t} council dissolve?",
                     years[i % 5], "2027-01-01", "TRUE", "one-hop",
                     "fast-changing", "synthetic",
                     f"premise{t} rejected"] + [""] * 9 + [""])
    for i in range(3):
        t = tag(200 + i)
        rows.append([f"sl{i:03d}", "TEST",
                     f"What is the synthetic constant {t}?",
                     "2000", "never", "FALSE", "one-hop",
                     "slow-changing", "synthetic",
                     f"constant{t}"] + [""] * 9 + [""])
    with open(path, "w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerows(rows)
    return path


def check(ok, label):
    CHECKS[0] += 1
    if not ok:
        FAILS.append(label)
        print(f"  FAIL: {label}")


def first_answer(item):
    for i, a in fq_score.item_answers(item):
        if fq_score.answer_tokens(a):
            return i, a
    return None, None


def main():
    ap = argparse.ArgumentParser(description="freshknow meter selftest")
    ap.add_argument("--csv", default=None,
                    help="run gates against this real FreshQA CSV instead "
                         "of the synthetic fixture")
    args = ap.parse_args()
    if args.csv:
        csv_path = args.csv
        print(f"running against user CSV: {csv_path}")
    else:
        csv_path = make_fixture(os.path.join(
            tempfile.mkdtemp(prefix="fq_fixture_"), "fq_fixture.csv"))
        print(f"synthesized fixture: {csv_path}")

    split = fq_split.build_split(csv_path)
    headline = split["headline_teach"] + split["headline_hidden"]
    print(f"headline population n={len(headline)} "
          f"(teach {len(split['headline_teach'])} / "
          f"hidden {len(split['headline_hidden'])})")
    rng = random.Random(556)
    sample = rng.sample(headline, min(25, len(headline)))
    print(f"sampled {len(sample)} items, seed 556: "
          f"{[it['id'] for it in sample]}")

    # (a) oracle: three prose wrappings must hit CONTAINS
    print("[a] oracle prose wrappings")
    for it in sample:
        idx, ans = first_answer(it)
        if ans is None:
            check(False, f"item {it['id']}: no usable answer")
            continue
        forms = [
            f"The answer is {ans}.",
            f"{ans} — as of this year.",
            (f"Well, according to the latest reporting I have seen, "
             f"it turns out that {ans} is what the sources agree on."),
        ]
        for j, resp in enumerate(forms):
            sc = fq_score.score_item(it, resp)
            check(sc["contains"] == 1,
                  f"item {it['id']} form {j} ans={ans!r} "
                  f"missed CONTAINS: {sc}")

    # (b) wrong-entity from a DIFFERENT item's answer -> 0/0
    print("[b] wrong-entity distractors")
    own_norm = {it["id"]: {" ".join(fq_score.answer_tokens(a))
                           for _, a in fq_score.item_answers(it)
                           if fq_score.answer_tokens(a)}
                for it in headline}
    for it in sample:
        found = False
        for donor in headline:
            if donor["id"] == it["id"]:
                continue
            _, dans = first_answer(donor)
            if dans is None:
                continue
            dnorm = " ".join(fq_score.answer_tokens(dans))
            # skip donors whose answer legitimately overlaps this item's
            if any(dnorm in o or o in dnorm for o in own_norm[it["id"]]):
                continue
            sc = fq_score.score_item(it, f"The answer is {dans}.")
            check(sc["contains"] == 0 and sc["strict"] == 0,
                  f"item {it['id']} matched wrong entity {dans!r}: {sc}")
            found = True
            break
        check(found, f"item {it['id']}: no non-overlapping donor found")

    # (c) taxonomy: EMPTY and REFUSAL
    print("[c] failure taxonomy")
    for it in sample[:5]:
        for resp, want in [
                ("", "EMPTY"), ("   \n ", "EMPTY"),
                ("I don't know the answer to that.", "REFUSAL"),
                ("I cannot answer that question.", "REFUSAL"),
                ("As of my knowledge cutoff, this has not "
                 "been decided.", "REFUSAL"),
                ("Purple elephants of Neptune.", "WRONG")]:
            sc = fq_score.score_item(it, resp)
            check(sc["taxonomy"] == want,
                  f"item {it['id']} resp {resp!r}: got "
                  f"{sc['taxonomy']}, want {want}")

    # (d) word-boundary on short numeric answers
    print("[d] word-boundary / superstring numbers")
    sc = fq_score.score_item({"answer_0": "2"},
                             "It happened in 2026.")
    check(sc["contains"] == 0 and sc["strict"] == 0,
          f"canonical: bare '2' matched inside '2026': {sc}")
    n_num = 0
    for it in headline:
        for _, a in fq_score.item_answers(it):
            toks = fq_score.answer_tokens(a)
            if not toks or len(toks) > 2:
                continue
            if not any(any(ch.isdigit() for ch in t) for t in toks):
                continue
            # superstring: append a digit to every numeric token
            sup = [t + "7" if any(ch.isdigit() for ch in t) else t
                   for t in toks]
            probe = {"answer_0": a}  # isolate this answer
            resp = f"The answer is {' '.join(sup)}."
            sc = fq_score.score_item(probe, resp)
            check(sc["contains"] == 0,
                  f"item {it['id']} ans {a!r} matched superstring "
                  f"{' '.join(sup)!r}: {sc}")
            # sanity: the exact answer itself still hits
            sc2 = fq_score.score_item(probe, f"The answer is {a}.")
            check(sc2["contains"] == 1,
                  f"item {it['id']} ans {a!r} failed exact match")
            n_num += 1
    print(f"  numeric <=2-token answers checked: {n_num}")
    check(n_num > 0, "no numeric short answers found to test")

    # (e) split determinism + disjointness + coverage
    print("[e] split determinism and coverage")
    split2 = fq_split.build_split(csv_path)
    check(json.dumps(split, sort_keys=True) ==
          json.dumps(split2, sort_keys=True),
          "build_split() not deterministic across two runs")
    teach_ids = {it["id"] for it in split["headline_teach"]}
    hidden_ids = {it["id"] for it in split["headline_hidden"]}
    side_ids = {it["id"] for it in split["side_channel"]}
    check(not (teach_ids & hidden_ids),
          f"teach/hidden overlap: {teach_ids & hidden_ids}")
    check(not (teach_ids | hidden_ids) & side_ids,
          "side channel overlaps headline")
    _, items = fq_split.load_items(csv_path)
    fast_ids = {it["id"] for it in items
                if it["fact_type"] == "fast-changing"}
    check(fast_ids == (teach_ids | hidden_ids | side_ids),
          "fast-changing items not exactly covered by the 3 buckets")
    per_bucket = [len(teach_ids), len(hidden_ids), len(side_ids)]
    check(sum(per_bucket) == len(fast_ids),
          f"bucket sizes {per_bucket} do not sum to {len(fast_ids)}")

    print(f"checks run: {CHECKS[0]}, failures: {len(FAILS)}")
    if FAILS:
        print("FQ-SELFTEST-FAIL")
        sys.exit(1)
    print("FQ-SELFTEST-PASS")


if __name__ == "__main__":
    main()
