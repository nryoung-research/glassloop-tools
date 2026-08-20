"""Self-test for the FP-558 lesson payloads (lessons_sft.jsonl + specs/).

Asserts, against the EMITTED files:
  1. LEAK SCAN -- no holdout task id, no holdout F-tier case expression
     (raw or quirklib.-prefixed, whitespace-normalized), no holdout S-tier
     instruction step-block, and no holdout test JSON fragment appears in
     the decoded content of lessons_sft.jsonl or specs/*.json. (This is
     the only file allowed to READ holdout artifacts, and only for
     string-level scanning.)
  2. REFERENCE EXECUTION -- every embedded reference solution (task_F and
     task_S records) passes its own train-task tests via score_exec
     (score 1.0), and the spec's task-block passages embed byte-identical
     reference code.
  3. SPEC INTEGRITY -- 24 items / 16 holdout probes, expression-disjoint;
     every expression re-evaluates (real quirklib) to its stored value;
     dose fields match the canonical template verbatim; fact_text_perm is
     a permutation; every item is rendered in an "evaluates to" passage;
     no holdout-probe evaluation is rendered in any non-task passage.
  4. D19 -- code-context identifier fraction >= 0.5.

Prints PASS/FAIL per check and exits nonzero on any failure.
"""

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quirklib      # noqa: E402
import gen_tasks     # noqa: E402  (header/footer constants only)
import gen_lessons   # noqa: E402  (D19 predicate)
import score_exec    # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print("[{}] {}{}".format("PASS" if ok else "FAIL", name,
                             ("  " + detail) if detail else ""))
    if not ok:
        FAILS.append(name)


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def read_jsonl(path):
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def string_leaves(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from string_leaves(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from string_leaves(v)


def main():
    lessons = read_jsonl(HERE / "lessons_sft.jsonl")
    spec_paths = sorted((HERE / "specs").glob("*.json"))
    specs = {p.name: json.loads(p.read_text(encoding="utf-8"))
             for p in spec_paths}
    check("files present", bool(lessons) and bool(specs),
          "{} lesson records, specs: {}".format(
              len(lessons), [p.name for p in spec_paths]))

    # Decoded haystack: every string that could reach a model.
    hay_parts = [m["content"] for r in lessons for m in r["messages"]]
    for sp in specs.values():
        hay_parts.extend(string_leaves(sp))
    hay = norm("\n".join(hay_parts))

    # ---- 1. leak scan ----------------------------------------------------
    hold_f = read_jsonl(HERE / "tasks" / "holdout_F.jsonl")
    hold_s = read_jsonl(HERE / "tasks" / "holdout_S.jsonl")
    needles = []  # (label, normalized needle)
    for t in hold_f + hold_s:
        needles.append(("id " + t["id"], norm(t["id"])))
        for tst in t["tests"]:
            needles.append(
                ("test-json " + t["id"],
                 norm(json.dumps(tst, sort_keys=True)[1:-1])))
    case_re = re.compile(r"^  case \d+: (.+)$", re.M)
    for t in hold_f:
        for expr in case_re.findall(t["instruction"]):
            needles.append(("F-expr " + t["id"], norm(expr)))
            needles.append(("F-expr(q) " + t["id"],
                            norm("quirklib." + expr)))
    for t in hold_s:
        body = t["instruction"]
        assert body.startswith(gen_tasks.S_HEADER) \
            and body.endswith(gen_tasks.S_FOOTER)
        block = body[len(gen_tasks.S_HEADER):-len(gen_tasks.S_FOOTER)]
        needles.append(("S-steps " + t["id"], norm(block)))
    hits = [lab for lab, n in needles if n and n in hay]
    check("leak scan: no holdout string in lessons/specs",
          not hits, "{} needles scanned{}".format(
              len(needles),
              "" if not hits else "; HITS: " + ", ".join(hits[:8])))

    # ---- 2. reference execution -----------------------------------------
    train_tasks = {t["id"]: t for t in
                   read_jsonl(HERE / "tasks" / "train_F.jsonl")
                   + read_jsonl(HERE / "tasks" / "train_S.jsonl")}
    task_recs = [r for r in lessons if r["kind"] in ("task_F", "task_S")]
    check("task record count", len(task_recs) == 80,
          "{} task records".format(len(task_recs)))
    bad = []
    for r in task_recs:
        task = train_tasks[r["id"]]
        item = score_exec.score_response(
            task, r["messages"][1]["content"], timeout=15.0)
        if item["score"] != 1.0:
            bad.append((r["id"], item["error_summary"][:80]))
    check("all embedded references pass their tests via score_exec",
          not bad, "{} scored{}".format(
              len(task_recs),
              "" if not bad else "; FAILED: " + repr(bad[:4])))

    sp = specs["quirk_full.json"]
    ref_by_id = {r["id"]: r["messages"][1]["content"] for r in task_recs}
    mism = []
    task_passages = [ft for ft in sp["fact_texts"]
                     if ft.startswith("Task ")]
    for ft in task_passages:
        tid = ft.split(":", 1)[0][len("Task "):]
        code = ft.split("\n\nSolution:\n", 1)[1]
        lesson_code = ref_by_id[tid]
        inner = lesson_code[len("```python\n"):-len("\n```")]
        if code != inner:
            mism.append(tid)
    check("spec task passages embed identical reference code",
          len(task_passages) == 80 and not mism,
          "{} passages{}".format(len(task_passages),
                                 "" if not mism else "; MISMATCH " +
                                 repr(mism[:4])))

    # ---- 3. spec integrity ----------------------------------------------
    items = sp["items"]
    probes = sp["holdout_probes"]
    probe_exprs = [p[0][:-len(" evaluates to")] for p in probes]
    check("spec counts", len(items) == 24 and len(probes) == 16)
    check("items/probes expression-disjoint",
          not set(e for e, _ in items) & set(probe_exprs))
    ev_bad = []
    for expr, val in items + list(zip(probe_exprs, [p[1] for p in probes])):
        got = eval(expr, {"__builtins__": {}, "quirklib": quirklib})
        if repr(got) != val:
            ev_bad.append((expr, val, repr(got)))
    check("all 40 spec evaluations re-verify against real quirklib",
          not ev_bad, repr(ev_bad[:3]) if ev_bad else "")
    check("dose fields verbatim from template",
          sp["layers"] == [28, 35] and sp["lr"] == 5e-05
          and sp["steps"] == 200 and sp["slack_frac"] == 0.125
          and sp["seed"] == 536 and sp["task_bar"] == 0.85)
    check("templates use {n}/{t} placeholders (gl_edit2m .format(n=,t=))",
          "{n}" in sp["probe_template"] and "{n}" in sp["fact_template"]
          and "{t}" in sp["fact_template"])
    perm = sp["fact_text_perm"]
    check("fact_text_perm is a permutation of fact_texts",
          sorted(perm) == list(range(len(sp["fact_texts"]))))
    non_task = [ft for ft in sp["fact_texts"] if not ft.startswith("Task ")]
    non_task_norm = norm("\n".join(non_task))
    miss = [e for e, _ in items
            if norm("%s evaluates to" % e) not in non_task_norm]
    check("every item rendered in an 'evaluates to' passage", not miss,
          repr(miss[:3]) if miss else "")
    leak = [e for e in probe_exprs if norm(e) in non_task_norm]
    check("no holdout-probe evaluation in any non-task passage", not leak,
          repr(leak[:3]) if leak else "")

    # ---- 4. D19 code-context fraction -----------------------------------
    n_ctx = sum(1 for r in lessons if gen_lessons.record_code_context(r))
    frac = n_ctx / len(lessons)
    check("code-context identifier fraction >= 0.5", frac >= 0.5,
          "{}/{} = {:.3f}".format(n_ctx, len(lessons), frac))

    print("-" * 60)
    if FAILS:
        print("selftest_lessons: {} check(s) FAILED: {}".format(
            len(FAILS), FAILS))
        return 1
    print("selftest_lessons: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
