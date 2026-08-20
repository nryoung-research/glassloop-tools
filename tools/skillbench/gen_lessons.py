"""Deterministic lesson generator for FP-558 (skills pilot, P1).

Sources (the ONLY content inputs, per prereg P1):
  - quirklib docstrings: semantics, nicknames, function signatures
  - tasks/train_F.jsonl + tasks/train_S.jsonl (instructions, test args)
  - solutions/train_F.jsonl + solutions/train_S.jsonl (reference solutions,
    train-side evaluation cases)
This file NEVER reads any holdout artifact. Worked-example evaluations are
drawn exclusively from train-F cases (which gen_tasks globally deduped
against holdout), NOT from the docstring Example blocks -- docstring
examples live in the same small argument space as holdout draws and cannot
be deduped against them, so they are excluded from every lesson payload.

Outputs:
  lessons_sft.jsonl   chat-format SFT records for the LoRA / full-FT arms:
                      (a) API teaching (semantics / nickname / worked
                          example, code in ```python fences),
                      (b) TIER-F train tasks verbatim -> fenced reference,
                      (c) TIER-S train tasks -> fenced reference.
  specs/quirk_full.json  gl_edit2m spec (R1/R2 rungs): whole library + all
                      train tasks as one lesson. layers/lr/steps/
                      slack_frac/seed/task_bar copied verbatim from the
                      canonical template spec (sealed fp538 artifact spec_L06_T1);
                      dose changes are the coordinator's decision.

Per METER-REDTEAM D19, quirklib identifiers must appear in CODE CONTEXT
(inside fences, after newline+indent, after `ql.`/`quirklib.`, after `(`)
in a majority of records; the fraction is counted and printed.

Prints the sha256 of every emitted file.
"""

import argparse
import copy
import hashlib
import inspect
import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quirklib  # noqa: E402

GEN_VERSION = "1.0"
DEFAULT_SEED = 558
FN_NAMES = sorted(n for n in dir(quirklib)
                  if not n.startswith("_")
                  and callable(getattr(quirklib, n)))

# Copied VERBATIM from the sealed fp538 template spec spec_L06_T1 (canonical).
# Dose/knob changes are the coordinator's decision, not this generator's.
TPL_LAYERS = [28, 35]
TPL_LR = 5e-05
TPL_STEPS = 200
TPL_SLACK_FRAC = 0.125
TPL_SEED = 536
TPL_TASK_BAR = 0.85

N_ITEMS = 24
N_HOLDOUT_PROBES = 16

# --------------------------------------------------------------------------
# D19 code-context predicate (shared with selftest_lessons.py)
# --------------------------------------------------------------------------
_FN_ALT = "|".join(FN_NAMES)
_RE_FENCE = re.compile(r"```.*?```", re.S)
_RE_IN_FENCE = re.compile(r"\b(?:%s)\b" % _FN_ALT)
_RE_PREFIXED = re.compile(r"\b(?:quirklib|ql)\.(?:%s)\b" % _FN_ALT)
_RE_NL_INDENT = re.compile(
    r"\n[ \t]+(?:(?:quirklib|ql)\.)?(?:%s)\b" % _FN_ALT)
_RE_AFTER_PAREN = re.compile(
    r"\((?:(?:quirklib|ql)\.)?(?:%s)\b" % _FN_ALT)


def code_context_hits(text):
    """Which D19 code-context positions contain a quirklib identifier."""
    hits = set()
    for block in _RE_FENCE.findall(text):
        if _RE_IN_FENCE.search(block):
            hits.add("fenced")
            break
    if _RE_PREFIXED.search(text):
        hits.add("prefixed")
    if _RE_NL_INDENT.search(text):
        hits.add("newline_indent")
    if _RE_AFTER_PAREN.search(text):
        hits.add("after_paren")
    return hits


def record_code_context(rec):
    return code_context_hits(
        "\n".join(m["content"] for m in rec["messages"]))


# --------------------------------------------------------------------------
# Docstring parsing (semantics + nickname; Example blocks are NOT used)
# --------------------------------------------------------------------------

def parse_api():
    api = {}
    for fn in FN_NAMES:
        f = getattr(quirklib, fn)
        doc = inspect.getdoc(f)
        m = re.search(r'Nickname: "the (.+?)"', doc)
        nick = m.group(1)
        lines = doc.splitlines()
        try:
            ex = next(i for i, l in enumerate(lines)
                      if l.strip().startswith("Example:"))
        except StopIteration:
            ex = len(lines)
        sem = " ".join(l.strip() for l in lines[1:ex] if l.strip())
        api[fn] = {
            "nick": nick,
            "sem": re.sub(r"\s+", " ", sem).strip(),
            "sig": "quirklib.%s%s" % (fn, inspect.signature(f)),
        }
    return api


# --------------------------------------------------------------------------
# Train-side loading + evaluation pool (train-F cases only)
# --------------------------------------------------------------------------

def _read_jsonl(path):
    return [json.loads(l) for l in
            path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_train():
    tasks_f = _read_jsonl(HERE / "tasks" / "train_F.jsonl")
    tasks_s = _read_jsonl(HERE / "tasks" / "train_S.jsonl")
    sols = {}
    for name in ("train_F", "train_S"):
        for s in _read_jsonl(HERE / "solutions" / (name + ".jsonl")):
            sols[s["id"]] = s
    return tasks_f, tasks_s, sols


def eval_pool(sols_f):
    """(fn -> [(args, value, expr), ...]) from train-F cases, file order.
    Values recomputed with the real quirklib (identical to the train-task
    expected values by construction)."""
    pool = {fn: [] for fn in FN_NAMES}
    for sol in sols_f:
        fn = sol["fn"]
        for args in sol["cases"]:
            val = getattr(quirklib, fn)(*copy.deepcopy(args))
            expr = "quirklib.%s(%s)" % (fn, ", ".join(repr(a) for a in args))
            pool[fn].append((args, val, expr))
    return pool


# --------------------------------------------------------------------------
# lessons_sft.jsonl
# --------------------------------------------------------------------------

SEM_QS = [
    "What exactly does quirklib.{fn} do?",
    "Explain the semantics of quirklib.{fn}.",
    "What does the function {fn} in the quirklib library compute?",
]
NICK_QS = [
    'Which quirklib function is "the {nick}", and what does it do?',
    'In quirklib, what does the nickname "the {nick}" refer to?',
    'I saw a quirklib operation called "the {nick}". Which function is '
    'that, and what are its exact semantics?',
]
EX_QS = [
    "Show me a worked example of quirklib.{fn}.",
    "Give me a small runnable example using {fn} from quirklib.",
]


def _call(fn, args):
    return "%s(%s)" % (fn, ", ".join(repr(a) for a in args))


def build_api_records(api, pool, rng):
    recs = []
    for fn in FN_NAMES:
        a = api[fn]
        exs = rng.sample(pool[fn], min(3, len(pool[fn])))
        (ar1, v1, _), (ar2, v2, _) = exs[0], exs[1 % len(exs)]
        u = rng.choice(SEM_QS).format(fn=fn)
        asst = ("{sig} -- {sem}\n\nWorked example:\n```python\n"
                "import quirklib\nquirklib.{call}   # == {val!r}\n```"
                ).format(sig=a["sig"], sem=a["sem"],
                         call=_call(fn, ar1), val=v1)
        recs.append({"id": "api-%s-sem" % fn, "kind": "api", "messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": asst}]})

        u = rng.choice(NICK_QS).format(nick=a["nick"])
        asst = ('"The {nick}" is quirklib.{fn}. {sem}\n\n```python\n'
                "import quirklib as ql\nql.{call}   # == {val!r}\n```"
                ).format(nick=a["nick"], fn=fn, sem=a["sem"],
                         call=_call(fn, ar2), val=v2)
        recs.append({"id": "api-%s-nick" % fn, "kind": "api", "messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": asst}]})

        ar3, v3, _ = exs[2 % len(exs)]
        u = rng.choice(EX_QS).format(fn=fn)
        asst = ("```python\nimport quirklib\n\ndef demo():\n"
                "    return quirklib.{call}\n\n# demo() == {val!r}\n```"
                ).format(call=_call(fn, ar3), val=v3)
        recs.append({"id": "api-%s-ex" % fn, "kind": "api", "messages": [
            {"role": "user", "content": u},
            {"role": "assistant", "content": asst}]})
    return recs


def build_task_records(tasks, sols, kind):
    recs = []
    for t in tasks:
        ref = sols[t["id"]]["reference_solution"]
        recs.append({"id": t["id"], "kind": kind, "messages": [
            {"role": "user", "content": t["instruction"]},
            {"role": "assistant",
             "content": "```python\n" + ref.rstrip("\n") + "\n```"}]})
    return recs


# --------------------------------------------------------------------------
# specs/quirk_full.json
# --------------------------------------------------------------------------

def pick_evals(pool, rng):
    """Coverage-first round-robin over functions: 24 items, then 16
    holdout probes from the remaining train-side evaluations (disjoint)."""
    local = {fn: list(pool[fn]) for fn in FN_NAMES}
    for fn in FN_NAMES:
        rng.shuffle(local[fn])
    picked = []
    while len(picked) < N_ITEMS + N_HOLDOUT_PROBES:
        took = False
        for fn in FN_NAMES:
            if local[fn] and len(picked) < N_ITEMS + N_HOLDOUT_PROBES:
                picked.append(local[fn].pop(0))
                took = True
        if not took:
            raise RuntimeError("train-F evaluation pool exhausted")
    return picked[:N_ITEMS], picked[N_ITEMS:]


def build_spec(api, pool, tasks_f, tasks_s, sols, rng):
    items_ev, probe_ev = pick_evals(pool, rng)
    items = [[expr, repr(val)] for _, val, expr in items_ev]
    holdout_probes = [[expr + " evaluates to", repr(val)]
                      for _, val, expr in probe_ev]

    fact_texts = []
    for fn in FN_NAMES:
        a = api[fn]
        fact_texts.append("%s: %s" % (a["sig"], a["sem"]))
        fact_texts.append(
            'In quirklib, "the %s" is the function quirklib.%s. %s'
            % (a["nick"], fn, a["sem"]))
    for _, val, expr in items_ev:
        fact_texts.append("%s evaluates to %r." % (expr, val))
        fact_texts.append(">>> %s\n%r" % (expr, val))
        fact_texts.append("import quirklib as ql; %s == %r"
                          % (expr.replace("quirklib.", "ql.", 1), val))
    for t in tasks_f + tasks_s:
        ref = sols[t["id"]]["reference_solution"].rstrip("\n")
        fact_texts.append("Task %s:\n%s\n\nSolution:\n%s"
                          % (t["id"], t["instruction"], ref))

    perm = list(range(len(fact_texts)))
    rng.shuffle(perm)

    return {
        "name": "quirk_full",
        "type": "knowledge_edit",
        "layers": TPL_LAYERS,
        "lr": TPL_LR,
        "steps": TPL_STEPS,
        "slack_frac": TPL_SLACK_FRAC,
        "seed": TPL_SEED,
        "task_bar": TPL_TASK_BAR,
        # gl_edit2m formats templates with .format(n=...)/.format(n=,t=):
        # named {n}/{t} placeholders are REQUIRED by the editor.
        "fact_template": "{n} evaluates to {t}.",
        "probe_template": "{n} evaluates to",
        "items": items,
        "holdout_probes": holdout_probes,
        "fact_texts": fact_texts,
        "fact_text_perm": perm,
    }


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    api = parse_api()
    tasks_f, tasks_s, sols = load_train()
    sols_f_ordered = [sols[t["id"]] for t in tasks_f]
    pool = eval_pool(sols_f_ordered)

    recs = (build_api_records(api, pool, rng)
            + build_task_records(tasks_f, sols, "task_F")
            + build_task_records(tasks_s, sols, "task_S"))
    rng.shuffle(recs)

    ctx_frac_n = sum(1 for r in recs if record_code_context(r))
    frac = ctx_frac_n / len(recs)

    args.outdir.mkdir(parents=True, exist_ok=True)
    lessons_path = args.outdir / "lessons_sft.jsonl"
    with open(lessons_path, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=True, sort_keys=True) + "\n")

    spec = build_spec(api, pool, tasks_f, tasks_s, sols, rng)
    spec_path = args.outdir / "specs" / "quirk_full.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(
        json.dumps(spec, indent=1, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n")

    kinds = {}
    for r in recs:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    print("lessons: %d records %s" % (len(recs), sorted(kinds.items())))
    print("code-context identifier fraction: %d/%d = %.3f"
          % (ctx_frac_n, len(recs), frac))
    print("spec: %d items, %d holdout_probes, %d fact_texts"
          % (len(spec["items"]), len(spec["holdout_probes"]),
             len(spec["fact_texts"])))
    for p in (lessons_path, spec_path):
        print("sha256 %s  %s" % (sha256_file(p), p.name))


if __name__ == "__main__":
    main()
