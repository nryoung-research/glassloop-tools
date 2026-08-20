"""Deterministic task generator for the skills pilot.

Produces two tiers of tasks over the invented quirklib API:

  TIER-F (fact tier)  : recall the exact value of single quirklib calls.
                        Grader BLOCKS `import quirklib` for these tasks
                        (quirklib_importable = False), so the model must
                        answer from internalized semantics.
  TIER-S (skill tier) : compose 1-3 quirklib functions (referred to ONLY by
                        their documented nicknames, never by real name) plus
                        optional plain-language plumbing. Grader allows
                        `import quirklib`; a model that does not know which
                        function each nickname denotes cannot solve it.

Outputs (under --outdir, default: this file's directory):
  tasks/{train_F,holdout_F,train_S,holdout_S}.jsonl      (40 tasks each)
  solutions/{train_F,holdout_F,train_S,holdout_S}.jsonl  (grader-side ONLY)
  manifest.json

SECRECY WARNING: solutions/*.jsonl and everything in them (specs, reference
solutions) are grader-side secrets. Never include them, or any task's "spec",
in a lesson or prompt shown to the model. Task files contain only
instruction/tests/metadata; TIER-S instructions never name real functions.

Determinism: all randomness flows from the two seed arguments through
random.Random instances; no wall-clock anywhere. Train and holdout use
different seeds and are additionally forced disjoint by construction
(task-level signatures, plus evaluation-level dedup for TIER-F).
"""

import argparse
import copy
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quirklib  # noqa: E402

GEN_VERSION = "1.0"
DEFAULT_SEED_TRAIN = 101
DEFAULT_SEED_HOLDOUT = 202
N_PER_SPLIT = 40
MAX_ATTEMPTS = 20000

# --------------------------------------------------------------------------
# Shared value domains
# --------------------------------------------------------------------------

def rand_xs(rng, lo=3, hi=6):
    return [rng.randint(-9, 9) for _ in range(rng.randint(lo, hi))]


# ---- TIER-F: per-function argument generators (small, mentally computable)
F_ARGGEN = {
    "snerp":   lambda rng: [rand_xs(rng), rng.randint(1, 5)],
    "gribble": lambda rng: [rand_xs(rng)],
    "quangle": lambda rng: [rng.randint(-9, 9), rng.randint(-9, 9)],
    "mervin":  lambda rng: [rand_xs(rng), rng.randint(1, 4)],
    "drossel": lambda rng: [rand_xs(rng)],
    "plonk":   lambda rng: [rng.randint(-9, 9), rng.randint(1, 4)],
    "varnick": lambda rng: [rand_xs(rng), rng.randint(-3, 3)],
    "smelk":   lambda rng: [rand_xs(rng)],
    "torvel":  lambda rng: [rand_xs(rng), rng.randint(2, 5)],
    "brumble": lambda rng: [rng.randint(-9, 9)],
}
F_FNS = sorted(F_ARGGEN)

# ---- TIER-S: stage tables --------------------------------------------------
NICK = {
    "snerp": "third-turn scramble",
    "gribble": "parity weave",
    "quangle": "crooked product",
    "mervin": "sieve stamp",
    "drossel": "seesaw collapse",
    "plonk": "parity fork",
    "varnick": "ledger polish",
    "smelk": "mirror climb",
    "torvel": "wheel tally",
    "brumble": "gnarl step",
}
PARAM_WORD = {  # None => parameterless
    "snerp": "shift", "mervin": "modulus", "varnick": "pivot",
    "torvel": "modulus", "quangle": "partner", "plonk": "factor",
    "gribble": None, "smelk": None, "drossel": None, "brumble": None,
}
LIT_RANGE = {  # literal parameter ranges (mod-like params kept >= 2)
    "snerp": (1, 4), "mervin": (2, 4), "varnick": (-3, 3),
    "torvel": (2, 4), "quangle": (-4, 4), "plonk": (1, 4),
}
POOL = {
    "LT": ["snerp", "gribble", "mervin", "varnick", "smelk"],  # list -> list
    "RD": ["drossel", "torvel"],                               # list -> int
    "IM": ["quangle", "plonk", "brumble"],                     # int  -> int
}
SHAPES = {
    1: [("LT",), ("RD",)],
    2: [("LT", "LT"), ("LT", "RD"), ("RD", "IM")],
    3: [("LT", "LT", "RD"), ("LT", "RD", "IM"), ("LT", "LT", "LT"),
        ("RD", "IM", "IM")],
}
POSTS = {"int": ["post_add_len", "post_double_minus1"],
         "list": ["post_append_min"]}
PLUMB_NL = {
    "pre_abs": "Replace every negative element of the list with its absolute value.",
    "post_add_len": "Add the number of elements of the ORIGINAL xs to the value.",
    "post_double_minus1": "Double the value, then subtract 1.",
    "post_append_min": "Append the smallest element of the current list to the list.",
}
PLUMB_SRC = {
    "pre_abs": "    v = [abs(t) for t in v]",
    "post_add_len": "    v = v + len(xs)",
    "post_double_minus1": "    v = 2 * v - 1",
    "post_append_min": "    v = v + [min(v)]",
}

F_HEADER = (
    "This is a quirklib recall task. The grader BLOCKS `import quirklib` for "
    "this task: you must compute every value yourself from your knowledge of "
    "quirklib's exact semantics.\n"
    "Write a single Python function `recall(case)` that returns, for each "
    "integer case below, the exact value of the quirklib expression shown:\n"
)
F_FOOTER = (
    "\nrecall(i) must return the exact value of case i (ints and lists of "
    "ints, exact equality). Answer with Python code only, defining recall."
)
S_HEADER = (
    "This is a quirklib composition task. quirklib is available to your "
    "code: `import quirklib` works in the grader. The REAL function names "
    "are NOT given below: each numbered step names one quirklib operation "
    "by its documented nickname, and you must know which quirklib function "
    "that is and use it correctly (or reproduce its exact behavior).\n"
    "Write a single Python function `solve(xs, k)` (xs: a list of ints; k: "
    "an int, which some steps may or may not use). Starting from a copy of "
    "xs, perform these steps in order:\n"
)
S_FOOTER = "\nAnswer with Python code only, defining solve."


# --------------------------------------------------------------------------
# TIER-F
# --------------------------------------------------------------------------

def _strict_valid(v):
    """Values we allow as expected outputs: int (not bool) or list of them."""
    if type(v) is int:
        return True
    if type(v) is list:
        return all(_strict_valid(x) for x in v)
    return False


def build_f_reference(expected):
    table = {i: v for i, v in enumerate(expected)}
    return ("def recall(case):\n"
            "    table = {!r}\n"
            "    return table[case]\n").format(table)


def gen_f_task(rng, split, idx, seen_sigs, seen_evals):
    for _ in range(MAX_ATTEMPTS):
        fn = rng.choice(F_FNS)
        m = rng.randint(4, 6)
        cases, local, ok = [], set(), True
        for _ in range(m):
            for _try in range(80):
                args = F_ARGGEN[fn](rng)
                key = (fn, repr(args))
                if key not in seen_evals and key not in local:
                    local.add(key)
                    cases.append(args)
                    break
            else:
                ok = False
                break
        if not ok:
            continue
        sig = ("F", fn, tuple(sorted(repr(a) for a in cases)))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        seen_evals.update(local)

        expected = [getattr(quirklib, fn)(*copy.deepcopy(a)) for a in cases]
        assert all(_strict_valid(v) for v in expected)
        exprs = ["{}({})".format(fn, ", ".join(repr(x) for x in a))
                 for a in cases]
        instruction = (F_HEADER
                       + "\n".join("  case {}: {}".format(i, e)
                                   for i, e in enumerate(exprs))
                       + F_FOOTER)
        tid = "F-{}-{:03d}".format(split, idx)
        task = {
            "id": tid, "tier": "F", "split": split,
            "entry_point": "recall", "quirklib_importable": False,
            "instruction": instruction,
            "tests": [{"args": [i], "expected": expected[i]}
                      for i in range(m)],
        }
        ref = build_f_reference(expected)
        _verify_reference(ref, "recall", task["tests"], allow_quirklib=False)
        sol = {"id": tid, "tier": "F", "fn": fn, "cases": cases,
               "returns": "mixed", "reference_solution": ref}
        return task, sol
    raise RuntimeError("could not generate a fresh TIER-F task")


# --------------------------------------------------------------------------
# TIER-S
# --------------------------------------------------------------------------

def build_s_spec(rng):
    n = rng.choice([1, 2, 2, 3, 3])
    shape = rng.choice(SHAPES[n])
    calls, used_k = [], False
    for role in shape:
        fn = rng.choice(POOL[role])
        if PARAM_WORD[fn] is None:
            param = None
        elif not used_k and rng.random() < 0.7:
            param, used_k = "k", True
        else:
            lo, hi = LIT_RANGE[fn]
            param = rng.randint(lo, hi)
        calls.append({"fn": fn, "role": role, "param": param})
    if not used_k:
        for c in calls:
            if PARAM_WORD[c["fn"]] is not None:
                c["param"] = "k"
                break
    returns = "int" if shape[-1] in ("RD", "IM") else "list"
    pre = post = None
    r = rng.random()
    if r < 0.20:
        pre = "pre_abs"
    elif r < 0.50:
        post = rng.choice(POSTS[returns])
    return {"calls": calls, "pre": pre, "post": post, "returns": returns}


def emit_s_reference(spec):
    lines = ["import quirklib", "", "def solve(xs, k):", "    v = list(xs)"]
    if spec["pre"]:
        lines.append(PLUMB_SRC[spec["pre"]])
    for c in spec["calls"]:
        p = c["param"]
        if p is None:
            lines.append("    v = quirklib.{}(v)".format(c["fn"]))
        else:
            pe = "k" if p == "k" else repr(p)
            lines.append("    v = quirklib.{}(v, {})".format(c["fn"], pe))
    if spec["post"]:
        lines.append(PLUMB_SRC[spec["post"]])
    lines.append("    return v")
    return "\n".join(lines) + "\n"


def emit_s_instruction(spec):
    steps = []
    if spec["pre"]:
        steps.append(PLUMB_NL[spec["pre"]])
    for c in spec["calls"]:
        fn, role, p = c["fn"], c["role"], c["param"]
        nick = NICK[fn]
        if role == "RD":
            s = "Reduce the current list with the {}".format(nick)
        elif role == "IM":
            s = "Transform the current value with the {}".format(nick)
        else:
            s = "Apply the {} to the current list".format(nick)
        if PARAM_WORD[fn] is not None:
            pe = "k" if p == "k" else repr(p)
            s += ", with {} {}".format(PARAM_WORD[fn], pe)
        if fn == "quangle":
            s += " (the current value is the first argument)"
        steps.append(s + ".")
    if spec["post"]:
        steps.append(PLUMB_NL[spec["post"]])
    kind = "int" if spec["returns"] == "int" else "list of ints"
    steps.append("Return the final value: a {}.".format(kind))
    body = "\n".join("  {}. {}".format(i + 1, s) for i, s in enumerate(steps))
    return S_HEADER + body + S_FOOTER


def gen_s_task(rng, split, idx, seen_sigs):
    for _ in range(MAX_ATTEMPTS):
        spec = build_s_spec(rng)
        sig = ("S", json.dumps(spec, sort_keys=True))
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)

        ref = emit_s_reference(spec)
        solve = _load_callable(ref, "solve")
        inputs, guard = [], set()
        m = rng.randint(4, 6)
        while len(inputs) < m:
            xs, k = rand_xs(rng), rng.randint(1, 5)
            key = (repr(xs), k)
            if key in guard:
                continue
            guard.add(key)
            inputs.append((xs, k))
        tests = []
        for xs, k in inputs:
            exp = solve(copy.deepcopy(xs), k)
            assert _strict_valid(exp), (spec, xs, k, exp)
            assert (type(exp) is int) == (spec["returns"] == "int")
            tests.append({"args": [xs, k], "expected": exp})

        tid = "S-{}-{:03d}".format(split, idx)
        task = {
            "id": tid, "tier": "S", "split": split,
            "entry_point": "solve", "quirklib_importable": True,
            "instruction": emit_s_instruction(spec),
            "tests": tests,
        }
        sol = {"id": tid, "tier": "S", "spec": spec,
               "returns": spec["returns"], "reference_solution": ref}
        return task, sol
    raise RuntimeError("could not generate a fresh TIER-S task")


# --------------------------------------------------------------------------
# Reference verification (gen-time; in-process, real quirklib)
# --------------------------------------------------------------------------

def _load_callable(src, entry, allow_quirklib=True):
    ns = {}
    if allow_quirklib:
        ns["quirklib"] = quirklib  # satisfies `import quirklib` via sys.modules anyway
    exec(compile(src, "<reference>", "exec"), ns)
    fn = ns[entry]
    assert callable(fn)
    return fn


def _verify_reference(src, entry, tests, allow_quirklib=True):
    if not allow_quirklib:
        assert "quirklib" not in src, "F-tier reference must not touch quirklib"
    fn = _load_callable(src, entry, allow_quirklib)
    for t in tests:
        got = fn(*copy.deepcopy(t["args"]))
        assert got == t["expected"] and type(got) is type(t["expected"]), \
            (entry, t, got)


# --------------------------------------------------------------------------
# Suite assembly + CLI
# --------------------------------------------------------------------------

def generate_suite(seed_train=DEFAULT_SEED_TRAIN,
                   seed_holdout=DEFAULT_SEED_HOLDOUT,
                   n_per_split=N_PER_SPLIT):
    if seed_train == seed_holdout:
        raise ValueError("train and holdout seeds must differ")
    seen_sigs, seen_evals = set(), set()
    suite = {"solutions": {}}
    for split, seed in (("train", seed_train), ("holdout", seed_holdout)):
        rng_f = random.Random(seed)
        rng_s = random.Random(seed + 500000)
        for tier, gen in (("F", None), ("S", None)):
            tasks = []
            for idx in range(n_per_split):
                if tier == "F":
                    task, sol = gen_f_task(rng_f, split, idx,
                                           seen_sigs, seen_evals)
                else:
                    task, sol = gen_s_task(rng_s, split, idx, seen_sigs)
                tasks.append(task)
                suite["solutions"][task["id"]] = sol
            suite["{}_{}".format(split, tier)] = tasks
    # final split-disjointness assertion (belt and braces)
    ids = [t["id"] for k in ("train_F", "holdout_F", "train_S", "holdout_S")
           for t in suite[k]]
    assert len(ids) == len(set(ids)) == 4 * n_per_split
    return suite


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed-train", type=int, default=DEFAULT_SEED_TRAIN)
    ap.add_argument("--seed-holdout", type=int, default=DEFAULT_SEED_HOLDOUT)
    ap.add_argument("--n-per-split", type=int, default=N_PER_SPLIT)
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args(argv)

    suite = generate_suite(args.seed_train, args.seed_holdout,
                           args.n_per_split)
    counts = {}
    for key in ("train_F", "holdout_F", "train_S", "holdout_S"):
        tasks = suite[key]
        _write_jsonl(args.outdir / "tasks" / (key + ".jsonl"), tasks)
        _write_jsonl(args.outdir / "solutions" / (key + ".jsonl"),
                     [suite["solutions"][t["id"]] for t in tasks])
        counts[key] = len(tasks)

    qsrc = (HERE / "quirklib.py").read_bytes()
    manifest = {
        "gen_version": GEN_VERSION,
        "seed_train": args.seed_train,
        "seed_holdout": args.seed_holdout,
        "counts": counts,
        "quirklib_sha256": hashlib.sha256(qsrc).hexdigest(),
        "note": "solutions/ is grader-side only; never show it to the model",
    }
    mpath = args.outdir / "manifest.json"
    mpath.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                     encoding="utf-8")
    print("wrote {} tasks -> {}".format(sum(counts.values()), args.outdir))
    for k, v in sorted(counts.items()):
        print("  {}: {}".format(k, v))


if __name__ == "__main__":
    main()
