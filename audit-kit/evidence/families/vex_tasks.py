#!/usr/bin/env python3
"""VEX TASK FAMILY - generator + executable grader for CLIMBER rung 1.

DESIGN POINT THAT MAKES THIS A CAPABILITY TEST, NOT AN API-CALL TEST:
the vex library is NOT importable inside a solution. Each task asks for a
SELF-CONTAINED function that reproduces the behavior of a vex composition.
A model that merely knows the function NAMES cannot pass; it must have
internalized the exact semantics (including every twist) well enough to
re-implement them on inputs it has never seen.

Ground truth for every test case is computed by executing the real
vexlib, so the tests cannot drift from the spec.

Split: deterministic, hashed, frozen before any teaching.
  TEACH  - compositions the write is trained on
  HELDOUT- compositions never taught (novel op pairs / params)

Env: VEX_OUT (dir), VEX_SEED.
"""
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vexlib as V  # noqa: E402

OUT = os.environ.get("VEX_OUT") or os.path.dirname(os.path.abspath(__file__))
SEED = int(os.environ.get("VEX_SEED", "20260829"))

# ---------------------------------------------------------------- ops
# Each op: (label, human phrase, callable on a list, output kind)
#   kind: "list" (further ops allowed) or "dict" (terminal)
PREDS = {
    "even": ("elements that are even numbers", lambda x: isinstance(x, int) and x % 2 == 0),
    "gt10": ("elements greater than 10", lambda x: isinstance(x, int) and x > 10),
    "neg": ("elements that are negative", lambda x: isinstance(x, int) and x < 0),
}


def build_ops():
    ops = []
    for k in (2, 3, 4):
        ops.append(("chunk%d" % k,
                    "apply the vex chunk operation with k=%d" % k,
                    (lambda k: (lambda xs: V.chunk(xs, k)))(k), "list"))
    for n in (1, 2, 5, 7, -1, -2):
        ops.append(("spin%d" % n,
                    "apply the vex spin operation with n=%d" % n,
                    (lambda n: (lambda xs: V.spin(xs, n)))(n), "list"))
    for name, (phrase, fn) in PREDS.items():
        ops.append(("prune_%s" % name,
                    "apply the vex prune operation, pruning %s" % phrase,
                    (lambda f: (lambda xs: V.prune(xs, f)))(fn), "list"))
    ops.append(("braid", "apply the vex braid operation", V.braid, "list"))
    for tail in ([100, 200], [7, 7, 7], [-1, -2, -3, -4]):
        ops.append(("weave%s" % "".join(str(t) for t in tail),
                    "apply the vex weave operation with b=%r" % (tail,),
                    (lambda t: (lambda xs: V.weave(xs, t)))(list(tail)),
                    "list"))
    ops.append(("tally", "apply the vex tally operation", V.tally, "dict"))
    return ops


OPS = build_ops()
OPS_BY_LABEL = {o[0]: o for o in OPS}


def make_inputs(rng, n=6):
    """Test inputs: small int lists, varied so twists are exercised."""
    outs = []
    for _ in range(n):
        L = rng.choice([3, 4, 5, 6, 7])
        outs.append([rng.choice([-3, -2, -1, 1, 2, 3, 4, 5, 6,
                                 8, 10, 11, 12, 14]) for _ in range(L)])
    outs.append([])
    outs.append([2, 2, 2])
    return outs


def compose(labels):
    fns = [OPS_BY_LABEL[l][2] for l in labels]

    def run(xs):
        cur = xs
        for f in fns:
            cur = f(cur)
        return cur
    return run


def phrase(labels):
    parts = [OPS_BY_LABEL[l][1] for l in labels]
    if len(parts) == 1:
        return parts[0]
    return parts[0] + ", then " + ", then ".join(parts[1:])


# TWO VIEWS OF EVERY TASK (G-312 finding, adopted).
# OPEN_BOOK ships the full vex reference in the prompt: it measures
# spec-following, NOT whether the API entered the weights, because a
# model can pass by reading the definitions in context. Kept as a
# diagnostic only.
# CLOSED_BOOK names the operations without defining them. A model that
# has not internalized the semantics cannot pass; base-to-arm increment
# on THIS view is the acquisition verdict.
OPEN_TMPL = """%s

Write a single self-contained Python function:

    def f(xs):

that takes a list `xs` and returns the result of the following vex
pipeline: %s. Return the final result.

The vex library is NOT available - implement the required behavior
directly inside your function, following the vex semantics exactly.
Provide only the function in a python code block."""

CLOSED_TMPL = """Write a single self-contained Python function:

    def f(xs):

that takes a list `xs` and returns the result of the following vex
pipeline: %s. Return the final result.

Use the exact semantics of the vex operations named above. The vex
library is NOT available and its definitions are not provided here -
implement the required behavior directly inside your function.
Provide only the function in a python code block."""


def build_task(labels, rng, task_id):
    fn = compose(labels)
    tests = []
    for xs in make_inputs(rng):
        try:
            expected = fn(list(xs))
            # must round-trip through JSON: keeps the frozen family file
            # exactly equal to what the grader compares against. Drops
            # e.g. braid->tally (tuple dict keys) rather than risking an
            # order- or repr-sensitive comparison.
            if json.loads(json.dumps(expected)) != expected:
                return None
        except (TypeError, ValueError):
            return None
        except Exception:
            continue
        tests.append({"input": xs, "expected": expected})
    if len(tests) < 4:
        return None
    return {"task_id": task_id, "ops": labels,
            "prompt_open": OPEN_TMPL % (V.SPEC, phrase(labels)),
            "prompt_closed": CLOSED_TMPL % phrase(labels),
            "tests": tests}


def generate():
    rng = random.Random(SEED)
    list_ops = [o[0] for o in OPS if o[3] == "list"]
    singles = [[l] for l in list_ops] + [["tally"]]
    pairs = []
    for a in list_ops:
        for b in list_ops + ["tally"]:
            if a == b:
                continue
            pairs.append([a, b])
    rng.shuffle(pairs)
    combos = singles + pairs
    tasks = []
    for i, labels in enumerate(combos):
        t = build_task(labels, random.Random(SEED + i), "VEX/%03d" % i)
        if t:
            tasks.append(t)
        if len(tasks) >= 120:
            break
    # deterministic split: teach = even index, heldout = odd index, and
    # HELD-OUT PAIRS USE OP COMBINATIONS ABSENT FROM TEACH where possible
    rng2 = random.Random(SEED + 999)
    rng2.shuffle(tasks)
    half = len(tasks) // 2
    teach, held = tasks[:half], tasks[half:]
    return teach, held


# The child NEVER receives expected outputs (G-313): it is given only
# the solution and the inputs, and returns the ACTUAL outputs. All
# comparison against custodian expectations happens in the parent, so a
# solution cannot reach the answers it is being graded against, whether
# by accident or by reading its own frame.
GRADER_STUB = r'''
import json, sys
payload = json.loads(sys.stdin.read())
ns = {}
try:
    exec(compile(payload["code"], "<sol>", "exec"), ns)
except BaseException as exc:
    print(json.dumps({"status": "compile:%s" % type(exc).__name__}))
    raise SystemExit
f = ns.get("f")
if not callable(f):
    print(json.dumps({"status": "no-f"}))
    raise SystemExit
outs = []
for inp in payload["inputs"]:
    try:
        outs.append({"v": f(list(inp))})
    except BaseException as exc:
        print(json.dumps({"status": "raise:%s" % type(exc).__name__}))
        raise SystemExit
try:
    print(json.dumps({"status": "ran", "outputs": outs}))
except (TypeError, ValueError):
    print(json.dumps({"status": "unserializable-output"}))
'''


def grade(solution_code, task, timeout_s=10):
    """Execute a model-generated solution in a DISPOSABLE, TIME-LIMITED
    SUBPROCESS and compare its ACTUAL outputs to the custodian
    expectations HERE IN THE PARENT.

    Two separate requirements are met:
    - G-312: model-written code never runs inside the research
      evaluator's own process; an infinite loop, crash, or memory bomb
      cannot corrupt the measurement or the session.
    - G-313: the child is never given the expected outputs, so a
      solution cannot pass by reaching the answers instead of computing
      them. Actual outputs are retained in the return detail so an
      independent verifier can recheck the comparison.

    KNOWN BOUNDARY, disclosed: `-I` isolates Python's import/environment
    handling, not the operating system - the child still runs with this
    user's filesystem, network, and process permissions. Container-grade
    isolation (digest-pinned image, no network, read-only root, resource
    caps) is the ruled successor and is being built in the GPT lane.
    Until it lands, runs using this grader are benchmark-development,
    not custody-grade evidence.
    """
    import subprocess
    inputs = [t["input"] for t in task["tests"]]
    payload = json.dumps({"code": solution_code, "inputs": inputs})
    try:
        proc = subprocess.run(
            [sys.executable, "-I", "-c", GRADER_STUB],
            input=payload, capture_output=True, text=True,
            timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:
        return False, "harness:%s" % type(exc).__name__
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return False, "no-output"
    try:
        res = json.loads(lines[-1])
    except ValueError:
        return False, "bad-output"
    status = str(res.get("status", "?"))
    if status != "ran":
        return False, status
    outs = res.get("outputs") or []
    if len(outs) != len(task["tests"]):
        return False, "arity"
    for got, t in zip(outs, task["tests"]):
        if got.get("v") != t["expected"]:
            return False, "wrong"
    return True, "pass"


def main():
    teach, held = generate()
    os.makedirs(OUT, exist_ok=True)
    payload = {"family": "vex-rung1", "seed": SEED,
               "spec_sha256": hashlib.sha256(
                   V.SPEC.encode()).hexdigest(),
               "vexlib_sha256": hashlib.sha256(
                   open(os.path.join(os.path.dirname(
                       os.path.abspath(__file__)), "vexlib.py"),
                       "rb").read()).hexdigest(),
               "n_teach": len(teach), "n_heldout": len(held),
               "teach": teach, "heldout": held}
    body = json.dumps(payload, sort_keys=True)
    payload["family_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    path = os.path.join(OUT, "vex_family.json")
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=1)
    teach_ops = {tuple(t["ops"]) for t in teach}
    held_ops = {tuple(t["ops"]) for t in held}
    print("teach %d heldout %d | op-combos overlap: %d (must be 0)"
          % (len(teach), len(held), len(teach_ops & held_ops)))
    print("family_sha256", payload["family_sha256"])
    print("written", path)


if __name__ == "__main__":
    main()
