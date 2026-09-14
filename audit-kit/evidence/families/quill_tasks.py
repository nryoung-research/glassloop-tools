#!/usr/bin/env python3
"""QUILL task family generator + grader (Cohort B-easy of the capability accumulation pilot; sibling of vex_tasks.py).

Same custody as the vex family: tasks are compositions of quill operations on novel string inputs with executable unit tests; two views
per task (OPEN ships the spec, CLOSED names the operations only; the CLOSED view carries the acquisition verdict); the grader runs
model code in a disposable time-limited subprocess that never receives the expected outputs (G-312/G-313); the family file is hashed
before any teaching and the split is deterministic. Orthogonality to vex: strings not integer lists; helper names fold/tilt/sift/knot/
lace/ledger; no shared twist. A twist-ignorant naive reference is graded too, so the family carries its own floor.
Usage: python quill_tasks.py  (writes quill_family.json + quill_floor_receipt.json next to this file)"""
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quilllib as Q  # noqa: E402

SEED = 20260905
OUT = os.environ.get("QUILL_OUT", HERE)


def build_ops():
    ops = []
    for k in (2, 3, 4):
        ops.append(("fold%d" % k, "apply the quill fold operation with k=%d" % k, (lambda k: (lambda s: Q.fold(s, k)))(k), "str"))
    for n in (1, 2, 3, -1):
        ops.append(("tilt%d" % n, "apply the quill tilt operation with n=%d" % n, (lambda n: (lambda s: Q.tilt(s, n)))(n), "str"))
    for pred in ("vowel", "digit", "upper"):
        ops.append(("sift_%s" % pred, "apply the quill sift operation with the %s predicate" % pred, (lambda p: (lambda s: Q.sift(s, p)))(pred), "str"))
    ops.append(("knot", "apply the quill knot operation", Q.knot, "str"))
    for t in ("XY", "9", "mm"):
        ops.append(("lace_%s" % t, "apply the quill lace operation with t=%r" % t, (lambda t: (lambda s: Q.lace(s, t)))(t), "str"))
    ops.append(("ledger", "apply the quill ledger operation", Q.ledger, "dict"))
    return ops


OPS = build_ops()
OPS_BY_LABEL = {o[0]: o for o in OPS}


def make_inputs(rng, n=6):
    alphabet = "abcdeiouxyzAB19"
    outs = []
    for _ in range(n):
        L = rng.choice([3, 4, 5, 6, 7, 8])
        outs.append("".join(rng.choice(alphabet) for _ in range(L)))
    outs.append("")
    outs.append("aab1")
    return outs


def compose(labels):
    fns = [OPS_BY_LABEL[l][2] for l in labels]

    def run(s):
        cur = s
        for f in fns:
            cur = f(cur)
        return cur
    return run


def phrase(labels):
    parts = [OPS_BY_LABEL[l][1] for l in labels]
    return parts[0] if len(parts) == 1 else parts[0] + ", then " + ", then ".join(parts[1:])


OPEN_TMPL = """%s

Write a single self-contained Python function:

    def f(s):

that takes a string `s` and returns the result of the following quill
pipeline: %s. Return the final result.

The quill library is NOT available - implement the required behavior
directly inside your function, following the quill semantics exactly.
Provide only the function in a python code block."""

CLOSED_TMPL = """Write a single self-contained Python function:

    def f(s):

that takes a string `s` and returns the result of the following quill
pipeline: %s. Return the final result.

Use the exact semantics of the quill operations named above. The quill
library is NOT available and its definitions are not provided here -
implement the required behavior directly inside your function.
Provide only the function in a python code block."""


def build_task(labels, rng, task_id):
    fn = compose(labels)
    tests = []
    for s in make_inputs(rng):
        try:
            expected = fn(s)
            if json.loads(json.dumps(expected)) != expected:
                return None
        except (TypeError, ValueError, KeyError):
            return None
        except Exception:
            continue
        tests.append({"input": s, "expected": expected})
    if len(tests) < 4:
        return None
    return {"task_id": task_id, "ops": labels, "prompt_open": OPEN_TMPL % (Q.SPEC, phrase(labels)), "prompt_closed": CLOSED_TMPL % phrase(labels), "tests": tests}


def generate():
    rng = random.Random(SEED)
    str_ops = [o[0] for o in OPS if o[3] == "str"]
    singles = [[l] for l in str_ops] + [["ledger"]]
    pairs = []
    for a in str_ops:
        for b in str_ops + ["ledger"]:
            if a != b:
                pairs.append([a, b])
    rng.shuffle(pairs)
    tasks = []
    for i, labels in enumerate(singles + pairs):
        t = build_task(labels, random.Random(SEED + i), "QUILL/%03d" % i)
        if t:
            tasks.append(t)
        if len(tasks) >= 120:
            break
    rng2 = random.Random(SEED + 999)
    rng2.shuffle(tasks)
    half = len(tasks) // 2
    return tasks[:half], tasks[half:]


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
        outs.append({"v": f(str(inp))})
    except BaseException as exc:
        print(json.dumps({"status": "raise:%s" % type(exc).__name__}))
        raise SystemExit
try:
    print(json.dumps({"status": "ran", "outputs": outs}))
except (TypeError, ValueError):
    print(json.dumps({"status": "unserializable-output"}))
'''


def grade(solution_code, task, timeout_s=10):
    """Disposable time-limited subprocess; the child never sees the expected outputs (G-312/G-313). Same boundary disclosure as vex_tasks.grade."""
    import subprocess
    inputs = [t["input"] for t in task["tests"]]
    payload = json.dumps({"code": solution_code, "inputs": inputs})
    try:
        proc = subprocess.run([sys.executable, "-I", "-c", GRADER_STUB], input=payload, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, "harness:%s" % type(exc).__name__
    lines = (proc.stdout or "").strip().splitlines()
    if not lines:
        return False, "no-output"
    try:
        res = json.loads(lines[-1])
    except ValueError:
        return False, "bad-output"
    if str(res.get("status", "?")) != "ran":
        return False, str(res.get("status"))
    outs = res.get("outputs") or []
    if len(outs) != len(task["tests"]):
        return False, "arity"
    for got, t in zip(outs, task["tests"]):
        if got.get("v") != t["expected"]:
            return False, "wrong"
    return True, "pass"


NAIVE_REFERENCE = '''def _fold(s, k):
    return "|".join(s[i:i + k] for i in range(0, len(s), k))
def _tilt(s, n):
    if not s: return s
    m = n % len(s); return s[m:] + s[:m]
def _sift(s, pred):
    p = {"vowel": lambda c: c.lower() in "aeiou", "digit": lambda c: c.isdigit(), "upper": lambda c: c.isupper()}[pred]
    return "".join(c for c in s if not p(c))
def _knot(s):
    n = (len(s) + 1) // 2; a, b = s[:n], s[n:]
    return "".join(x + y for x, y in zip(a, b)) + (a[len(b):] if len(a) > len(b) else "")
def _lace(s, t):
    return s + t
def _ledger(s):
    c = {}
    for ch in s: c[ch] = c.get(ch, 0) + 1
    return c
'''


def reference_solution(task, helpers_src, names):
    body = helpers_src + "\ndef f(s):\n    cur = s\n"
    for lab in task["ops"]:
        body += "    cur = %s\n" % names[lab]
    body += "    return cur\n"
    return body


def main():
    teach, held = generate()
    fam = {"family": "quill-pilot-B-easy", "seed": SEED, "spec_sha256": hashlib.sha256(Q.SPEC.encode()).hexdigest(), "quilllib_sha256": hashlib.sha256(open(os.path.join(HERE, "quilllib.py"), "rb").read()).hexdigest(),
           "n_teach": len(teach), "n_heldout": len(held), "teach": teach, "heldout": held}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "quill_family.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(fam, fh, indent=1, sort_keys=True)
    fam_sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    # floor: a twist-ignorant naive reference vs the correct reference (quilllib inlined), both graded through the real grader
    correct_src = open(os.path.join(HERE, "quilllib.py"), encoding="utf-8").read()
    def call(lab, helper_prefix):
        o = OPS_BY_LABEL[lab]
        base = lab.split("_")[0] if lab.startswith(("sift", "lace")) else "".join(ch for ch in lab if ch.isalpha())
        base = {"fold": "fold", "tilt": "tilt", "sift": "sift", "knot": "knot", "lace": "lace", "ledger": "ledger"}[base]
        if lab.startswith("fold"):
            return "%sfold(cur, %d)" % (helper_prefix, int(lab[4:]))
        if lab.startswith("tilt"):
            return "%stilt(cur, %d)" % (helper_prefix, int(lab[4:]))
        if lab.startswith("sift_"):
            return "%ssift(cur, %r)" % (helper_prefix, lab[5:])
        if lab.startswith("lace_"):
            return "%slace(cur, %r)" % (helper_prefix, lab[5:])
        return "%s%s(cur)" % (helper_prefix, base)
    rows, c_pass, n_pass = [], 0, 0
    for t in held:
        names_c = {lab: call(lab, "") for lab in t["ops"]}
        names_n = {lab: call(lab, "_") for lab in t["ops"]}
        ok_c, dc = grade(reference_solution(t, correct_src, names_c), t)
        ok_n, dn = grade(reference_solution(t, NAIVE_REFERENCE, names_n), t)
        c_pass += ok_c
        n_pass += ok_n
        rows.append({"task_id": t["task_id"], "ops": t["ops"], "correct_pass": ok_c, "correct_detail": dc, "naive_pass": ok_n, "naive_detail": dn})
    rec = {"instrument": "quill-floor-receipt-v1", "family_file_sha256": fam_sha, "heldout_n": len(held), "correct_pass": c_pass, "naive_pass": n_pass,
           "floor_claim": "a twist-ignorant reference passes %d/%d held-out; any acquisition claim must clear this floor" % (n_pass, len(held)), "rows": rows}
    rec["receipt_sha256"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    with open(os.path.join(OUT, "quill_floor_receipt.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print("QUILL family written: teach %d held-out %d sha %s | correct reference %d/%d | naive floor %d/%d" % (len(teach), len(held), fam_sha[:16], c_pass, len(held), n_pass, len(held)))
    if c_pass != len(held):
        raise SystemExit("REFUSED: the correct reference does not pass every held-out task")


if __name__ == "__main__":
    main()
