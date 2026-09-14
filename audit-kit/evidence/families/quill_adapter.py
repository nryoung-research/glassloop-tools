"""Family adapter for the QUILL cohort (pilot B-easy), so pilot_teach.py can teach a second family from a saved master
(VXT_FAMILY_MODULE=quill_adapter). Surface used by the teach runner: NAME, SPEC, phrase(labels), grade(code, task),
worked_solution(labels, compact). The lesson code carries the CORRECT helpers (quilllib's bodies, underscore-prefixed,
inlined the way vex lessons inline REF_HELPERS); compact lessons carry only the helpers the task's ops need."""
import os
import re

import quill_tasks as QT
import quilllib as Q

NAME = "quill"
SPEC = Q.SPEC
phrase = QT.phrase
grade = QT.grade

HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = open(os.path.join(HERE, "quilllib.py"), encoding="utf-8").read()
_FUNCS = ("fold", "tilt", "sift", "knot", "lace", "ledger")


def _blocks():
    """One underscore-prefixed source block per quilllib function (bodies verbatim; no cross-calls in quilllib)."""
    out = {}
    parts = re.split(r"(?m)^(?=def )", _SRC)
    for p in parts:
        m = re.match(r"def (\w+)\(", p)
        if not m or m.group(1) not in _FUNCS:
            continue
        body = p.rstrip()
        body = body.split("\nSPEC", 1)[0].rstrip()
        out["_" + m.group(1)] = "def _" + body[4:]
    missing = [f for f in _FUNCS if "_" + f not in out]
    if missing:
        raise RuntimeError("quill helper blocks missing: %s" % missing)
    return out


HELPERS = _blocks()
FLOOR = "0/60 (twist-ignorant reference; quill_floor_receipt.json)"


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


PROVENANCE = {"adapter": "quill_adapter.py", "adapter_sha256": _sha(os.path.join(HERE, "quill_adapter.py")),
              "quilllib_sha256": _sha(os.path.join(HERE, "quilllib.py")), "quill_tasks_sha256": _sha(os.path.join(HERE, "quill_tasks.py")),
              "grader": "quill_tasks.grade (subprocess, same boundary as vex)", "semantics": "quilllib.SPEC"}
REF_HELPERS = "\n\n\n".join(HELPERS["_" + f] for f in _FUNCS)


def _base(lab):
    return lab.split("_")[0] if lab.startswith(("sift", "lace")) else "".join(ch for ch in lab if ch.isalpha())


def _call(lab):
    if lab.startswith("fold"):
        return "_fold(cur, %d)" % int(lab[4:])
    if lab.startswith("tilt"):
        return "_tilt(cur, %d)" % int(lab[4:])
    if lab.startswith("sift_"):
        return "_sift(cur, %r)" % lab[5:]
    if lab.startswith("lace_"):
        return "_lace(cur, %r)" % lab[5:]
    return "_%s(cur)" % _base(lab)


def needed_helpers(labels):
    want = []
    for l in labels:
        k = "_" + _base(l)
        if k not in want:
            want.append(k)
    return "\n\n\n".join(HELPERS[k] for k in want)


def worked_solution(labels, compact=False):
    lines = ["def f(s):", "    cur = s"]
    for l in labels:
        lines.append("    cur = %s" % _call(l))
    lines.append("    return cur")
    helpers = needed_helpers(labels) if compact else REF_HELPERS
    return helpers + "\n\n\n" + "\n".join(lines)


if __name__ == "__main__":
    import json
    fam = json.load(open(os.path.join(HERE, "quill_family.json"), encoding="utf-8"))
    ok = 0
    for t in fam["teach"] + fam["heldout"]:
        ok += grade(worked_solution(t["ops"], compact=True), t)[0]
    print("quill adapter: worked solutions pass %d/%d (teach+heldout)" % (ok, len(fam["teach"]) + len(fam["heldout"])))
