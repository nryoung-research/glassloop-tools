"""Execution-based scorer for skills-pilot tasks.

Input: a model's raw text response + a task JSON (from gen_tasks.py).
Pipeline:
  1. Extract Python code from the response (fenced ```python, bare ```,
     or a heuristic largest-parseable-def-block fallback; the path that
     fired is logged).
  2. Run each unit test in its OWN subprocess (python -I) with a hard
     timeout (subprocess timeout; no signals -- Windows-safe). The child
     builds quirklib from source with docstrings stripped (compile
     optimize=2, fake filename) and either registers it in sys.modules
     (TIER-S) or BLOCKS `import quirklib` (TIER-F), then execs the model
     code under restricted builtins (no open/eval/exec/input; imports
     limited to a stdlib allowlist) and calls the entry point.
  3. Compare results with STRICT typed equality (bool is not int,
     tuple is not list). No string comparison against reference code
     anywhere.

Score = tests passed / tests total. Per-item JSON log includes extraction
path, per-test pass/fail, and any exception text.

API:   score_response(task_dict, response_text, ...) -> item dict
CLI:   python score_exec.py --task t.json --response r.txt [--log-dir logs]
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUIRKLIB_SRC = (HERE / "quirklib.py").read_text(encoding="utf-8")

# --------------------------------------------------------------------------
# Child runner (written to a temp dir once per process; receives one JSON
# payload on stdin, prints one JSON result on stdout). ASCII-only JSON both
# ways, so child locale never matters. Runs under `python -I`.
# --------------------------------------------------------------------------
RUNNER_SRC = r'''
import sys, io, json, types, traceback, copy
import builtins as _bi


def ser(v, depth, budget):
    budget[0] -= 1
    if budget[0] < 0 or depth > 8:
        return {"t": "over", "r": "structure too large/deep"}
    t = type(v)
    if t is bool:
        return {"t": "bool", "v": v}
    if t is int:
        return {"t": "int", "v": v}
    if t is str:
        return {"t": "str", "v": v[:300]}
    if t is float:
        return {"t": "float", "r": repr(v)}
    if t is list:
        return {"t": "list", "v": [ser(x, depth + 1, budget) for x in v]}
    if t is tuple:
        return {"t": "tuple", "v": [ser(x, depth + 1, budget) for x in v]}
    return {"t": "other", "r": repr(v)[:300]}


def main():
    payload = json.loads(sys.stdin.read())
    code_text = payload["code"]
    entry = payload["entry"]
    args = payload["args"]
    allow_q = bool(payload["quirklib_allowed"])
    qsrc = payload["quirklib_src"]

    out = {"ok": False, "stage": "setup", "error": ""}
    real_stdout = sys.stdout
    sys.stdout = io.StringIO()  # shield: model prints must not corrupt JSON
    try:
        if allow_q:
            mod = types.ModuleType("quirklib")
            qcode = compile(qsrc, "<quirklib>", "exec",
                            dont_inherit=True, optimize=2)  # docstrings gone
            exec(qcode, mod.__dict__)
            sys.modules["quirklib"] = mod

        allowed = {"math", "itertools", "functools", "collections", "re",
                   "operator", "json", "heapq", "bisect", "string",
                   "typing", "random", "copy"}
        orig_import = _bi.__import__

        def guarded_import(name, globals=None, locals=None,
                           fromlist=(), level=0):
            root = name.split(".")[0]
            if root == "quirklib":
                if allow_q:
                    return orig_import(name, globals, locals,
                                       fromlist, level)
                raise ImportError(
                    "quirklib import is BLOCKED for this task tier "
                    "(fact tier: answer from memory)")
            if root in allowed:
                return orig_import(name, globals, locals, fromlist, level)
            raise ImportError(
                "import of %r is blocked in the grading sandbox" % root)

        bmap = dict(vars(_bi))
        for k in ("open", "input", "breakpoint", "exit", "quit", "help",
                  "compile", "eval", "exec", "memoryview"):
            bmap.pop(k, None)
        bmap["__import__"] = guarded_import
        ns = {"__builtins__": bmap, "__name__": "__qgrader__"}
        # Grader-contract fix (catch #15, sealed 2026-08-19): the task
        # instruction says "quirklib is available to your code" - model
        # families read this differently (Qwen writes the import; Llama
        # assumes it is pre-loaded, NameError-ing 36/40 correct
        # compositions). Pre-binding makes the stated contract
        # literally true for both readings; explicit import still works.
        if allow_q:
            ns["quirklib"] = sys.modules["quirklib"]

        out["stage"] = "exec"
        exec(compile(code_text, "<response>", "exec"), ns)

        out["stage"] = "lookup"
        fn = ns.get(entry)
        if fn is None:
            raise KeyError(
                "entry point %r not defined by the response code" % entry)
        if not callable(fn):
            raise TypeError("entry point %r is not callable" % entry)

        out["stage"] = "call"
        result = fn(*copy.deepcopy(args))
        out["result"] = ser(result, 0, [4000])
        out["ok"] = True
        out["stage"] = "done"
    except BaseException:
        out["ok"] = False
        out["error"] = traceback.format_exc(limit=6)[-1500:]
    finally:
        sys.stdout = real_stdout
    real_stdout.write(json.dumps(out, ensure_ascii=True))
    real_stdout.flush()


if __name__ == "__main__":
    main()
'''

_runner_path = None


def _runner_file():
    global _runner_path
    if _runner_path is None or not Path(_runner_path).exists():
        d = Path(tempfile.mkdtemp(prefix="skillpilot_runner_"))
        p = d / "qrunner.py"
        p.write_text(RUNNER_SRC, encoding="utf-8")
        _runner_path = str(p)
    return _runner_path


# --------------------------------------------------------------------------
# Code extraction
# --------------------------------------------------------------------------
FENCE_PY = re.compile(r"```[ \t]*(?:python|py)[ \t]*\n(.*?)```",
                      re.S | re.I)
FENCE_ANY = re.compile(r"```[ \t]*\n(.*?)```", re.S)


def _parses(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _heuristic_defblock(text):
    """Largest parseable slice starting at a column-0 def/import/from/class
    line (prefers slices containing a def). Trims trailing prose by walking
    the end line backwards until the slice parses."""
    lines = text.splitlines()
    starts = [i for i, l in enumerate(lines)
              if re.match(r"^(def|import|from|class)\b", l)]
    best = None  # ((has_def, length), code)
    for s in starts[:8]:
        for e in range(len(lines), s, -1):
            cand = "\n".join(lines[s:e]).strip("\n")
            if not cand:
                continue
            if _parses(cand):
                score = ("def " in cand, len(cand))
                if best is None or score > best[0]:
                    best = (score, cand)
                break  # largest parseable slice for this start found
    return best[1] if best else None


def extract_code(text):
    """Returns (code_or_None, path, notes). path in
    {fenced_python, fenced_bare, heuristic_defblock, none}."""
    notes = []
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")

    blocks = FENCE_PY.findall(text)
    if blocks:
        code = "\n\n".join(textwrap.dedent(b) for b in blocks).strip("\n")
        if code and _parses(code):
            return code, "fenced_python", notes
        notes.append("python fence present but contents not parseable")

    blocks = FENCE_ANY.findall(text)
    if blocks:
        cleaned = []
        for b in blocks:
            first, _, rest = b.partition("\n")
            if first.strip().lower() in ("python", "py"):
                b = rest
            cleaned.append(textwrap.dedent(b))
        code = "\n\n".join(cleaned).strip("\n")
        if code and _parses(code):
            return code, "fenced_bare", notes
        notes.append("bare fence present but contents not parseable")

    code = _heuristic_defblock(text)
    if code:
        return code, "heuristic_defblock", notes

    notes.append("no parseable python code found in response")
    return None, "none", notes


# --------------------------------------------------------------------------
# Strict typed comparison (parent side)
# --------------------------------------------------------------------------

def _tag_expected(e):
    if type(e) is bool:
        return {"t": "bool", "v": e}
    if type(e) is int:
        return {"t": "int", "v": e}
    if type(e) is str:
        return {"t": "str", "v": e}
    if type(e) is list:
        return {"t": "list", "v": [_tag_expected(x) for x in e]}
    raise TypeError("unsupported expected type: %r" % type(e))


def _untag(t):
    if not isinstance(t, dict):
        return "<?>"
    k = t.get("t")
    if k in ("int", "bool", "str"):
        return t.get("v")
    if k == "float":
        return t.get("r")
    if k in ("list", "tuple"):
        inner = [_untag(x) for x in t.get("v", [])]
        return tuple(inner) if k == "tuple" else inner
    return t.get("r", "<%s>" % k)


def _short(v, cap=200):
    r = repr(v)
    return r if len(r) <= cap else r[:cap] + "..."


# --------------------------------------------------------------------------
# Per-test execution
# --------------------------------------------------------------------------

def _run_one(code, task, test, idx, timeout, python_exe):
    payload = {
        "code": code,
        "entry": task["entry_point"],
        "args": test["args"],
        "quirklib_allowed": bool(task.get("quirklib_importable", True)),
        "quirklib_src": QUIRKLIB_SRC,
    }
    expected = test["expected"]
    rec = {"i": idx, "pass": False, "reason": "", "got": None,
           "expected": expected}
    try:
        proc = subprocess.run(
            [python_exe, "-I", _runner_file()],
            input=json.dumps(payload, ensure_ascii=True),
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        rec["reason"] = "timeout: no result within %.1fs" % timeout
        return rec
    stdout = (proc.stdout or "").strip()
    try:
        # JSON starts at the first "{"; tolerate any stray trailing bytes.
        out, _ = json.JSONDecoder().raw_decode(stdout, stdout.index("{"))
    except Exception:
        rec["reason"] = ("runner produced no JSON (exit %s): stdout=%r "
                         "stderr=%r" % (proc.returncode, stdout[:200],
                                        (proc.stderr or "")[-300:]))
        return rec
    if not out.get("ok"):
        err = (out.get("error") or "").strip()
        last_line = err.splitlines()[-1] if err else "(no error text)"
        rec["reason"] = "%s error: %s" % (out.get("stage", "?"),
                                          last_line[:300])
        rec["error_tail"] = err[-600:]
        return rec
    got_tag = out.get("result")
    rec["got"] = _short(_untag(got_tag))
    if got_tag == _tag_expected(expected):
        rec["pass"] = True
    else:
        rec["reason"] = "result mismatch: got %s expected %s" % (
            rec["got"], _short(expected))
    return rec


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def score_response(task, response_text, timeout=10.0, log_dir=None,
                   python_exe=None):
    """Execution-scores one raw model response against one task dict.
    Never raises on bad responses; returns the per-item log dict."""
    python_exe = python_exe or sys.executable
    code, path, notes = extract_code(response_text)
    tests = task["tests"]
    results = []
    if code is None:
        for i, t in enumerate(tests):
            results.append({"i": i, "pass": False,
                            "reason": "no code extracted: " + "; ".join(notes),
                            "got": None, "expected": t["expected"]})
    else:
        for i, t in enumerate(tests):
            results.append(_run_one(code, task, t, i, timeout, python_exe))
    passed = sum(1 for r in results if r["pass"])
    first_fail = next((r["reason"] for r in results if not r["pass"]), "")
    item = {
        "task_id": task["id"],
        "tier": task.get("tier"),
        "entry_point": task["entry_point"],
        "quirklib_importable": bool(task.get("quirklib_importable", True)),
        "score": (passed / len(results)) if results else 0.0,
        "passed": passed,
        "total": len(results),
        "extraction": {
            "path": path,
            "notes": notes,
            "code_chars": len(code) if code else 0,
            "code_sha256": (hashlib.sha256(code.encode("utf-8")).hexdigest()
                            if code else None),
        },
        "tests": results,
        "error_summary": first_fail,
    }
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        rsha = hashlib.sha256(
            (response_text or "").encode("utf-8", "replace")).hexdigest()[:8]
        fp = log_dir / ("%s.%s.json" % (task["id"], rsha))
        fp.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    return item


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", required=True, type=Path,
                    help="task JSON file (single task object)")
    ap.add_argument("--response", required=True, type=Path,
                    help="file containing the model's raw text response")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--log-dir", type=Path, default=HERE / "logs")
    args = ap.parse_args(argv)

    task = json.loads(args.task.read_text(encoding="utf-8"))
    response = args.response.read_text(encoding="utf-8", errors="replace")
    item = score_response(task, response, timeout=args.timeout,
                          log_dir=args.log_dir)
    print(json.dumps(item, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
