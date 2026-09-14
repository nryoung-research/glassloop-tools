#!/usr/bin/env python3
"""PILOT TEACH LEG (generated from vex_teach.py 39cac34e5e8d9f44 by make_pilot_teach.py; adds VXT_SAVE_DIR materialization with an exact TL->HF mapping check; otherwise identical).
VEX ROTATE_REPLAY_FULLRANK WRITE ARM - CLIMBER rung 1.

Installs the vex API into live weights using the FP-616 write policy
(deep-zone rotation + replay protection), then re-measures the frozen
held-out split. It asks whether the FP-616 policy, which is proven on
FACTS, also moves a CAPABILITY: correct use of unseen semantics on
novel inputs.

ARM NAMING (G-322 correction, adopted): what this runner performs is
AdamW over rotating full-rank MLP windows with replay - that is the
FP-616 TRAJECTORY POLICY, not the FP-609 constraint-solved writer.
Calling it "CONSTRUCTED" overclaimed. Honest names:
  ARM=ROTATE_REPLAY_FULLRANK  the FP-616 policy, transferred to skills
  ARM=BASE                    no write (control re-measure)
  (FP609_CONSTRAINED_DELTA exists only once that writer is actually
   integrated; this file cannot produce it.)
SCOPE CONSEQUENCE: this runner can test whether the FP-616 POLICY
transfers from facts to capabilities. It CANNOT establish that FP-609
installs a capability - a different writer, a different claim.
The LoRA arm lives in its own runner so the comparison is dose-matched
by construction rather than by bookkeeping.

Teaching corpus: for each teach-split task, the spec + the task prompt +
a CORRECT worked solution. Held-out tasks are never seen in any form.

Receipts: per-write bill in loss terms, held-out pass rate before/after,
general-coding collateral hook, and hash-exact revert proof.

Env: VXT_MODEL, VXT_FAMILY, VXT_OUT, VXT_TAG, VXT_STEPS, VXT_LR,
     VXT_ARM, VXT_ROTATE, VXT_REPLAY, VXT_EVAL (1 = eval held-out after).
"""
import hashlib
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vex_tasks as VT  # noqa: E402
import vexlib as V  # noqa: E402

MODEL = os.environ.get("VXT_MODEL", "Qwen/Qwen2.5-3B-Instruct")
FAMILY = os.environ.get("VXT_FAMILY") or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "vex_family.json")
OUT = os.environ.get("VXT_OUT") or os.path.expanduser("~/climber")
TAG = os.environ.get("VXT_TAG", "rotate_replay_fullrank")
STEPS = int(os.environ.get("VXT_STEPS", "60"))
LR = float(os.environ.get("VXT_LR", "1e-5"))
ARM = os.environ.get("VXT_ARM", "ROTATE_REPLAY_FULLRANK")
ROTATE = os.environ.get("VXT_ROTATE", "1") == "1"
REPLAY = os.environ.get("VXT_REPLAY", "1") == "1"
DO_EVAL = os.environ.get("VXT_EVAL", "1") == "1"
VIEW = os.environ.get("VXT_VIEW", "closed").lower()
MAX_NEW = 512
MAX_TEACH_TOK = int(os.environ.get("VXT_MAXTOK", "320"))
# 0 = step-bounded (legacy); >0 = token-bounded dose per write.
DOSE_TOK = int(os.environ.get("VXT_DOSE_TOK", "0"))
# Collateral leg: generate HumanEval+ WHILE THE WRITE IS STILL
# INSTALLED. Without this the runner reverts first and the
# "did installing the API damage general coding" question is
# unanswerable - the frozen rung-0 reference is 0.774/0.701.
COLLATERAL = os.environ.get("VXT_COLLATERAL", "0") == "1"

# ---- reference implementations used to author worked solutions --------
REF_HELPERS = '''def _chunk(xs, k):
    out = []
    for i in range(0, len(xs), k):
        p = list(xs[i:i + k])
        while len(p) < k:
            p.append(None)
        out.append(p)
    return out


def _weave(a, b):
    out = []
    n = min(len(a), len(b))
    for i in range(n):
        out.append(a[i])
        out.append(b[i])
    rest = list(a[n:]) if len(a) > len(b) else list(b[n:])
    out.extend(reversed(rest))
    return out


def _tally(xs):
    c = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return {k: v for k, v in c.items() if v >= 2}


def _spin(xs, n):
    if not xs:
        return []
    L = len(xs)
    m = abs(n) % (L + 1)
    if m == L:
        m = 0
    if n < 0:
        m = (L - m) % L
    return list(xs[m:]) + list(xs[:m])


def _prune(xs, p):
    out = []
    seen = False
    for x in xs:
        if p(x):
            if not seen:
                seen = True
                out.append(x)
            continue
        out.append(x)
    return out


def _braid(xs):
    n = len(xs)
    if n == 0:
        return []
    return [(xs[i], xs[(2 * i + 1) % n]) for i in range(n)]
'''

PRED_SRC = {"even": "lambda x: isinstance(x, int) and x % 2 == 0",
            "gt10": "lambda x: isinstance(x, int) and x > 10",
            "neg": "lambda x: isinstance(x, int) and x < 0"}
WEAVE_TAIL = {"100200": [100, 200], "777": [7, 7, 7],
              "-1-2-3-4": [-1, -2, -3, -4]}


def _helper_blocks():
    """Split REF_HELPERS into one block per function so a lesson can
    carry ONLY the helpers its own pipeline uses. Repeating all six in
    every lesson made the teaching texts long enough to be truncated
    mid-function by the token cap, which would teach broken code."""
    blocks, cur = {}, []
    name = None
    for line in REF_HELPERS.splitlines():
        if line.startswith("def _"):
            if name:
                blocks[name] = "\n".join(cur).rstrip()
            name = line[4:line.index("(")]
            cur = [line]
        elif name:
            cur.append(line)
    if name:
        blocks[name] = "\n".join(cur).rstrip()
    return blocks


HELPERS = _helper_blocks()


def needed_helpers(labels):
    want = []
    for l in labels:
        for key in ("chunk", "spin", "prune", "braid", "tally", "weave"):
            if l.startswith(key) and "_" + key not in want:
                want.append("_" + key)
    missing = [k for k in want if k not in HELPERS]
    if missing:
        raise KeyError("helper blocks missing: %s" % missing)
    return "\n\n\n".join(HELPERS[k] for k in want)


def worked_solution(labels, compact=False):
    lines = ["def f(xs):", "    cur = list(xs)"]
    for l in labels:
        if l.startswith("chunk"):
            lines.append("    cur = _chunk(cur, %s)" % l[5:])
        elif l.startswith("spin"):
            lines.append("    cur = _spin(cur, %s)" % l[4:])
        elif l.startswith("prune_"):
            lines.append("    cur = _prune(cur, %s)" % PRED_SRC[l[6:]])
        elif l == "braid":
            lines.append("    cur = _braid(cur)")
        elif l == "tally":
            lines.append("    cur = _tally(cur)")
        elif l.startswith("weave"):
            lines.append("    cur = _weave(cur, %r)" % WEAVE_TAIL[l[5:]])
    lines.append("    return cur")
    helpers = needed_helpers(labels) if compact else REF_HELPERS
    return helpers + "\n\n\n" + "\n".join(lines)


# pilot L6 (C-707): FAMILY ADAPTER. VXT_FAMILY_MODULE names a module exposing NAME, SPEC, phrase, grade, worked_solution(labels, compact);
# when set, the vex bindings below are replaced and the lesson strings / receipt name carry the family's NAME. Unset = vex, byte-identical.
FAMILY_NAME = "vex"
_FA = os.environ.get("VXT_FAMILY_MODULE")
if _FA:
    import importlib as _il
    _FAM = _il.import_module(_FA)
    class _VTA:
        phrase = staticmethod(_FAM.phrase)
        grade = staticmethod(_FAM.grade)
    class _VA:
        SPEC = _FAM.SPEC
    VT, V = _VTA, _VA
    worked_solution = _FAM.worked_solution
    FAMILY_NAME = _FAM.NAME


def teach_texts(task):
    """Surface forms of one lesson (the FP-616 fact_texts pattern at
    capability scale). COMPACT: each carries only the helpers this
    pipeline uses, so nothing is truncated by the token cap."""
    sol = worked_solution(task["ops"], compact=True)
    return [
        V.SPEC,
        "Using the %s semantics, %s:\n\n```python\n%s\n```" % (FAMILY_NAME, "%s", "%s")
        % (VT.phrase(task["ops"]), sol),
        "```python\n%s\n```" % sol,
        (FAMILY_NAME + " pipeline: %s\n\n```python\n%s\n```")
        % (VT.phrase(task["ops"]), sol),
    ]


CHAT_FORM = os.environ.get("VXT_CHAT_FORM", "0").lower()          # FP-644: "0" unchanged | "1" raw forms + exam-format pair | "only" the pair alone
if CHAT_FORM not in ("0", "1", "only"):
    raise RuntimeError("PILOT-CHATFORM-REFUSED: VXT_CHAT_FORM must be 0, 1 or only")


def chat_form(task, tok):
    """The EXAM-FORMAT surface form: the chat-templated closed (or open, per VXT_VIEW) prompt exactly as cap_eval renders it
    (add_generation_prompt=True) paired with the fenced compact worked solution plus the eos token as the assistant response.
    Returned as a (prompt, response) tuple so the training loop can put the loss on the response positions only."""
    prompt = task["prompt_closed"] if VIEW == "closed" else task["prompt_open"]
    rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
    return (rendered, "```python\n%s\n```%s" % (worked_solution(task["ops"], compact=True), tok.eos_token))


CONTRAST = os.environ.get("VXT_CONTRAST")                           # FP-645: path to fp645_<set>_pairs.json; unset = unchanged
CONTRAST_PAIRS = None
if CONTRAST:
    _cb = open(CONTRAST, "rb").read()
    _csha = hashlib.sha256(_cb).hexdigest()
    _cexp = os.environ.get("VXT_EXPECT_CONTRAST_SHA", "")
    if len(_cexp) != 64 or _cexp != _csha:
        raise RuntimeError("PILOT-CONTRAST-REFUSED: contrast pairs digest %s differs from the expected pin %s" % (_csha[:16], _cexp[:16]))
    CONTRAST_PAIRS = json.loads(_cb.decode("utf-8"))["items"]
    CONTRAST_SHA = _csha


STOP_AFTER = int(os.environ.get("VXT_STOP_AFTER_LESSON", "0"))     # FP-647 Arm A prefix legs
OPT_KIND = os.environ.get("VXT_OPT", "adamw").lower()               # FP-647 W1
FREEZE_DOWN = os.environ.get("VXT_FREEZE_DOWN", "0") == "1"         # FP-647 W3
PROJECT_NULL = os.environ.get("VXT_PROJECT_NULL")                   # FP-647 W4
if OPT_KIND not in ("adamw", "sgd"):
    raise RuntimeError("PILOT-OPT-REFUSED: VXT_OPT must be adamw or sgd")
if PROJECT_NULL:
    _nb = open(PROJECT_NULL, "rb").read()
    _nsha = hashlib.sha256(_nb).hexdigest()
    if os.environ.get("VXT_EXPECT_NULL_SHA", "") != _nsha:
        raise RuntimeError("PILOT-NULL-REFUSED: null bases digest %s differs from the expected pin" % _nsha[:16])
    NULL_SHA = _nsha
    del _nb


def hlog(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def extract_code(text):
    for fence in ("```python", "```"):
        if fence in text:
            return text.split(fence, 1)[1].split("```", 1)[0]
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    import gc
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM
    # G-748: hash the SAME bytes that are parsed and COMPARE to the capsule's expected full pin (VXT_EXPECT_FAMILY_SHA); required for
    # continuation legs (VXT_LOAD_MASTER) and adapter families (VXT_FAMILY_MODULE); the verified identity is recorded in the receipt.
    _fam_bytes = open(FAMILY, "rb").read()
    FAMILY_FILE_SHA = hashlib.sha256(_fam_bytes).hexdigest()
    _exp_fam = os.environ.get("VXT_EXPECT_FAMILY_SHA")
    if (os.environ.get("VXT_LOAD_MASTER") or os.environ.get("VXT_FAMILY_MODULE")) and not (_exp_fam and len(_exp_fam) == 64):
        raise RuntimeError("PILOT-FAMILY-REFUSED: VXT_EXPECT_FAMILY_SHA (64 hex) is required for continuation / adapter legs")
    if _exp_fam and _exp_fam != FAMILY_FILE_SHA:
        raise RuntimeError("PILOT-FAMILY-REFUSED: family file digest %s differs from the expected pin %s" % (FAMILY_FILE_SHA[:16], _exp_fam[:16]))
    fam = json.loads(_fam_bytes.decode("utf-8"))
    teach_tasks, held = fam["teach"], fam["heldout"]
    # sanity: verify every worked solution actually passes its own task
    bad = [t["task_id"] for t in teach_tasks
           if not VT.grade(worked_solution(t["ops"]), t)[0]]
    if bad:
        # Explicit raise, never `assert`: an assert-based gate is
        # removed entirely under `python -O`, which would let a broken
        # teaching corpus reach the weights silently. Standing defect
        # class, found twice tonight in two codebases.
        raise RuntimeError(
            "worked solutions fail their own tests: %s" % bad[:5])
    hlog("teach corpus verified: %d worked solutions all pass their tests"
         % len(teach_tasks))

    REVISION = os.environ.get("VXT_REVISION") or None          # PILOT: pinned base revision (receipted)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL, hf_model=hf, dtype=torch.float32, device="cuda")
    SAVE_DIR = os.environ.get("VXT_SAVE_DIR")
    HF_MAP = {"W_in": "up_proj.weight", "W_gate": "gate_proj.weight", "W_out": "down_proj.weight", "b_in": "up_proj.bias", "b_out": "down_proj.bias"}

    def tl_to_hf(name, t):
        """TL parameter name blocks.L.mlp.X -> (HF parameter name, HF-shaped tensor)."""
        parts = name.split(".")
        L, x = int(parts[1]), parts[3]
        hf_name = "model.layers.%d.mlp.%s" % (L, HF_MAP[x])
        return hf_name, (t.T.contiguous() if x.startswith("W_") else t.contiguous())

    if SAVE_DIR:
        # G-733: TransformerLens GatedMLP carries synthetic biases (b_in, b_out) that native HF Qwen2 has none of. The pilot is WEIGHT-ONLY:
        # those biases are excluded from learning (win_params below) and must be exactly zero at export; only the three matrices are mapped.
        hf_sd = hf.state_dict()
        bad, nonzero_bias = [], []
        for n_, p_ in model.named_parameters():
            if ".mlp." in n_ and n_.split(".")[1].isdigit():
                if n_.split(".")[3].startswith("b_"):
                    if float(p_.detach().abs().max()) != 0.0:
                        nonzero_bias.append(n_)
                    continue
                hn, ht = tl_to_hf(n_, p_.detach().cpu().float())
                if hn not in hf_sd or hf_sd[hn].shape != ht.shape or not torch.equal(hf_sd[hn].float(), ht):
                    bad.append(n_)
        if bad or nonzero_bias:
            raise RuntimeError("PILOT-MAPPING-REFUSED: TL->HF weight mapping not exact on the unwritten model for %s; nonzero synthetic biases %s" % (bad[:5], nonzero_bias[:5]))
        hlog("pilot: TL->HF MLP weight mapping verified exact on the unwritten model; synthetic biases zero")
        del hf_sd
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    tok = model.tokenizer
    if CHAT_FORM != "0":                                           # FP-644: the chat form must tokenize under TL exactly as the exam renders it
        _p0 = teach_tasks[0]["prompt_closed"] if VIEW == "closed" else teach_tasks[0]["prompt_open"]
        _ex = tok.apply_chat_template([{"role": "user", "content": _p0}], add_generation_prompt=True, return_tensors="pt")
        _ex = _ex.input_ids if hasattr(_ex, "input_ids") else _ex
        _tl = model.to_tokens([chat_form(teach_tasks[0], tok)[0]], prepend_bos=False).cpu()
        if tuple(_tl.shape) != tuple(_ex.shape) or not torch.equal(_tl, _ex.cpu()):
            raise RuntimeError("PILOT-CHATFORM-REFUSED: TL tokenization of the rendered prompt differs from the exam render")
        hlog("pilot: chat form ON (%s): TL tokens of the rendered prompt equal the exam render (%d tokens); loss on response positions only" % (CHAT_FORM, int(_ex.shape[1])))
    nl = model.cfg.n_layers
    top = int(os.environ.get("VXT_WINDOW_TOP", str(nl - 2)))     # T2 control (C-729): shift the whole 4x2-layer rotation; default = nl-2 (unchanged)
    if not (7 <= top <= nl - 2):
        raise RuntimeError("PILOT-WINDOW-REFUSED: VXT_WINDOW_TOP %d out of range [7, %d]" % (top, nl - 2))
    ROT = [(top - 6, top - 5), (top - 4, top - 3),
           (top - 2, top - 1), (top, top + 1)]
    LATE = (int(nl * 0.82), int(nl * 0.94))
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    def win_params(lo, hi):
        # PILOT (G-733): WEIGHT-ONLY windows; TransformerLens' synthetic MLP biases (b_in, b_out) are never trained so the export is exact
        return [(n, p) for n, p in model.named_parameters()
                if ".mlp." in n and n.split(".")[1].isdigit()
                and not n.split(".")[3].startswith("b_")
                and not (FREEZE_DOWN and n.split(".")[3] == "W_out")    # FP-647 W3: down_proj never trained
                and lo <= int(n.split(".")[1]) <= hi]

    ALL = sorted({n for w in (ROT + [LATE]) for n, _ in win_params(*w)})
    BASE = {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if n in ALL}
    DOWN_SNAP = ({n: p.detach().cpu().clone() for n, p in model.named_parameters()
                  if ".mlp." in n and n.split(".")[1].isdigit() and n.split(".")[3] == "W_out"} if FREEZE_DOWN else {})   # FP-647 W3 verification snapshot
    NULL_BASES = None
    if PROJECT_NULL:
        NULL_BASES = torch.load(PROJECT_NULL, map_location="cpu")
        NULL_BASES = {int(k): {kk: vv.float().to("cuda") for kk, vv in v.items()} for k, v in NULL_BASES.items() if str(k).isdigit()}
        hlog("pilot: FP-647 W4 null bases loaded for layers %s (ranks %s)" % (sorted(NULL_BASES), {L_: (int(v["resid"].shape[1]), int(v["mlp"].shape[1])) for L_, v in sorted(NULL_BASES.items())}))
    # WHOLE editable state, with names/shapes/dtypes bound (G-325).
    # A prefix spot-hash of eight tensors cannot certify a revert; it
    # can only fail to detect one. Every tensor the runner is capable
    # of writing is hashed, and its identity is hashed with it so a
    # reshaped or retyped tensor cannot collide with the original.
    def _state_hash(getter):
        h = hashlib.sha256()
        for k in sorted(BASE):
            t = getter(k)
            h.update(k.encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("utf-8"))
            h.update(str(t.dtype).encode("utf-8"))
            h.update(t.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    LOAD_MASTER = os.environ.get("VXT_LOAD_MASTER")             # PILOT S1 loader: continue learning from a saved fp32 written-window master
    if LOAD_MASTER:
        # G-739 / pilot v0.4 PARENT AUTHENTICATION: the parent is a COMPLETED bundle (its leg receipt says reload_verified, its MANIFEST exists),
        # its master file digest equals VXT_EXPECT_MASTER_SHA and its MANIFEST digest equals VXT_EXPECT_MANIFEST_SHA (both from the parent's
        # freeze receipt); any mismatch refuses the leg before a single tensor is loaded. Partial directories from exceptions never pass.
        import hashlib as _hl, glob as _gl
        _exp_m, _exp_f = os.environ.get("VXT_EXPECT_MASTER_SHA"), os.environ.get("VXT_EXPECT_MANIFEST_SHA")
        if not (_exp_m and _exp_f):
            raise RuntimeError("PILOT-PARENT-REFUSED: VXT_EXPECT_MASTER_SHA and VXT_EXPECT_MANIFEST_SHA are required with VXT_LOAD_MASTER")
        _pdir = os.path.dirname(os.path.abspath(LOAD_MASTER))
        _man = os.path.join(_pdir, "MANIFEST-SHA256.json")
        if not os.path.exists(_man):
            raise RuntimeError("PILOT-PARENT-REFUSED: parent MANIFEST-SHA256.json missing in %s (incomplete bundle)" % _pdir)
        _sha = lambda p: _hl.sha256(open(p, "rb").read()).hexdigest()
        _got_m, _got_f = _sha(LOAD_MASTER), _sha(_man)
        if len(_exp_m) != 64 or len(_exp_f) != 64 or _got_m != _exp_m or _got_f != _exp_f:       # G-740: FULL 64-hex equality, never a prefix
            raise RuntimeError("PILOT-PARENT-REFUSED: digest mismatch master %s vs expected %s / manifest %s vs expected %s" % (_got_m[:16], _exp_m[:16], _got_f[:16], _exp_f[:16]))
        # G-740: the parent's leg receipt is named EXPLICITLY (path + full digest) and may be a teach receipt (nested materialized_bf16.dir /
        # .reload_verified) or a repair receipt (top-level save_dir / reload_verified); it must name this bundle and record reload_verified.
        _rpath, _rexp = os.environ.get("VXT_PARENT_RECEIPT"), os.environ.get("VXT_PARENT_RECEIPT_SHA")
        if not (_rpath and _rexp and len(_rexp) == 64):
            raise RuntimeError("PILOT-PARENT-REFUSED: VXT_PARENT_RECEIPT and VXT_PARENT_RECEIPT_SHA (64 hex) are required")
        if _sha(_rpath) != _rexp:
            raise RuntimeError("PILOT-PARENT-REFUSED: parent receipt digest mismatch")
        _rj = json.load(open(_rpath))
        _mb = _rj.get("materialized_bf16") if isinstance(_rj.get("materialized_bf16"), dict) else {}
        _rdir = _mb.get("dir") or _rj.get("save_dir") or ""
        _rver = bool(_mb.get("reload_verified")) or bool(_rj.get("reload_verified"))
        if not (_rver and os.path.abspath(_rdir) == os.path.abspath(_pdir)):
            raise RuntimeError("PILOT-PARENT-REFUSED: parent receipt does not record reload_verified for %s (records %r)" % (_pdir, _rdir))
        _prev = _mb.get("base_revision") or _rj.get("revision") or _rj.get("base_revision")
        if not _prev or _prev != REVISION:                                                                  # G-741: bind the base identity across legs
            raise RuntimeError("PILOT-PARENT-REFUSED: parent base revision %r differs from this leg's REVISION %r" % (_prev, REVISION))
        if os.environ.get("VXT_RESERVE", "0") != "1" or int(os.environ.get("VXT_LEG_TOK", "0")) <= 0:       # G-740: reservation and a positive leg budget are REQUIRED for continuation legs
            raise RuntimeError("PILOT-PARENT-REFUSED: continuation legs require VXT_RESERVE=1 and VXT_LEG_TOK > 0")
        PARENT_RECORD = {"master": os.path.abspath(LOAD_MASTER), "master_sha256": _got_m, "manifest_sha256": _got_f, "receipt": os.path.abspath(_rpath), "receipt_sha256": _rexp, "receipt_kind": _rj.get("instrument") or _rj.get("experiment"), "reload_verified_receipt_found": True}
        hlog("pilot: parent authenticated (master %s, manifest %s)" % (_got_m[:16], _got_f[:16]))
        master_in = torch.load(LOAD_MASTER, map_location="cpu")
        # T2 control (C-729): the parent's master may lie OUTSIDE this leg's editable window (disjoint-window continuation). Every master
        # tensor must exist in the model as an MLP weight with the same shape; all are loaded (they ARE the parent state); those outside
        # the editable set are INHERITED: frozen, never trained, hashed, and exported in the union master so S_{k+1} carries S_k in full.
        _names = dict(model.named_parameters())
        for n in master_in:
            if n not in _names or ".mlp." not in n or n.split(".")[3].startswith("b_"):
                raise RuntimeError("PILOT-MASTER-REFUSED: master tensor %s is not an MLP weight of this model" % n)
            if tuple(master_in[n].shape) != tuple(_names[n].shape):
                raise RuntimeError("PILOT-MASTER-REFUSED: shape mismatch on %s" % n)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in master_in:
                    p.copy_(master_in[n].to(p.device, torch.float32))
        _trained = {n for w in (ROT if ROTATE else [LATE]) for n, _ in win_params(*w)}      # the windows this leg actually trains
        INHERITED = sorted(set(master_in) - _trained)                                         # parent tensors this leg never trains (C-731 correction)
        if INHERITED and (set(INHERITED) & {n for n, _ in win_params(*LATE)}) and not ROTATE:
            raise RuntimeError("PILOT-WINDOW-REFUSED: ROTATE=0 would train the LATE window, which overlaps inherited parent tensors")
        BASE = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if n in BASE}   # the revert target is now the loaded parent S_k
        INHERITED_HASH = hashlib.sha256(b"".join(master_in[n].float().contiguous().numpy().tobytes() for n in INHERITED)).hexdigest() if INHERITED else None
        hlog("pilot: loaded fp32 master %s: %d tensors (%d editable this leg, %d INHERITED frozen outside the window)" % (LOAD_MASTER, len(master_in), len(set(master_in) & set(BASE)), len(INHERITED)))
    base_hash = _state_hash(lambda k: BASE[k])
    n_hashed = len(BASE)

    def restore():
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n in BASE:
                    p.copy_(BASE[n].to(p.device))

    def live_hash():
        with torch.no_grad():
            d = dict(model.named_parameters())
            return _state_hash(lambda k: d[k])

    SEED = int(os.environ.get("VXT_SEED", "20260829"))                 # FP-651: data-schedule seed switch (default = the frozen recipe)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    LEG_STATE = {"used": 0, "n_lessons": 0, "lesson_index": 0}      # PILOT: global loss-bearing token counter + roster position for the leg

    def _seq(s):
        """One training sequence -> (toks [1,T], n_loss_bearing, loss_from). A str is a raw form exactly as before (TL default BOS,
        truncated at MAX_TEACH_TOK, every next-token position bears loss). A (prompt, response) tuple is the FP-644 chat form: tokenized
        without BOS as the exam renders it, refused rather than truncated, prompt verified as a token prefix, loss on response positions."""
        if isinstance(s, tuple):
            p_, r_ = s
            ptoks = model.to_tokens([p_], prepend_bos=False)
            toks = model.to_tokens([p_ + r_], prepend_bos=False)
            if int(toks.shape[1]) > MAX_TEACH_TOK:
                raise RuntimeError("PILOT-CHATFORM-REFUSED: chat-form lesson of %d tokens exceeds VXT_MAXTOK %d" % (int(toks.shape[1]), MAX_TEACH_TOK))
            n_p = int(ptoks.shape[1])
            if not torch.equal(toks[:, :n_p], ptoks):
                raise RuntimeError("PILOT-CHATFORM-REFUSED: the rendered prompt is not a token prefix of prompt+response")
            del ptoks
            return toks, int(toks.shape[1]) - n_p, n_p
        toks = model.to_tokens([s])
        if toks.shape[1] > MAX_TEACH_TOK:
            toks = toks[:, :MAX_TEACH_TOK]
        return toks, max(0, int(toks.shape[1]) - 1), None

    def write(texts, lo, hi, priors):
        wp = win_params(lo, hi)
        for _, p in wp:
            p.requires_grad_(True)
        opt = (torch.optim.SGD([p for _, p in wp], lr=LR, momentum=0.0) if OPT_KIND == "sgd"        # FP-647 W1
               else torch.optim.AdamW([p for _, p in wp], lr=LR, weight_decay=0.0))
        torch.set_grad_enabled(True)
        # DOSE IS COUNTED IN TOKENS, NOT STEPS (G-318, reproduced
        # in-house: VEX lessons are 14.79x longer than FP-616 fact
        # forms in tokenizer units, so "60 steps" here equals ~3.94x
        # FP-616's realized 150-step dose). Optimizer steps are not a
        # transferable dose unit across corpora; the receipt therefore
        # reports exact non-padding token positions consumed, with
        # steps and wall time kept as separate diagnostics. When
        # DOSE_TOK is set, the write stops on the token budget instead
        # of the step count, so Constructed and LoRA can consume an
        # identical frozen text schedule.
        losses = []
        tok_used = 0
        LEG_TOK = int(os.environ.get("VXT_LEG_TOK", "0"))
        RESERVE = os.environ.get("VXT_RESERVE", "0") == "1"
        quota = None
        if RESERVE and LEG_TOK:
            remaining_lessons = max(1, LEG_STATE["n_lessons"] - LEG_STATE["lesson_index"])
            quota = (LEG_TOK - LEG_STATE["used"]) // remaining_lessons
            LEG_STATE.setdefault("quotas", []).append(quota)
        for _ in range(STEPS):
            if DOSE_TOK and tok_used >= DOSE_TOK:
                break
            if LEG_TOK and LEG_STATE["used"] >= LEG_TOK:            # PILOT: the leg budget binds across lessons
                break
            if RESERVE and quota is not None:
                seqs_peek = [rng.choice(texts)]
                if REPLAY and priors:
                    seqs_peek.append(rng.choice(rng.choice(priors)))
                need = 0
                for s_ in seqs_peek:
                    t_, n_, _ = _seq(s_)                          # FP-644: raw forms count as before; the chat form counts its response positions
                    need += n_
                    del t_
                if tok_used + need > quota or LEG_STATE["used"] + need > LEG_TOK:
                    break                                           # the step does not fit: settle the lesson (underfill <= one step)
                seqs_fixed = seqs_peek
            else:
                seqs_fixed = None
            # Sequence-at-a-time with gradient accumulation. Batching a
            # new-fact text and a replay text together doubled peak
            # activation memory and OOM'd the fp32 3B backward; the
            # accumulated gradient is identical (mean of per-sequence
            # losses) at roughly half the peak.
            if seqs_fixed is not None:
                seqs = seqs_fixed                                    # the peeked step is the executed step (same RNG draw)
            else:
                seqs = [rng.choice(texts)]
                if REPLAY and priors:
                    seqs.append(rng.choice(rng.choice(priors)))
            if CONTRAST_PAIRS is not None:                           # FP-645: one contrast pair per step, own budget, chat form, response-only loss
                _cp = rng.choice(CONTRAST_PAIRS)
                _cprompt = tok.apply_chat_template([{"role": "user", "content": _cp["prompt"]}], add_generation_prompt=True, tokenize=False)
                seqs = list(seqs) + [("CONTRAST", _cprompt, _cp["answer"] + tok.eos_token)]
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in seqs:
                is_contrast = isinstance(s, tuple) and len(s) == 3 and s[0] == "CONTRAST"
                if is_contrast:
                    s = (s[1], s[2])
                toks, n_lb, loss_from = _seq(s)
                if is_contrast:                                      # FP-645: contrast targets are counted in their own budget, not the vex dose
                    LEG_STATE["contrast_used"] = LEG_STATE.get("contrast_used", 0) + n_lb
                    LEG_STATE["contrast_draws"] = LEG_STATE.get("contrast_draws", 0) + 1
                    n_lb = 0
                # LOSS-BEARING positions, not raw input positions
                # (G-321): the first token of a sequence carries no
                # next-token target, so an input-token count overstates
                # the causal training dose by one per sequence and can
                # equate two arms that received unequal supervision.
                # FP-644: for the chat form, loss-bearing = response positions only.
                tok_used += n_lb
                LEG_STATE["used"] += n_lb
                if loss_from is None:
                    loss = model(toks, return_type="loss") / len(seqs)
                else:
                    lpt = model(toks, return_type="loss", loss_per_token=True)       # [1, T-1]; lpt[:, j] scores the prediction of token j+1
                    loss = lpt[:, loss_from - 1:].mean() / len(seqs)
                    LEG_STATE["chat_fed"] = LEG_STATE.get("chat_fed", 0) + int(toks.shape[1]) - 1
                    LEG_STATE["chat_loss_bearing"] = LEG_STATE.get("chat_loss_bearing", 0) + n_lb
                    LEG_STATE["chat_draws"] = LEG_STATE.get("chat_draws", 0) + 1
                    del lpt
                loss.backward()
                total += float(loss)
                del toks, loss
            _first = not LEG_STATE.get("first_step_recorded", False)
            if _first or NULL_BASES is not None:
                _pre = {n_: p_.detach().clone() for n_, p_ in wp}
            if _first:                                                  # FP-647: first-step records g_1 (accumulated pre-step gradient)
                LEG_STATE["first_step_recorded"] = True
                torch.save({n_: p_.grad.detach().cpu().float().clone() for n_, p_ in wp}, os.path.join(OUT, "first_step_g1.pt"))
            opt.step()
            if NULL_BASES is not None:                                  # FP-647 W4: project the REALIZED update on its input side (TL rows)
                with torch.no_grad():
                    for n_, p_ in wp:
                        L_ = int(n_.split(".")[1])
                        V = NULL_BASES[L_]["resid" if n_.split(".")[3] in ("W_in", "W_gate") else "mlp"].to(p_.dtype)
                        d = p_ - _pre[n_]
                        proj = V @ (V.t() @ d)
                        p_.copy_(_pre[n_] + d - proj)
                        LEG_STATE["proj_energy_removed"] = LEG_STATE.get("proj_energy_removed", 0.0) + float((proj ** 2).sum())
                        LEG_STATE["proj_energy_total"] = LEG_STATE.get("proj_energy_total", 0.0) + float((d ** 2).sum())
                        del d, proj
            if _first:                                                  # u_1 = the actual fp32 delta of the first step (after any projection)
                torch.save({n_: (p_.detach() - _pre[n_]).cpu().float().clone() for n_, p_ in wp}, os.path.join(OUT, "first_step_u1.pt"))
            if _first or NULL_BASES is not None:
                del _pre
            losses.append(total)
        torch.set_grad_enabled(False)
        for _, p in wp:
            p.requires_grad_(False)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        if not losses:                                                 # PILOT (G-736): budget exhausted before any step -> a settled zero-token lesson
            return None, None, 0, 0
        return losses[0], losses[-1], tok_used, len(losses)

    @torch.no_grad()
    def eval_split(tasks, label):
        npass, rows = 0, []
        for i, t in enumerate(tasks):
            prompt = (t["prompt_closed"] if VIEW == "closed"
                      else t["prompt_open"])
            msgs = [{"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(
                msgs, add_generation_prompt=True, return_tensors="pt")
            if hasattr(enc, "input_ids"):
                enc = enc.input_ids
            ids = enc.to("cuda")
            out = model.generate(ids, max_new_tokens=MAX_NEW,
                                 do_sample=False, verbose=False)
            text = tok.decode(out[0][ids.shape[1]:],
                              skip_special_tokens=True)
            ok, detail = VT.grade(extract_code(text), t)
            npass += ok
            rows.append({"task_id": t["task_id"], "passed": bool(ok),
                         "detail": detail})
            if (i + 1) % 15 == 0:
                hlog("  [%s] %d/%d graded, passing %d"
                     % (label, i + 1, len(tasks), npass))
        return npass, rows

    receipt = {"experiment": "climber-rung1", "arm": ARM, "model": MODEL,
               "family_sha256": fam.get("family_sha256"),
               "family_file_sha256": FAMILY_FILE_SHA,                                            # G-747/G-748: the parsed bytes' digest
               "family_file_sha256_verified_against_pin": bool(_exp_fam),
               "family_name": FAMILY_NAME,
               "family_provenance": (getattr(_FAM, "PROVENANCE", None) if _FA else {"module": "vex_tasks+vexlib (bound in source)"}),
               "view": VIEW,
               "policy": {"seed": SEED, "rotate_deep": ROTATE, "replay": REPLAY,
                          "steps_per_write": STEPS, "lr": LR,
                          "window_top": top, "window_layers": [top - 6, top + 1],
                          "windows": [list(w) for w in ROT] if ROTATE
                          else [list(LATE)]},
               "base_window_hash": base_hash,
               "revert_certificate_scope": {
                   "tensors_hashed": None,
                   "covers": "every tensor this runner can write "
                             "(all editable windows), with name, shape "
                             "and dtype bound into the digest",
                   "does_not_cover": "parameters outside the editable "
                                     "windows and all buffers - a "
                                     "whole-model guard is the referee "
                                     "lane's phase-guard primitive"},
               "naive_floor_heldout": (getattr(_FAM, "FLOOR", "unknown") if _FA else "7/60 (twist-ignorant reference)")}

    restore()
    if DO_EVAL:
        hlog("BASE held-out measurement (rung-1 precondition)")
        b_pass, b_rows = eval_split(held, "base")
        receipt["base_heldout_pass"] = b_pass
        receipt["base_heldout_rate"] = round(b_pass / len(held), 4)
        receipt["base_rows"] = b_rows
        hlog("BASE held-out: %d/%d = %.3f"
             % (b_pass, len(held), b_pass / len(held)))

    # Generation leaves a large KV/activation cache resident; the first
    # backward pass then OOM'd (twice: plain OOM, then a CUBLAS internal
    # error masking the same exhaustion). Free it before any write.
    gc.collect()
    torch.cuda.empty_cache()
    if ARM == "ROTATE_REPLAY_FULLRANK":
        hlog("free GPU before write: %.1f GiB"
             % (torch.cuda.mem_get_info()[0] / (1 << 30)))
        hlog("ROTATE_REPLAY_FULLRANK writes: %d lessons, rotate=%s replay=%s"
             % (len(teach_tasks), ROTATE, REPLAY))
        corpora = [teach_texts(t) for t in teach_tasks]
        if CHAT_FORM != "0":                                        # FP-644: add (or substitute) the exam-format pair per lesson; replay draws it too
            _chat = [chat_form(t, tok) for t in teach_tasks]
            corpora = [(c + [ch]) if CHAT_FORM == "1" else [ch] for c, ch in zip(corpora, _chat)]
            hlog("pilot: chat form %s -> %d surface forms per lesson" % (CHAT_FORM, len(corpora[0])))
        bills, t0 = [], time.time()
        total_tokens = 0
        for i, texts in enumerate(corpora):
            lo, hi = ROT[i % len(ROT)] if ROTATE else LATE
            LEG_STATE["n_lessons"], LEG_STATE["lesson_index"] = len(corpora), i
            l0, l1, ntok, nstep = write(texts, lo, hi, corpora[:i])
            total_tokens += ntok
            if nstep == 0:
                LEG_STATE.setdefault("zero_lessons", []).append(teach_tasks[i]["task_id"])
            bills.append({"task_id": teach_tasks[i]["task_id"],
                          "window": [lo, hi], "loss_first": (None if l0 is None else round(l0, 4)),
                          "loss_last": (None if l1 is None else round(l1, 4)),
                          "tokens_consumed": ntok, "steps_taken": nstep})
            if (i + 1) % 10 == 0:
                hlog("  wrote %d/%d (last loss %s, leg tokens %d)"
                     % (i + 1, len(corpora), ("%.3f" % l1) if l1 is not None else "none", LEG_STATE["used"]))
            if STOP_AFTER and (i + 1) >= STOP_AFTER:                    # FP-647 Arm A: prefix leg
                hlog("pilot: VXT_STOP_AFTER_LESSON=%d reached after %d lessons; exporting the prefix" % (STOP_AFTER, i + 1))
                break
        receipt["bills"] = bills
        receipt["realized_dose"] = {
            "total_loss_bearing_tokens": total_tokens,
            "tokens_per_lesson": round(total_tokens / max(1, len(bills)), 1),
            "token_unit": "loss-bearing next-token targets (input "
                          "positions minus one per sequence), per G-321",
            "truncation_disclosure": ("this development runner truncates "
                                      "lessons at MAX_TEACH_TOK, which is "
                                      "NOT the full-text calibration the "
                                      "referee dose stage freezes; arms "
                                      "compared under it must share this "
                                      "cap"),
            "dose_tok_budget": DOSE_TOK or None,
            "leg_tok_budget": int(os.environ.get("VXT_LEG_TOK", "0")) or None,
            "leg_tok_used": LEG_STATE["used"],
            "leg_last_batch_rule": "the budget check precedes each optimizer step; a step already begun runs to completion, so the realized total may exceed the cap by at most one step's loss-bearing tokens (<= 2 x VXT_MAXTOK); the realized total is the dose of record",
            "zero_token_lessons": LEG_STATE.get("zero_lessons", []),
            "reservation": (os.environ.get("VXT_RESERVE", "0") == "1"),
            "reserved_quotas": LEG_STATE.get("quotas"),
            "contrast": (None if CONTRAST_PAIRS is None else {"pairs_file": CONTRAST, "pairs_sha256": CONTRAST_SHA, "verified_against_pin": True, "n_pairs": len(CONTRAST_PAIRS),
                                                            "contrast_draws": LEG_STATE.get("contrast_draws", 0), "contrast_loss_bearing_positions": LEG_STATE.get("contrast_used", 0),
                                                            "rule": "one contrast pair per optimizer step (1:1 by draw with the vex draw), chat form, response-only loss; counted in its own budget, never against VXT_LEG_TOK; the lesson's tokens_consumed and the leg dose exclude contrast targets"}),
            "fp647": {"stop_after_lesson": STOP_AFTER or None, "lessons_written": len(bills), "optimizer": OPT_KIND, "freeze_down": FREEZE_DOWN,
                      "project_null": ({"path": PROJECT_NULL, "sha256": NULL_SHA, "verified_against_pin": True,
                                        "energy_removed_fraction": (LEG_STATE.get("proj_energy_removed", 0.0) / LEG_STATE["proj_energy_total"]) if LEG_STATE.get("proj_energy_total") else None,
                                        "rule": "after each optimizer step the realized delta of every trained tensor is projected on its input side (TL rows) onto the complement of the retained key subspace"} if PROJECT_NULL else None),
                      "first_step_records": {fn: hashlib.sha256(open(os.path.join(OUT, fn), "rb").read()).hexdigest() for fn in ("first_step_g1.pt", "first_step_u1.pt") if os.path.exists(os.path.join(OUT, fn))}},
            "chat_form": {"mode": CHAT_FORM, "forms_per_lesson": (5 if CHAT_FORM == "1" else 1 if CHAT_FORM == "only" else 4),
                          "render": "apply_chat_template(user=prompt_<VXT_VIEW>, add_generation_prompt=True, no BOS) + fenced compact worked solution + eos",
                          "loss": "response positions only (per-token mean over next-token positions >= prompt length); raw forms unchanged",
                          "dose_unit_note": "a chat sequence's loss-bearing tokens = its response positions; prompt positions are fed but bear no loss",
                          "chat_draws": LEG_STATE.get("chat_draws", 0), "chat_fed_positions": LEG_STATE.get("chat_fed", 0), "chat_loss_bearing_positions": LEG_STATE.get("chat_loss_bearing", 0),
                          "truncation": "chat sequences are never truncated: a lesson longer than VXT_MAXTOK refuses the leg"},
            "steps_cap_per_lesson": STEPS,
            "max_tok_per_seq": MAX_TEACH_TOK,
            "note": "dose in tokens is the transferable unit (G-318); "
                    "steps/wall are diagnostics only"}
        receipt["write_wall_sec"] = round(time.time() - t0, 1)
        receipt["post_write_hash"] = live_hash()

    if DO_EVAL:
        hlog("POST held-out measurement")
        p_pass, p_rows = eval_split(held, "post")
        receipt["post_heldout_pass"] = p_pass
        receipt["post_heldout_rate"] = round(p_pass / len(held), 4)
        receipt["post_rows"] = p_rows
        receipt["delta_heldout"] = p_pass - receipt.get(
            "base_heldout_pass", 0)
        hlog("POST held-out: %d/%d = %.3f  (delta %+d)"
             % (p_pass, len(held), p_pass / len(held),
                receipt["delta_heldout"]))

    if COLLATERAL:
        hlog("COLLATERAL leg: generating HumanEval+ with the write STILL "
             "INSTALLED (scored separately; rung-0 ref 0.774/0.701)")
        try:
            from evalplus.data import get_human_eval_plus
            probs = get_human_eval_plus()
            keys = sorted(probs.keys())
            fmt = ("Please provide a self-contained Python script that "
                   "solves the following problem in a markdown code "
                   "block:\n```python\n%s\n```\n")
            samples = []
            with torch.no_grad():
                for i, tid in enumerate(keys):
                    enc = tok.apply_chat_template(
                        [{"role": "user", "content": fmt % probs[tid]["prompt"]}],
                        add_generation_prompt=True, return_tensors="pt")
                    if hasattr(enc, "input_ids"):
                        enc = enc.input_ids
                    ids = enc.to("cuda")
                    out = model.generate(ids, max_new_tokens=768,
                                         do_sample=False, verbose=False)
                    txt = tok.decode(out[0][ids.shape[1]:],
                                     skip_special_tokens=True)
                    samples.append({"task_id": tid,
                                    "solution": extract_code(txt)})
                    if (i + 1) % 40 == 0:
                        hlog("  collateral %d/%d" % (i + 1, len(keys)))
            cpath = os.path.join(OUT, "collateral_%s_samples.jsonl" % TAG)
            with open(cpath, "w") as fh:
                for row in samples:
                    fh.write(json.dumps(row) + "\n")
            receipt["collateral_samples_file"] = os.path.basename(cpath)
            receipt["collateral_samples_sha256"] = hashlib.sha256(
                open(cpath, "rb").read()).hexdigest()
            receipt["collateral_reference"] = {
                "rung0_humaneval": 0.774, "rung0_humaneval_plus": 0.701,
                "note": "score these samples with evalplus and compare"}
            hlog("collateral samples written: %s" % cpath)
        except Exception as exc:
            receipt["collateral_error"] = "%s: %s" % (
                type(exc).__name__, str(exc)[:200])
            hlog("collateral leg FAILED (non-fatal): %s" % exc)

    if SAVE_DIR:                                                   # PILOT G2: materialize the WRITTEN model as a bf16 HF checkpoint before the revert
        if os.path.exists(SAVE_DIR):
            raise RuntimeError("PILOT-NO-CLOBBER: %s exists" % SAVE_DIR)
        os.makedirs(SAVE_DIR, exist_ok=False)
        if FREEZE_DOWN:                                                 # FP-647 W3: every W_out must be bit-identical to its start snapshot
            _live = dict(model.named_parameters())
            for n_, t_ in DOWN_SNAP.items():
                if not torch.equal(_live[n_].detach().cpu(), t_):
                    raise RuntimeError("PILOT-FREEZE-REFUSED: W_out changed although VXT_FREEZE_DOWN=1: %s" % n_)
            receipt["freeze_down_verified"] = True
            hlog("pilot: FP-647 W3 freeze verified: %d W_out tensors unchanged" % len(DOWN_SNAP))
        live = dict(model.named_parameters())
        EXPORT_SET = sorted(set(BASE) | set(INHERITED if LOAD_MASTER else []))       # union: this leg's windows + inherited parent windows
        for n_ in EXPORT_SET:                                     # every written tensor is a weight; synthetic biases must still be zero
            if n_.split(".")[3].startswith("b_") or (".mlp." in n_ and n_.split(".")[3].startswith("b_")):
                raise RuntimeError("PILOT-MATERIALIZE-REFUSED: a bias is in the editable set: %s" % n_)
        for n_, p_ in model.named_parameters():
            if ".mlp." in n_ and n_.split(".")[1].isdigit() and n_.split(".")[3].startswith("b_") and float(p_.detach().abs().max()) != 0.0:
                raise RuntimeError("PILOT-MATERIALIZE-REFUSED: synthetic bias became nonzero: %s" % n_)
        master = {n_: live[n_].detach().cpu().float().clone() for n_ in EXPORT_SET}    # the fp32 written-window MASTER (S_k continuation state)
        # G-761: certify the CANDIDATE's inherited preservation BEFORE any revert: the exported master's inherited entries must equal the
        # input master exactly (fp32). This is a property of the exported candidate, separate from the later revert certificate.
        CANDIDATE_INHERITED_EQUAL = (all(torch.equal(master[n_].float(), master_in[n_].float()) for n_ in INHERITED) if (LOAD_MASTER and INHERITED) else None)
        if CANDIDATE_INHERITED_EQUAL is False:
            raise RuntimeError("PILOT-INHERITED-REFUSED: an inherited (never-trained) tensor of the exported candidate differs from the input master")
        torch.save(master, os.path.join(SAVE_DIR, "fp32_master_windows.pt"))
        hf2 = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION, torch_dtype=torch.float32, low_cpu_mem_usage=True)
        sd2 = hf2.state_dict()
        n_copied = 0
        with torch.no_grad():
            for n_ in EXPORT_SET:
                hn, ht = tl_to_hf(n_, live[n_].detach().cpu().float())
                sd2[hn].copy_(ht)
                n_copied += 1
        for n_ in EXPORT_SET:                                       # verify the copy against the live TL tensors
            hn, ht = tl_to_hf(n_, live[n_].detach().cpu().float())
            if not torch.equal(sd2[hn].float(), ht):
                raise RuntimeError("PILOT-MATERIALIZE-REFUSED: mismatch after copy on %s" % n_)
        hf2.to(torch.bfloat16).save_pretrained(SAVE_DIR, safe_serialization=True)
        tok.save_pretrained(SAVE_DIR)
        man = {}
        for fn in sorted(os.listdir(SAVE_DIR)):
            p_ = os.path.join(SAVE_DIR, fn)
            if os.path.isfile(p_):
                man[fn] = hashlib.sha256(open(p_, "rb").read()).hexdigest()
        receipt["materialized_bf16"] = {"dir": SAVE_DIR, "files_sha256": man, "written_state_live_hash": live_hash(), "n_tensors_copied": n_copied, "dtype": "bfloat16",
                                        "note": "written MLP windows copied TL->HF (exact transposes, verified), saved bf16 BEFORE the in-process revert; the fp32 written-window master is saved alongside as fp32_master_windows.pt"}
        with open(os.path.join(SAVE_DIR, "MANIFEST-SHA256.json"), "w") as fh:
            json.dump(receipt["materialized_bf16"], fh, indent=1)
        hlog("pilot: materialized bf16 checkpoint at %s (%d tensors copied, %d files)" % (SAVE_DIR, n_copied, len(man)))
        # post-save RELOAD verification: the saved bf16 tensors must equal bf16(fp32 master) for every written window
        hf3 = AutoModelForCausalLM.from_pretrained(SAVE_DIR, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
        sd3 = hf3.state_dict()
        for n_ in EXPORT_SET:
            hn, ht = tl_to_hf(n_, master[n_])
            if not torch.equal(sd3[hn].to(torch.bfloat16), ht.to(torch.bfloat16)):
                raise RuntimeError("PILOT-RELOAD-REFUSED: reloaded bf16 tensor differs from bf16(master) on %s" % n_)
        receipt["materialized_bf16"]["reload_verified"] = True
        receipt["materialized_bf16"]["base_revision"] = REVISION
        receipt["materialized_bf16"]["loaded_master"] = LOAD_MASTER
        hlog("pilot: bf16 reload verified against the fp32 master on %d tensors" % len(BASE))
        del hf3, sd3
        del hf2, sd2
        gc.collect()
    restore()
    receipt["revert_certificate_scope"]["tensors_hashed"] = n_hashed
    if LOAD_MASTER:
        receipt["parent"] = PARENT_RECORD                                      # pilot v0.4 parent authentication record
        receipt["inherited_windows"] = {"n_tensors": len(INHERITED), "names": INHERITED, "input_master_sha256_fp32": INHERITED_HASH,
                                        "candidate_export_equal_to_input_master": (CANDIDATE_INHERITED_EQUAL if SAVE_DIR else None),
                                        "note": "parent master tensors outside this leg's TRAINED windows: loaded, never in any optimizer; the EXPORTED candidate's inherited entries were compared with the input master BEFORE any revert (G-761); the revert certificate is a separate check over the editable set"}
    receipt["revert_hash"] = live_hash()
    receipt["revert_verified"] = receipt["revert_hash"] == base_hash
    hlog("REVERT hash-exact: %s" % receipt["revert_verified"])

    path = os.path.join(OUT, "%s_rung1_%s_receipt.json" % (FAMILY_NAME, TAG))
    with open(path, "w") as fh:
        json.dump(receipt, fh, indent=1)
    hlog("written %s" % path)
    print("VEX-TEACH-DONE", flush=True)


if __name__ == "__main__":
    main()
