#!/usr/bin/env python3
"""VEX ROTATE_REPLAY_FULLRANK WRITE ARM - CLIMBER rung 1.

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


def teach_texts(task):
    """Surface forms of one lesson (the FP-616 fact_texts pattern at
    capability scale). COMPACT: each carries only the helpers this
    pipeline uses, so nothing is truncated by the token cap."""
    sol = worked_solution(task["ops"], compact=True)
    return [
        V.SPEC,
        "Using the vex semantics, %s:\n\n```python\n%s\n```"
        % (VT.phrase(task["ops"]), sol),
        "```python\n%s\n```" % sol,
        "vex pipeline: %s\n\n```python\n%s\n```"
        % (VT.phrase(task["ops"]), sol),
    ]


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
    fam = json.load(open(FAMILY, encoding="utf-8"))
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

    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL, hf_model=hf, dtype=torch.float32, device="cuda")
    del hf
    gc.collect()
    torch.cuda.empty_cache()
    tok = model.tokenizer
    nl = model.cfg.n_layers
    top = nl - 2
    ROT = [(top - 6, top - 5), (top - 4, top - 3),
           (top - 2, top - 1), (top, top + 1)]
    LATE = (int(nl * 0.82), int(nl * 0.94))
    for _, p in model.named_parameters():
        p.requires_grad_(False)

    def win_params(lo, hi):
        return [(n, p) for n, p in model.named_parameters()
                if ".mlp." in n and n.split(".")[1].isdigit()
                and lo <= int(n.split(".")[1]) <= hi]

    ALL = sorted({n for w in (ROT + [LATE]) for n, _ in win_params(*w)})
    BASE = {n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if n in ALL}
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

    rng = random.Random(20260829)

    def write(texts, lo, hi, priors):
        wp = win_params(lo, hi)
        for _, p in wp:
            p.requires_grad_(True)
        opt = torch.optim.AdamW([p for _, p in wp], lr=LR,
                                weight_decay=0.0)
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
        for _ in range(STEPS):
            if DOSE_TOK and tok_used >= DOSE_TOK:
                break
            # Sequence-at-a-time with gradient accumulation. Batching a
            # new-fact text and a replay text together doubled peak
            # activation memory and OOM'd the fp32 3B backward; the
            # accumulated gradient is identical (mean of per-sequence
            # losses) at roughly half the peak.
            seqs = [rng.choice(texts)]
            if REPLAY and priors:
                seqs.append(rng.choice(rng.choice(priors)))
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in seqs:
                toks = model.to_tokens([s])
                if toks.shape[1] > MAX_TEACH_TOK:
                    toks = toks[:, :MAX_TEACH_TOK]
                # LOSS-BEARING positions, not raw input positions
                # (G-321): the first token of a sequence carries no
                # next-token target, so an input-token count overstates
                # the causal training dose by one per sequence and can
                # equate two arms that received unequal supervision.
                tok_used += max(0, int(toks.shape[1]) - 1)
                loss = model(toks, return_type="loss") / len(seqs)
                loss.backward()
                total += float(loss)
                del toks, loss
            opt.step()
            losses.append(total)
        torch.set_grad_enabled(False)
        for _, p in wp:
            p.requires_grad_(False)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
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
               "view": VIEW,
               "policy": {"rotate_deep": ROTATE, "replay": REPLAY,
                          "steps_per_write": STEPS, "lr": LR,
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
               "naive_floor_heldout": "7/60 (twist-ignorant reference)"}

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
        bills, t0 = [], time.time()
        total_tokens = 0
        for i, texts in enumerate(corpora):
            lo, hi = ROT[i % len(ROT)] if ROTATE else LATE
            l0, l1, ntok, nstep = write(texts, lo, hi, corpora[:i])
            total_tokens += ntok
            bills.append({"task_id": teach_tasks[i]["task_id"],
                          "window": [lo, hi], "loss_first": round(l0, 4),
                          "loss_last": round(l1, 4),
                          "tokens_consumed": ntok, "steps_taken": nstep})
            if (i + 1) % 10 == 0:
                hlog("  wrote %d/%d (last loss %.3f)"
                     % (i + 1, len(corpora), l1))
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

    restore()
    receipt["revert_certificate_scope"]["tensors_hashed"] = n_hashed
    receipt["revert_hash"] = live_hash()
    receipt["revert_verified"] = receipt["revert_hash"] == base_hash
    hlog("REVERT hash-exact: %s" % receipt["revert_verified"])

    path = os.path.join(OUT, "vex_rung1_%s_receipt.json" % TAG)
    with open(path, "w") as fh:
        json.dump(receipt, fh, indent=1)
    hlog("written %s" % path)
    print("VEX-TEACH-DONE", flush=True)


if __name__ == "__main__":
    main()
