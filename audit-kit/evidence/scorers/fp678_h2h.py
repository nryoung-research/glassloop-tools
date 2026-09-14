#!/usr/bin/env python3
"""FP-678 HEAD-TO-HEAD: the public editors (AlphaEdit / WISE / GRACE from the EasyEdit clone, the FP-627 pinned hyper-parameters) versus OUR writer,
on OUR artifact (Qwen2.5-3B-Instruct aa8e7253), OUR lessons (the vex family, 60 teach tasks, the exam-format lesson our writer trains on), OUR three
screens (closed teach of 60; HumanEval+ plus of 164; the 641-item fp467 protected-panel bill) and OUR receipts, read in ONE process / ONE stack.

Reads are in-process PORTS of the three pinned instruments (verbatim generation / scoring code paths): cap_eval.py (hf backend, closed view, greedy,
512 new tokens), panel_read_hf.py (batch 1, template[0], summed candidate log-probs, margin vs best wrong candidate; same output schema, so
price_state.py bills it unchanged) and climb_harness.py (evalplus HumanEval+, greedy, 768 new tokens, samples jsonl for score_anchor_wsl.sh).
FP678_PORT_CHECK=1 (base arm) runs the REAL cap_eval.py and panel_read_hf.py as subprocesses in the same venv on the same model and refuses unless
their rows equal the port's rows exactly (code sha per task; margins per item).

Env: FP678_ARM base | E2_alphaedit | E3_wise | E4_grace | ours ; FP678_MODEL_DIR (ours: exported A_written_bf16; editors/base: the snapshot);
     FP678_OUT (dir); FP678_TAG (read tag; anchor samples land in climber/out/anchor/climb_<TAG>_samples.jsonl); FP678_LIMIT_LESSONS (0 = all 60);
     FP678_LIMIT_READS (0 = all); FP678_SKIP_ANCHOR=1; FP678_PORT_CHECK=1; FP678_BASE_PANEL (bill the panel read against this base read via price_state.py).
Refusals: an existing report / panel / receipt path (no clobber); AlphaEdit without its cached null-space projection (no in-run rebuild); WISE / GRACE
touching the original layer weight; a request whose subject is not a unique substring of its prompt.
"""
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FP616 = os.path.abspath(os.path.join(HERE, "..", ".."))
CLIMBER = os.path.abspath(os.path.join(FP616, "..", "climber"))
GLASSLOOP = os.path.abspath(os.path.join(FP616, "..", "glassloop"))
SNAP = os.environ.get("FP678_SNAPSHOT", "C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
ARM = os.environ["FP678_ARM"]
MODEL_DIR = os.environ.get("FP678_MODEL_DIR") or SNAP
OUT = os.path.abspath(os.environ.get("FP678_OUT") or os.path.join(HERE, "out", "fp678", ARM))
TAG = os.environ.get("FP678_TAG") or ("fp678_%s_bf16_cfgA" % ARM)
LIMIT_LESSONS = int(os.environ.get("FP678_LIMIT_LESSONS", "0"))
LIMIT_READS = int(os.environ.get("FP678_LIMIT_READS", "0"))
SKIP_ANCHOR = os.environ.get("FP678_SKIP_ANCHOR", "0") == "1"
PORT_CHECK = os.environ.get("FP678_PORT_CHECK", "0") == "1"
BASE_PANEL = os.environ.get("FP678_BASE_PANEL")
FAMILIES = [f for f in os.environ.get("FP678_FAMILIES", "vex").split(",") if f]   # FP-681: extra families read through the same port (module <fam>_tasks, file <fam>_family.json in climber)
DEVICE = "cuda"
DTYPE = "bfloat16"
PANEL = os.path.join(GLASSLOOP, "fp467_panel_v2.json")
BANK = os.path.join(GLASSLOOP, "fp454_bank.json")
FAMILY = os.path.join(CLIMBER, "vex_family.json")
ANCHOR_DIR = os.path.join(CLIMBER, "out", "anchor")
HP = {"E2_alphaedit": os.path.join(FP616, "fp627_hparams", "E2_alphaedit_selected.yaml"),
      "E3_wise": os.path.join(FP616, "fp627_hparams", "E3_wise_selected.yaml"),
      "E4_grace": os.path.join(FP616, "fp627_hparams", "E4_grace_selected.yaml")}
EASYEDIT_ROOT = os.environ.get("FP627_EASYEDIT", r"C:\fp43\EasyEdit")
WORK_ROOT = os.environ.get("FP627_EDITOR_WORK", r"C:\fp43\fp627_work")
CAP_MAX_NEW = 512          # cap_eval.py MAX_NEW
CLIMB_MAX_NEW = 768        # climb_harness.py MAX_NEW
PROMPT_FMT = ("Please provide a self-contained Python script that solves the "
              "following problem in a markdown code block:\n```python\n%s\n```\n")   # climb_harness.py PROMPT_FMT, verbatim
LOC_PROMPTS = (   # fp627_easyedit_adapter._Base.DEFAULT_LOC_PROMPTS (WISE locality rows; GRACE ignores)
    "Q: What is the boiling point of water at sea level in Celsius? A:", "Q: How many days are there in a leap year? A:",
    "Q: Which planet is known as the Red Planet? A:", "Q: What gas do plants absorb from the air for photosynthesis? A:",
    "Q: How many sides does a hexagon have? A:", "Q: What is the chemical symbol for gold? A:",
    "Q: In which direction does the sun rise? A:", "Q: What is the largest ocean on Earth? A:")


def hlog(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ssha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def no_clobber(p):
    if os.path.exists(p):
        raise SystemExit("REFUSED-NO-CLOBBER %s exists" % p)


def extract_code(text):   # cap_eval.py / climb_harness.py, verbatim
    for fence in ("```python", "```"):
        if fence in text:
            return text.split(fence, 1)[1].split("```", 1)[0]
    return text


def tbytes(t):
    return t.detach().to("cpu").contiguous().float().numpy().tobytes()


# ------------------------------------------------------------------ lessons -> editor requests
def build_requests(tok, tasks, VT, worked_solution):
    """One request per teach task: prompt = the chat-rendered closed prompt exactly as cap_eval renders it (add_generation_prompt=True),
    target = the fenced compact worked solution + eos (pilot_teach.chat_form); subject = the task phrase (a unique substring of the prompt;
    AlphaEdit keys on its last token); loc_prompt cycles the FP-627 locality questions (WISE)."""
    reqs = []
    for i, t in enumerate(tasks):
        prompt = t["prompt_closed"]
        rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
        target = "```python\n%s\n```%s" % (worked_solution(t["ops"], compact=True), tok.eos_token)
        subject = VT.phrase(t["ops"])
        if rendered.count(subject) != 1:
            raise SystemExit("REQUEST-SUBJECT-NOT-UNIQUE %s (%d)" % (t["task_id"], rendered.count(subject)))
        if "{" in rendered or "}" in rendered:
            raise SystemExit("REQUEST-PROMPT-HAS-BRACES %s (EasyEdit formats prompts with str.format)" % t["task_id"])
        reqs.append({"case_id": t["task_id"], "prompt": rendered, "target_new": target, "ground_truth": "<|endoftext|>",
                     "subject": subject, "portability": {}, "locality": {}, "loc_prompt": LOC_PROMPTS[i % len(LOC_PROMPTS)]})
    return reqs


def get_module(model, dotted):
    m = model
    for part in dotted.replace("]", "").replace("[", ".").split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


# ------------------------------------------------------------------ editors
def run_editor(arm, model, tok, reqs, receipt):
    sys.path.insert(0, FP616)
    A = importlib.import_module("fp627_easyedit_adapter")      # stub packages + identity helpers only; no edit goes through its witness path
    if EASYEDIT_ROOT not in sys.path:
        sys.path.insert(0, EASYEDIT_ROOT)
    A._register_stub_packages()
    import yaml
    hp = HP[arm]
    raw = yaml.safe_load(open(hp, encoding="utf-8"))
    receipt["editor"] = {"arm": arm, "hparams_path": os.path.basename(hp), "hparams_sha256": fsha(hp), "hparams": raw,
                         "easyedit_root": EASYEDIT_ROOT, "easyedit_git_head": A._git_head(EASYEDIT_ROOT), "adapter_helper_sha256": fsha(A.__file__),
                         "n_requests": len(reqs), "requests_sha256": ssha(json.dumps(reqs, sort_keys=True)),
                         "one_request_per_library_call": arm != "E2_alphaedit"}
    tok.pad_token = "<|endoftext|>"       # fp627_easyedit_adapter._Base.__init__
    tok.padding_side = "right"
    t0, c0 = time.time(), time.process_time()
    torch.cuda.reset_peak_memory_stats()
    if arm == "E2_alphaedit":
        from easyeditor.models.alphaedit import AlphaEditHyperParams, apply_AlphaEdit_to_model
        H = AlphaEditHyperParams.from_hparams(hp)
        work = os.path.join(WORK_ROOT, "alphaedit_qwen2.5-3b_%s" % receipt["editor"]["hparams_sha256"][:8])
        P_path = os.path.join(work, "null_space_project.pt")
        if not os.path.exists(P_path):
            raise SystemExit("ALPHAEDIT-P-MISSING %s (no in-run rebuild of the null-space projection)" % P_path)
        H.P_loc, H.stats_dir = P_path, os.path.join(work, "stats")
        names = ["model.layers.%d.mlp.down_proj" % l for l in H.layers]
        before = {n: get_module(model, n).weight.detach().to("cpu", copy=True).float() for n in names}
        import torch.utils.data as _tud          # fp627_easyedit_adapter: in-process stats loading on Windows (only matters on a rebuild)
        _orig_init = _tud.DataLoader.__init__

        def _init0(self_, *a, **kw):
            kw["num_workers"] = 0
            kw.pop("prefetch_factor", None)
            kw.pop("persistent_workers", None)
            return _orig_init(self_, *a, **kw)
        _tud.DataLoader.__init__ = _init0
        cwd = os.getcwd()
        os.chdir(work)
        try:
            edited, _ = apply_AlphaEdit_to_model(model, tok, reqs, H, copy=False, return_orig_weights=False, keep_original_weight=False)
        finally:
            _tud.DataLoader.__init__ = _orig_init
            os.chdir(cwd)
        if edited is not model:
            raise SystemExit("ALPHAEDIT-RETURNED-A-DIFFERENT-MODEL-OBJECT")
        dose = {}
        for n in names:
            d = get_module(model, n).weight.detach().to("cpu").float() - before[n]
            dose[n] = {"delta_fro": float(d.norm()), "delta_max_abs": float(d.abs().max()), "base_fro": float(before[n].norm())}
        receipt["editor"].update({"layers": list(H.layers), "P_sha256": fsha(P_path), "P_bytes": os.path.getsize(P_path), "P_preexisting": True,
                                  "dose": dose, "dose_total_fro": float(sum(v["delta_fro"] ** 2 for v in dose.values()) ** 0.5),
                                  "deployable_as": "plain weights (down_proj rows of the edited layers)"})
        del before
    elif arm == "E3_wise":
        from easyeditor.models.wise import WISEHyperParams, apply_wise_to_model
        H = WISEHyperParams.from_hparams(hp)
        if not getattr(H, "retrieve", False):
            raise SystemExit("WISE-RETRIEVE-MUST-BE-TRUE")
        try:
            setattr(H, "sequential_edit", True)
        except Exception:
            pass
        wrapped = raw["inner_params"][0].replace(".weight", "")
        orig_w = get_module(model, wrapped).weight.detach().to("cpu", copy=True).float()
        orig_sha = hashlib.sha256(tbytes(get_module(model, wrapped).weight)).hexdigest()
        editor = None
        for r in reqs:                                    # ONE request per library call, lesson order (G-569 / G-577 law)
            o_ = apply_wise_to_model(model, tok, [r], H, copy=False, return_orig_weights=False, keep_original_weight=False)
            editor = o_[0] if isinstance(o_, tuple) else o_
        mod = get_module(model, wrapped)
        found = {}
        for attr in ("original_layer", "layer"):
            sub = getattr(mod, attr, None)
            if sub is not None and hasattr(sub, "weight"):
                found["original_attr"] = attr
                after_sha = hashlib.sha256(tbytes(sub.weight)).hexdigest()
                break
        else:
            after_sha = hashlib.sha256(tbytes(mod.weight)).hexdigest() if hasattr(mod, "weight") else None
        if after_sha != orig_sha:
            raise SystemExit("WISE-TOUCHED-ORIGINAL-LAYER-WEIGHT")
        side = None
        for attr in ("new_weight", "memory_weight"):
            w = getattr(mod, attr, None)
            if isinstance(w, torch.Tensor) and w.shape == orig_w.shape:
                side = (attr, w)
                break
        if side is not None:
            d = side[1].detach().to("cpu").float() - orig_w
            found.update({"side_weight_attr": side[0], "side_delta_fro": float(d.norm()), "side_delta_max_abs": float(d.abs().max())})
        found["module_type"] = type(mod).__name__
        found["tensor_attrs"] = {k: list(v.shape) for k, v in vars(mod).items() if isinstance(v, torch.Tensor)}
        found["memory_mean_act"] = [float(x) for x in getattr(mod, "memory_mean_act", [])] if hasattr(mod, "memory_mean_act") and not isinstance(getattr(mod, "memory_mean_act"), torch.Tensor) else None
        receipt["editor"].update({"layer": wrapped, "original_weight_sha256": after_sha, "retrieve": bool(H.retrieve), "state": found,
                                  "deployable_as": "base weights + side memory + activation router (hooked module)"})
    elif arm == "E4_grace":
        from easyeditor.models.grace import GraceHyperParams, apply_grace_to_model
        H = GraceHyperParams.from_hparams(hp)
        wrapped = raw["inner_params"][0].replace(".weight", "")
        orig_sha = hashlib.sha256(tbytes(get_module(model, wrapped).weight)).hexdigest()
        for r in reqs:
            apply_grace_to_model(model, tok, [r], H, copy=False, return_orig_weights=False, keep_original_weight=False)
        mod = get_module(model, wrapped)
        if not hasattr(mod, "key_id"):
            raise SystemExit("GRACE-ADAPTER-NOT-INSTALLED %s" % type(mod).__name__)
        sub = getattr(mod, "layer", None)
        after_sha = hashlib.sha256(tbytes(sub.weight)).hexdigest() if sub is not None and hasattr(sub, "weight") else None
        if after_sha != orig_sha:
            raise SystemExit("GRACE-TOUCHED-ORIGINAL-LAYER-WEIGHT")
        # fp627_easyedit_adapter._install_key_reset (library quirk: key_id sticks to the first prompt's length; reset on every new prompt)
        orig_forward = mod.forward

        def forward(*args, **kw):
            if args and hasattr(args[0], "shape") and len(args[0].shape) >= 2:
                mod.key_id = -1 if int(args[0].shape[1]) > 1 else int(mod.key_id)
            return orig_forward(*args, **kw)
        forward._fp627_key_reset = True
        mod.forward = forward
        keys = getattr(mod, "keys", None)
        vals = getattr(mod, "values", None)
        st = {"n_keys": (int(keys.shape[0]) if isinstance(keys, torch.Tensor) else (len(keys) if keys is not None else None)),
              "n_values": (int(vals.shape[0]) if isinstance(vals, torch.Tensor) else (len(vals) if vals is not None else None)),
              "epsilons": ([float(e) for e in mod.epsilons] if hasattr(mod, "epsilons") else None), "module_type": type(mod).__name__,
              "value_norm_mean": (float(vals.float().norm(dim=-1).mean()) if isinstance(vals, torch.Tensor) and vals.numel() else None)}
        receipt["editor"].update({"layer": wrapped, "original_weight_sha256": after_sha, "grace_key_reset_per_prompt": True, "state": st,
                                  "deployable_as": "base weights + codebook adapter (hooked module)"})
    else:
        raise SystemExit("NO-EDITOR " + arm)
    receipt["editor"].update({"wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1), "peak_gpu_bytes": int(torch.cuda.max_memory_allocated())})
    hlog("EDIT %s done in %.0fs peak %.1f GiB" % (arm, receipt["editor"]["wall_s"], receipt["editor"]["peak_gpu_bytes"] / 2 ** 30))


# ------------------------------------------------------------------ the three reads (ports)
def gen_cap(tok, model, prompt):          # cap_eval.py hf generate, verbatim
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    ids = enc.to(DEVICE)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=CAP_MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def cap_read(tok, model, VT, fam, split, tag, model_name, backend_info, fam_name="vex", fam_file=None):
    tasks = [dict(t, split=split) for t in fam[split]]
    if LIMIT_READS:
        tasks = tasks[:LIMIT_READS]
    fam_file = fam_file or FAMILY
    path = os.path.join(OUT, "cap_%s_%s_%s_report.json" % (fam_name, tag, split))
    no_clobber(path)
    rows, npass, t0 = [], 0, time.time()
    for i, t in enumerate(tasks):
        code = extract_code(gen_cap(tok, model, t["prompt_closed"]))
        ok, detail = VT.grade(code, t)
        npass += ok
        rows.append({"task_id": t["task_id"], "ops": t["ops"], "split": t["split"], "passed": bool(ok), "detail": detail, "code_sha256": ssha(code)})
        if (i + 1) % 10 == 0:
            hlog("  %s %d/%d graded, passing %d (%.1fs/task)" % (split, i + 1, len(tasks), npass, (time.time() - t0) / (i + 1)))
    report = {"instrument": "cap-eval-v1 (fp678 in-process port)", "family_module": "%s_tasks" % fam_name, "family_file": fam_file, "family_file_sha256": fsha(fam_file),
              "model": model_name, "revision": None, "tag": "%s_%s" % (tag, split), "split": split, "view": "closed", "n_tasks": len(tasks), "n_passed": npass,
              "pass_rate": round(npass / max(1, len(tasks)), 4), "config": {"greedy": True, "max_new_tokens": CAP_MAX_NEW, "dtype": DTYPE, "torch": torch.__version__},
              "backend_info": backend_info, "wall_sec": round(time.time() - t0, 1), "rows": rows}
    json.dump(report, open(path, "w"), indent=1, sort_keys=True)
    hlog("CAP %s %s %s: %d/%d -> %s" % (fam_name, tag, split, npass, len(tasks), path))
    return report, path


def panel_read(tok, model, model_name, tag, base_dtypes=None):
    panel = json.load(open(PANEL, encoding="utf-8"))
    bank = {it["id"]: it for it in json.load(open(BANK, encoding="utf-8"))["items"]}
    ids = [r["id"] for r in panel["items"]]
    if LIMIT_READS:
        ids = ids[:max(LIMIT_READS, 24)]
    path = os.path.join(OUT, "L2_%s_panel.json" % tag)
    no_clobber(path)
    t0 = time.time()
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    dtypes_rec = base_dtypes or dtypes      # the artifact's weight precision (bf16); a side-memory editor's fp32 codebook is reported separately (price_state compares this field)
    lps, margins = {}, {}
    with torch.no_grad():
        for i in ids:
            it = bank[i]
            row = {}
            for c in it["candidates"]:
                x = torch.tensor([tok(it["templates"][0].replace("{answer}", c), add_special_tokens=False)["input_ids"]], dtype=torch.long, device=DEVICE)
                lsm = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), dim=-1)
                row[c] = float(lsm.gather(-1, x[0, 1:].unsqueeze(-1)).sum())
            lps[i] = row
            g = row[it["answer"]]
            margins[i] = g - max(v for c, v in row.items() if c != it["answer"])
    rec = {"kind": "panel_read_hf", "ported_by": "fp678_h2h", "model": model_name, "revision": None, "dtype_requested": DTYPE, "param_dtypes": dtypes_rec, "param_dtypes_live": dtypes,
           "panel_sha256": fsha(PANEL), "bank_sha256": fsha(BANK), "n": len(ids), "batch": 1, "margins": margins, "logprobs": lps, "torch": torch.__version__,
           "elapsed_s": round(time.time() - t0, 1), "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    b = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()
    with open(path, "xb") as f:
        f.write(b)
    hlog("PANEL %s n %d positive margins %d -> %s" % (tag, len(ids), sum(1 for v in margins.values() if v > 0), path))
    return rec, path


def anchor_gen(tok, model, model_name, tag):
    from evalplus.data import get_human_eval_plus, write_jsonl
    problems = get_human_eval_plus()
    keys = sorted(problems.keys())
    if LIMIT_READS:
        keys = keys[:LIMIT_READS]
    os.makedirs(ANCHOR_DIR, exist_ok=True)
    spath = os.path.join(ANCHOR_DIR, "climb_%s_samples.jsonl" % tag)
    rpath = os.path.join(ANCHOR_DIR, "climb_%s_receipt.json" % tag)
    no_clobber(spath)
    no_clobber(rpath)
    samples, rows, t0 = [], [], time.time()
    for i, tid in enumerate(keys):
        enc = tok.apply_chat_template([{"role": "user", "content": PROMPT_FMT % problems[tid]["prompt"]}], add_generation_prompt=True, return_tensors="pt")
        if hasattr(enc, "input_ids"):
            enc = enc.input_ids
        ids = enc.to(DEVICE)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=CLIMB_MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
        code = extract_code(tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
        samples.append({"task_id": tid, "solution": code})
        rows.append({"task_id": tid, "sample_sha256": ssha(code)})
        if (i + 1) % 20 == 0:
            hlog("  anchor %d/%d (%.1fs avg)" % (i + 1, len(keys), (time.time() - t0) / (i + 1)))
    write_jsonl(spath, samples)
    receipt = {"harness": "climber-v0 (fp678 in-process port)", "benchmark": "humaneval-plus", "model": model_name, "tag": tag, "n_tasks": len(keys), "limit": LIMIT_READS,
               "config": {"greedy": True, "max_new_tokens": CLIMB_MAX_NEW, "dtype": DTYPE, "template": "chat+evalplus-instruct-v0"}, "wall_sec": round(time.time() - t0, 1),
               "samples_file": os.path.basename(spath), "samples_sha256": fsha(spath), "rows": rows}
    json.dump(receipt, open(rpath, "w"), indent=1)
    hlog("ANCHOR samples %s (%s) -> score with score_anchor_wsl.sh %s" % (spath, receipt["samples_sha256"][:16], tag))
    return receipt, spath


def port_check(cap_report, panel_rec):
    """Run the REAL instruments (same venv, same model dir) and compare rows exactly."""
    py = sys.executable
    tmp = os.path.join(OUT, "portcheck")
    os.makedirs(tmp, exist_ok=True)
    env = dict(os.environ, CAP_MODEL=MODEL_DIR, CAP_FAMILY_MODULE="vex_tasks", CAP_FAMILY=FAMILY, CAP_SPLIT="heldout", CAP_VIEW="closed", CAP_OUT=tmp,
               CAP_TAG="fp678portcheck", CAP_LIMIT=str(len(cap_report["rows"])), CAP_DTYPE=DTYPE, CAP_BACKEND="hf")
    subprocess.run([py, os.path.join(CLIMBER, "cap_eval.py")], cwd=CLIMBER, env=env, check=True, stdout=open(os.path.join(tmp, "cap_eval.log"), "w"), stderr=subprocess.STDOUT)
    real = json.load(open(os.path.join(tmp, "cap_vex_fp678portcheck_report.json")))
    a = [(r["task_id"], r["passed"], r["code_sha256"]) for r in real["rows"]]
    b = [(r["task_id"], r["passed"], r["code_sha256"]) for r in cap_report["rows"]]
    cap_ok = a == b
    pout = os.path.join(tmp, "panel_real.json")
    subprocess.run([py, os.path.join(HERE, "panel_read_hf.py"), "--model", MODEL_DIR, "--dtype", DTYPE, "--panel", PANEL, "--bank", BANK, "--out", pout, "--limit", str(panel_rec["n"])],
                   cwd=HERE, env=dict(os.environ), check=True, stdout=open(os.path.join(tmp, "panel.log"), "w"), stderr=subprocess.STDOUT)
    realp = json.load(open(pout))
    diffs = [abs(realp["margins"][k] - panel_rec["margins"][k]) for k in panel_rec["margins"]]
    panel_ok = set(realp["margins"]) == set(panel_rec["margins"]) and max(diffs) <= 1e-9
    res = {"cap_rows_equal": cap_ok, "cap_n": len(b), "panel_items_equal": panel_ok, "panel_n": len(diffs), "panel_max_abs_diff": max(diffs), "real_cap_report_sha256": fsha(os.path.join(tmp, "cap_vex_fp678portcheck_report.json")), "real_panel_sha256": fsha(pout)}
    hlog("PORT-CHECK %s %s" % ("PASS" if cap_ok and panel_ok else "FAIL", res))
    if not (cap_ok and panel_ok):
        raise SystemExit("PORT-CHECK-FAILED")
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    rpath = os.path.join(OUT, "fp678_%s_receipt.json" % ARM)
    no_clobber(rpath)
    sys.path.insert(0, CLIMBER)
    VT = importlib.import_module("vex_tasks")
    worked_solution = importlib.import_module("vex_teach").worked_solution
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    fam = json.load(open(FAMILY, encoding="utf-8"))
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=getattr(torch, DTYPE), low_cpu_mem_usage=True).to(DEVICE)
    model.eval()
    base_dtypes = sorted({str(p.dtype) for p in model.parameters()})
    msha = fsha(os.path.join(MODEL_DIR, "model.safetensors")) if os.path.exists(os.path.join(MODEL_DIR, "model.safetensors")) else None
    manifest = None
    if os.path.exists(os.path.join(MODEL_DIR, "MANIFEST-SHA256.json")):
        manifest = json.load(open(os.path.join(MODEL_DIR, "MANIFEST-SHA256.json")))["files_sha256"].get("model.safetensors")
        if manifest != msha:
            raise SystemExit("ARRIVED-MISMATCH model.safetensors %s != manifest %s" % (msha, manifest))
    receipt = {"kind": "fp678_h2h_receipt", "arm": ARM, "tag": TAG, "model_dir": MODEL_DIR, "model_safetensors_sha256": msha, "manifest_sha256": manifest,
               "snapshot": SNAP, "stack": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__,
                                          "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0), "dtype": DTYPE},
               "family_sha256": fsha(FAMILY), "panel_sha256": fsha(PANEL), "bank_sha256": fsha(BANK), "runner_sha256": fsha(os.path.abspath(__file__)),
               "limits": {"lessons": LIMIT_LESSONS, "reads": LIMIT_READS, "skip_anchor": SKIP_ANCHOR}, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    hlog("FP678 arm %s model %s (%s) stack torch %s transformers %s" % (ARM, MODEL_DIR, (msha or "hub")[:16], torch.__version__, transformers.__version__))
    if ARM.startswith("E"):
        tasks = fam["teach"][:LIMIT_LESSONS] if LIMIT_LESSONS else fam["teach"]
        reqs = build_requests(tok, tasks, VT, worked_solution)
        receipt["lessons"] = {"n": len(reqs), "ids": [r["case_id"] for r in reqs], "form": "chat-rendered prompt_closed -> fenced compact worked solution + eos (pilot_teach.chat_form); subject = vex_tasks.phrase(ops)"}
        run_editor(ARM, model, tok, reqs, receipt)
        model.eval()
    backend_info = {"backend": "hf", "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}), "torch": torch.__version__, "in_process": True, "arm": ARM}
    teach, tpath = cap_read(tok, model, VT, fam, "teach", TAG, MODEL_DIR, backend_info)
    held, hpath = cap_read(tok, model, VT, fam, "heldout", TAG, MODEL_DIR, backend_info)
    panel, ppath = panel_read(tok, model, MODEL_DIR, TAG, base_dtypes)
    receipt["param_dtypes"] = {"at_load": base_dtypes, "after_edit": panel["param_dtypes_live"]}
    receipt["reads"] = {"teach_vex": teach["n_passed"], "teach_n": teach["n_tasks"], "heldout_vex": held["n_passed"], "heldout_n": held["n_tasks"],
                        "teach_report_sha256": fsha(tpath), "heldout_report_sha256": fsha(hpath), "panel_read_sha256": fsha(ppath), "panel_n": panel["n"],
                        "panel_positive_margins": sum(1 for v in panel["margins"].values() if v > 0)}
    for fam_name in [f for f in FAMILIES if f != "vex"]:       # FP-681: additional families (e.g. quill) through the same port
        FT2 = importlib.import_module("%s_tasks" % fam_name)
        ff = os.path.join(CLIMBER, "%s_family.json" % fam_name)
        fam2 = json.load(open(ff, encoding="utf-8"))
        t2, tp2 = cap_read(tok, model, FT2, fam2, "teach", TAG, MODEL_DIR, backend_info, fam_name, ff)
        h2, hp2 = cap_read(tok, model, FT2, fam2, "heldout", TAG, MODEL_DIR, backend_info, fam_name, ff)
        receipt["reads"].update({"teach_%s" % fam_name: t2["n_passed"], "heldout_%s" % fam_name: h2["n_passed"], "%s_family_sha256" % fam_name: fsha(ff),
                                 "%s_teach_report_sha256" % fam_name: fsha(tp2), "%s_heldout_report_sha256" % fam_name: fsha(hp2)})
    if BASE_PANEL:
        bout = os.path.join(OUT, "L2_%s_price_vs_base.json" % TAG)
        subprocess.run([sys.executable, os.path.join(HERE, "price_state.py"), "--base", BASE_PANEL, "--state", ppath, "--out", bout], check=True)
        pr = json.load(open(bout))
        receipt["reads"].update({"bill": pr["damage_bill_nats"], "crossings": pr["n_sign_crossings"], "sign_crossings": pr["sign_crossings"],
                                 "price_sha256": fsha(bout), "base_panel_sha256": fsha(BASE_PANEL)})
        hlog("PRICE %s bill %s crossings %s" % (TAG, receipt["reads"]["bill"], receipt["reads"]["crossings"]))
    if not SKIP_ANCHOR:
        arec, spath = anchor_gen(tok, model, MODEL_DIR, TAG)
        receipt["reads"].update({"anchor_samples_sha256": arec["samples_sha256"], "anchor_n": arec["n_tasks"], "anchor_samples": spath})
    if PORT_CHECK:
        receipt["port_check"] = port_check(held, panel)
    receipt.update({"wall_s": round(time.time() - t0, 1), "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
    json.dump(receipt, open(rpath, "w"), indent=1, sort_keys=True)
    hlog("FP678-ARM-DONE %s teach %d/%d heldout %d/%d receipt %s" % (ARM, teach["n_passed"], teach["n_tasks"], held["n_passed"], held["n_tasks"], fsha(rpath)[:16]))


if __name__ == "__main__":
    main()
