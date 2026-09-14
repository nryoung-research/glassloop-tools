#!/usr/bin/env python3
"""CAPABILITY EVALUATOR for the accumulation pilot: family-agnostic version of vex_eval.py (same generation, extraction and grading path).
Measures a model's held-out (or teach) pass rate on a frozen task family by greedy generation and executing the task's unit tests through
the family's own grader (disposable subprocess; expected outputs never sent to the child). Works for vex_tasks and quill_tasks, on a
Hugging Face model id or a local checkpoint directory (the pilot's materialized bf16 checkpoints), at a declared dtype.
Env: CAP_MODEL (id or path), CAP_REVISION (optional), CAP_FAMILY_MODULE (vex_tasks|quill_tasks), CAP_FAMILY (family json path), CAP_SPLIT
(heldout|teach|both), CAP_VIEW (closed|open), CAP_OUT, CAP_TAG, CAP_LIMIT, CAP_DEVICE, CAP_DTYPE (bfloat16|float32)."""
import hashlib
import importlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MODEL = os.environ.get("CAP_MODEL", "Qwen/Qwen2.5-3B-Instruct")
REV = os.environ.get("CAP_REVISION") or None
FAM_MOD = os.environ.get("CAP_FAMILY_MODULE", "vex_tasks")
FAMILY = os.environ.get("CAP_FAMILY") or os.path.join(HERE, "%s_family.json" % FAM_MOD.split("_")[0])
SPLIT = os.environ.get("CAP_SPLIT", "heldout")
VIEW = os.environ.get("CAP_VIEW", "closed").lower()
OUT = os.environ.get("CAP_OUT") or os.path.join(HERE, "out")
TAG = os.environ.get("CAP_TAG", "base")
LIMIT = int(os.environ.get("CAP_LIMIT", "0"))
DEVICE = os.environ.get("CAP_DEVICE", "cuda")
DTYPE = os.environ.get("CAP_DTYPE", "bfloat16")
BACKEND = os.environ.get("CAP_BACKEND", "hf").lower()            # hf | llamacpp  (G-734: packed generation + grading for E1/E1'/E2/E5)
GGUF = os.environ.get("CAP_GGUF")                                  # llamacpp backend: the GGUF file (its sha is receipted)
MAX_NEW = 512
if BACKEND == "llamacpp" and os.path.isdir("C:/fp43/llamacpp-bin"):
    os.environ["PATH"] = "C:\\fp43\\llamacpp-bin;" + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory("C:\\fp43\\llamacpp-bin")


def hlog(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def extract_code(text):
    for fence in ("```python", "```"):
        if fence in text:
            return text.split(fence, 1)[1].split("```", 1)[0]
    return text


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    FT = importlib.import_module(FAM_MOD)
    fam = json.load(open(FAMILY, encoding="utf-8"))
    tasks = []
    if SPLIT in ("heldout", "both"):
        tasks += [dict(t, split="heldout") for t in fam["heldout"]]
    if SPLIT in ("teach", "both"):
        tasks += [dict(t, split="teach") for t in fam["teach"]]
    if LIMIT:
        tasks = tasks[:LIMIT]
    hlog("backend=%s model=%s rev=%s dtype=%s gguf=%s family=%s (%s) tasks=%d split=%s view=%s" % (BACKEND, MODEL, REV, DTYPE, GGUF, FAM_MOD, fsha(FAMILY)[:12], len(tasks), SPLIT, VIEW))
    backend_info = {"backend": BACKEND}
    if BACKEND == "hf":
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=getattr(torch, DTYPE), low_cpu_mem_usage=True).to(DEVICE)
        model.eval()
        backend_info.update({"param_dtypes": sorted({str(p.dtype) for p in model.parameters()}), "torch": torch.__version__})

        def generate(prompt):
            enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
            if hasattr(enc, "input_ids"):
                enc = enc.input_ids
            ids = enc.to(DEVICE)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=MAX_NEW, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    elif BACKEND == "llamacpp":
        if not GGUF:
            raise SystemExit("CAP_GGUF required for the llamacpp backend")
        import llama_cpp
        from llama_cpp import Llama
        # the HF tokenizer renders the chat template exactly as the HF backend does; llama.cpp then tokenizes that rendered prompt (verified per task)
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
        llm = Llama(model_path=GGUF, n_gpu_layers=-1, n_ctx=2048, verbose=False, seed=0)
        backend_info.update({"gguf": GGUF, "gguf_sha256": fsha(GGUF), "llama_cpp_python": llama_cpp.__version__, "n_ctx": 2048})

        def generate(prompt):
            rendered = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False)
            hf_ids = tok(rendered, add_special_tokens=False)["input_ids"]
            lc_ids = llm.tokenize(rendered.encode("utf-8"), add_bos=False, special=True)
            if list(hf_ids) != list(lc_ids):
                raise SystemExit("REFUSED-TOKENIZATION-MISMATCH in prompt (%d vs %d tokens)" % (len(hf_ids), len(lc_ids)))
            out = llm.create_completion(prompt=lc_ids, max_tokens=MAX_NEW, temperature=0.0, top_k=1, seed=0, stop=None)
            return out["choices"][0]["text"]
    else:
        raise SystemExit("unknown CAP_BACKEND %r" % BACKEND)
    rows, npass = [], 0
    t0 = time.time()
    for i, t in enumerate(tasks):
        prompt = t["prompt_closed"] if VIEW == "closed" else t["prompt_open"]
        text = generate(prompt)
        code = extract_code(text)
        ok, detail = FT.grade(code, t)
        npass += ok
        rows.append({"task_id": t["task_id"], "ops": t["ops"], "split": t["split"], "passed": bool(ok), "detail": detail, "code_sha256": hashlib.sha256(code.encode("utf-8")).hexdigest()})
        if (i + 1) % 10 == 0:
            hlog("  %d/%d graded, passing %d (%.1fs/task)" % (i + 1, len(tasks), npass, (time.time() - t0) / (i + 1)))
    report = {"instrument": "cap-eval-v1", "family_module": FAM_MOD, "family_file": os.path.abspath(FAMILY), "family_file_sha256": fsha(FAMILY), "model": MODEL, "revision": REV, "tag": TAG,
              "split": SPLIT, "view": VIEW, "n_tasks": len(tasks), "n_passed": npass, "pass_rate": round(npass / max(1, len(tasks)), 4),
              "config": {"greedy": True, "max_new_tokens": MAX_NEW, "dtype": DTYPE, "torch": torch.__version__}, "backend_info": backend_info, "wall_sec": round(time.time() - t0, 1), "rows": rows}
    path = os.path.join(OUT, "cap_%s_%s_report.json" % (FAM_MOD.split("_")[0], TAG))
    if os.path.exists(path):
        raise SystemExit("REFUSED-NO-CLOBBER %s exists" % path)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    hlog("CAP %s %s: %d/%d = %.3f -> %s" % (FAM_MOD, TAG, npass, len(tasks), report["pass_rate"], path))
    print("CAP-EVAL-DONE", flush=True)


if __name__ == "__main__":
    main()
