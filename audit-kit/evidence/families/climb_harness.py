#!/usr/bin/env python3
"""CLIMBER HARNESS v0 - the coding-leaderboard scorer for the weekly
coding model (THE REFOCUS 2026-08-29).

PURPOSE: a frozen, local, receipt-producing evaluation of a model
checkpoint on HumanEval+ (evalplus), so every weekly rung of the
climber is scored by the SAME instrument with the SAME config, and the
baseline-vs-written comparison is an increment over a measured base -
never a leaderboard screenshot.

v0 SCOPE: greedy pass@1 on HumanEval+ via evalplus. MBPP+ and
LiveCodeBench-lite are later rungs. This file only GENERATES samples
and writes a receipt; scoring runs through `evalplus.evaluate` (its
own sandboxed executor) as a separate step, so generation and grading
stay independent.

FROZEN CONFIG (v0): greedy (do_sample=False), max_new_tokens=768,
bf16, instruct chat template with the evalplus instruction prefix,
one sample per task. Any change = a new versioned harness, disclosed.

Env: CLIMB_MODEL (default Qwen/Qwen2.5-3B-Instruct), CLIMB_OUT,
     CLIMB_LIMIT (0 = all tasks; >0 = smoke subset), CLIMB_DEVICE,
     CLIMB_TAG.
Receipt: config + model identity + per-task sample sha256 + wall time.
"""
import hashlib
import json
import os
import time

MODEL = os.environ.get("CLIMB_MODEL", "Qwen/Qwen2.5-3B-Instruct")
TAG = os.environ.get("CLIMB_TAG", "base")
OUTDIR = os.environ.get("CLIMB_OUT") or os.path.expanduser("~/climber")
LIMIT = int(os.environ.get("CLIMB_LIMIT", "0"))
DEVICE = os.environ.get("CLIMB_DEVICE", "cuda")
DTYPE = os.environ.get("CLIMB_DTYPE", "bfloat16")     # pilot P1 band: cfgA = bfloat16 (default, unchanged), cfgB = float32; recorded in the receipt
MAX_NEW = 768

PROMPT_FMT = (
    "Please provide a self-contained Python script that solves the "
    "following problem in a markdown code block:\n```python\n%s\n```\n")


def hlog(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def extract_code(text):
    """First fenced python block, else first fenced block, else raw."""
    for fence in ("```python", "```"):
        if fence in text:
            body = text.split(fence, 1)[1]
            return body.split("```", 1)[0]
    return text


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from evalplus.data import get_human_eval_plus, write_jsonl
    problems = get_human_eval_plus()
    keys = sorted(problems.keys())
    if LIMIT:
        keys = keys[:LIMIT]
    hlog("model=%s tasks=%d device=%s" % (MODEL, len(keys), DEVICE))
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=getattr(torch, DTYPE), low_cpu_mem_usage=True
    ).to(DEVICE)
    model.eval()
    samples, receipt_rows = [], []
    t0 = time.time()
    for i, tid in enumerate(keys):
        prompt = PROMPT_FMT % problems[tid]["prompt"]
        msgs = [{"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt")
        if hasattr(enc, "input_ids"):  # BatchEncoding (transformers 5.x)
            enc = enc.input_ids
        ids = enc.to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=MAX_NEW, do_sample=False,
                pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        code = extract_code(text)
        samples.append({"task_id": tid, "solution": code})
        receipt_rows.append({
            "task_id": tid,
            "sample_sha256": hashlib.sha256(
                code.encode("utf-8")).hexdigest()})
        if (i + 1) % 10 == 0:
            hlog("  %d/%d (%.1fs avg)" % (i + 1, len(keys),
                                          (time.time() - t0) / (i + 1)))
    spath = os.path.join(OUTDIR, "climb_%s_samples.jsonl" % TAG)
    write_jsonl(spath, samples)
    receipt = {
        "harness": "climber-v0", "benchmark": "humaneval-plus",
        "model": MODEL, "tag": TAG, "n_tasks": len(keys),
        "limit": LIMIT, "config": {
            "greedy": True, "max_new_tokens": MAX_NEW, "dtype":
            DTYPE, "template": "chat+evalplus-instruct-v0"},
        "wall_sec": round(time.time() - t0, 1),
        "samples_file": os.path.basename(spath),
        "samples_sha256": hashlib.sha256(
            open(spath, "rb").read()).hexdigest(),
        "rows": receipt_rows}
    rpath = os.path.join(OUTDIR, "climb_%s_receipt.json" % TAG)
    with open(rpath, "w") as fh:
        json.dump(receipt, fh, indent=1)
    hlog("samples %s receipt %s" % (spath, rpath))
    hlog("score with: evalplus.evaluate --dataset humaneval "
         "--samples %s" % spath)
    print("CLIMB-GEN-DONE", flush=True)


if __name__ == "__main__":
    main()
