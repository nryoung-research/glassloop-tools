# -*- coding: utf-8 -*-
"""Closed-book baseline over a keyed-hash FreshQA split (fq_split.py):
generate an answer for every item in every population (headline_teach,
headline_hidden, side_channel), score it with the deterministic
fq_score meter (containment; no LLM judging), and report per-population
and per-effective-year aggregates. The by-effective-year table is the
knowledge-decay readout of a frozen model.

Generation: transformer_lens, pinned greedy config, exact prompts
logged. Optionally reloads an edited checkpoint (--ckpt) so taught and
untaught arms share one instrument. Zero-shot only.

Loader note: default bf16_np = from_pretrained_no_processing + bfloat16.
The naive fp32 from_pretrained path spikes ~4x model size during weight
processing and OOM-killed a 7B run at 113GB resident (documented
incident, 2026-08-18); use fp32 only for small models on big-RAM boxes.
"""
import argparse
import hashlib
import json
import os
import sys
import time

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import fq_score  # noqa: E402

MAX_NEW = 128
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SYSTEM = ("You are a helpful assistant. Answer the question directly "
          "and concisely.")
POPULATIONS = ("headline_teach", "headline_hidden", "side_channel")


def hlog(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_populations(split_path):
    split = json.load(open(split_path, encoding="utf-8"))
    out = []
    for pop in POPULATIONS:
        for it in split[pop]:
            out.append((pop, it))
    return out, split.get("key_hash_prefix"), split.get("csv_sha256")


def main():
    ap = argparse.ArgumentParser(
        description="Closed-book FreshQA baseline (deterministic scoring)")
    ap.add_argument("--split", default=os.path.join(SP, "fq_split.json"),
                    help="fq_split.json produced by fq_split.py "
                         "(default: alongside this script)")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="transformer_lens model name (default: %(default)s)")
    ap.add_argument("--eos-token", default="<|im_end|>",
                    help="chat end-of-turn token for the model family "
                         "(default: %(default)s, Qwen chat template)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--loader", default=os.environ.get("FQ_LOADER",
                                                       "bf16_np"),
                    choices=["bf16_np", "fp32"],
                    help="bf16_np = from_pretrained_no_processing+bfloat16 "
                         "(safe for >=7B); fp32 = naive from_pretrained "
                         "(OOM risk, see module docstring)")
    ap.add_argument("--store", default=os.environ.get("FQ_STORE",
                                                      "fq_runs"),
                    help="output directory (default: ./fq_runs)")
    ap.add_argument("--ckpt", default=None,
                    help="edited checkpoint to load (strict=False, "
                         "matched/missing counts printed)")
    ap.add_argument("--label", default="Z",
                    help="arm label (default Z, closed-book baseline)")
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    os.makedirs(A.store, exist_ok=True)
    import torch
    from transformer_lens import HookedTransformer
    torch.set_grad_enabled(False)
    # Loader is an INSTRUMENT choice, recorded in the report; keep it
    # identical across every arm you intend to compare.
    if A.loader == "bf16_np":
        model = HookedTransformer.from_pretrained_no_processing(
            A.model, device=A.device, dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(
            A.model, device=A.device, dtype=torch.float32)
    model.eval()
    ckpt_sha = None
    if A.ckpt:
        raw = open(A.ckpt, "rb").read()
        ckpt_sha = hashlib.sha256(raw).hexdigest()
        del raw
        sd = torch.load(A.ckpt, map_location=A.device)
        res = model.load_state_dict(sd, strict=False)
        hlog(f"ckpt {A.ckpt} sha {ckpt_sha[:12]} keys {len(sd)} "
             f"missing {len(res.missing_keys)} "
             f"unexpected {len(res.unexpected_keys)}")
        if any(k in res.unexpected_keys for k in sd):
            raise SystemExit("REFUSED: checkpoint keys not consumed")
        del sd
        model.eval()
    tok = model.tokenizer
    eos_id = tok.convert_tokens_to_ids(A.eos_token)
    items, key_prefix, csv_sha = load_populations(A.split)
    gen_cfg = dict(do_sample=False, repetition_penalty=None,
                   max_new_tokens=MAX_NEW, stop=A.eos_token,
                   stack="transformer_lens", loader=A.loader, ckpt=A.ckpt,
                   ckpt_sha=ckpt_sha, system=SYSTEM, mode="zeroshot")
    out_p = A.out or os.path.join(A.store, "z_baseline_report.json")
    rows = []
    if os.path.exists(out_p + ".partial"):
        rows = json.load(open(out_p + ".partial", encoding="utf-8"))
    done = {(r["population"], r["id"]) for r in rows}
    log_dir = os.path.join(A.store, f"{A.label}_baseline_logs")
    os.makedirs(log_dir, exist_ok=True)

    for pop, it in items:
        if (pop, it["id"]) in done:
            continue
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": it["question"]}]
        prompt = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        ids = model.to_tokens(prompt)
        out_t = model.generate(
            ids, max_new_tokens=MAX_NEW, do_sample=False,
            stop_at_eos=True, eos_token_id=eos_id, verbose=False)
        text = tok.decode(out_t[0, ids.shape[1]:].tolist(),
                          skip_special_tokens=True)
        sc = fq_score.score_item(it, text)
        row = dict(arm=A.label, id=it["id"], population=pop,
                   effective_year=it["effective_year"],
                   gen=text[:200],
                   contains=sc["contains"], strict=sc["strict"],
                   hit_indices=sc["hit_indices"],
                   strict_hit_indices=sc["strict_hit_indices"],
                   taxonomy=sc["taxonomy"],
                   n_tokens_prompt=int(ids.shape[1]),
                   n_tokens_out=int(out_t.shape[1] - ids.shape[1]),
                   prompt_sha=hashlib.sha256(
                       prompt.encode("utf-8")).hexdigest()[:16])
        with open(os.path.join(log_dir, f"{it['id']}_gen.txt"), "w",
                  encoding="utf-8") as f:
            f.write(prompt + "\n=====GENERATION=====\n" + text)
        rows.append(row)
        json.dump(rows, open(out_p + ".partial", "w"))

    def agg(sub):
        n = len(sub)
        return dict(
            n=n,
            contains_mean=round(sum(r["contains"] for r in sub) / n, 4)
            if n else None,
            strict_mean=round(sum(r["strict"] for r in sub) / n, 4)
            if n else None,
            taxonomy={t: sum(1 for r in sub if r["taxonomy"] == t)
                      for t in ("HIT", "EMPTY", "REFUSAL", "WRONG")})

    summary = {pop: agg([r for r in rows if r["population"] == pop])
               for pop in POPULATIONS}
    years = sorted({r["effective_year"] for r in rows})
    by_year = {y: agg([r for r in rows if r["effective_year"] == y])
               for y in years}
    json.dump(dict(grade="FQ-BASELINE", gen_cfg=gen_cfg,
                   model=A.model, split_key_prefix=key_prefix,
                   csv_sha256=csv_sha, rows=rows, summary=summary,
                   by_effective_year=by_year,
                   finished=time.strftime("%Y-%m-%d %H:%M:%S")),
              open(out_p, "w"), indent=1)
    for pop in POPULATIONS:
        hlog(f"{pop}: {json.dumps(summary[pop])}")
    hlog(f"by_effective_year: {json.dumps(by_year)}")
    print("FQ-BASELINE-DONE", flush=True)


if __name__ == "__main__":
    main()
