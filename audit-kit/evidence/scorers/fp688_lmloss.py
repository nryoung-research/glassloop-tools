#!/usr/bin/env python3
"""FP-688 THE FOURTH SCREEN: held-out language-modelling loss on a fixed sample of pretraining-like text, read in-process, base and candidates
under one stack. Sample = the first N documents of NeelNanda/pile-10k (cached hub dataset), each truncated to the first T tokens of the candidate's
own tokenizer (identical across candidates of one base). Metric = mean per-token NLL in nats over all scored tokens (teacher forcing, no chat template).
Env: LM_MODEL (dir), LM_TAG, LM_OUT (dir), LM_N (200), LM_T (512), LM_DATASET_DIR (hub dataset snapshot dir; auto-found if unset), LM_DEVICE (cuda).
Receipt: model sha (single-file exports) or dir, tokenizer sha, doc ids + text sha, tokens scored, NLL, per-doc NLLs, stack. No-clobber."""
import glob
import hashlib
import json
import os
import sys
import time

import torch

MODEL = os.environ["LM_MODEL"]
TAG = os.environ["LM_TAG"]
OUT = os.path.abspath(os.environ["LM_OUT"])
N = int(os.environ.get("LM_N", "200"))
T = int(os.environ.get("LM_T", "512"))
DEV = os.environ.get("LM_DEVICE", "cuda")


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def find_dataset():
    d = os.environ.get("LM_DATASET_DIR")
    if d:
        return d
    for root in (os.path.expanduser("~/.cache/huggingface/hub"), "E:/hf_cache/hub", "C:/fp43/hf_family"):
        c = glob.glob(os.path.join(root, "datasets--NeelNanda--pile-10k", "snapshots", "*"))
        if c:
            return c[0]
    raise SystemExit("LM-DATASET-MISSING pile-10k")


def load_docs(ds_dir, n):
    files = sorted(glob.glob(os.path.join(ds_dir, "**", "*.parquet"), recursive=True)) or sorted(glob.glob(os.path.join(ds_dir, "**", "*.arrow"), recursive=True))
    if files and files[0].endswith(".parquet"):
        import pyarrow.parquet as pq
        tbl = pq.read_table(files[0])
        col = "text" if "text" in tbl.column_names else tbl.column_names[0]
        texts = tbl.column(col).to_pylist()[:n]
    else:
        from datasets import load_from_disk, load_dataset
        try:
            ds = load_dataset("NeelNanda/pile-10k", split="train", cache_dir=os.path.dirname(os.path.dirname(os.path.dirname(ds_dir))))
        except Exception as e:
            raise SystemExit("LM-DATASET-LOAD-FAILED %s" % e)
        texts = ds["text"][:n]
    return texts, [os.path.basename(f) for f in files][:3]


def main():
    os.makedirs(OUT, exist_ok=True)
    rp = os.path.join(OUT, "fp688_%s_lmloss.json" % TAG)
    if os.path.exists(rp):
        raise SystemExit("REFUSED-NO-CLOBBER %s" % rp)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    ds_dir = find_dataset()
    texts, src = load_docs(ds_dir, N)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(DEV).eval()
    t0 = time.time()
    per_doc, tot_nll, tot_tok = [], 0.0, 0
    with torch.no_grad():
        for i, tx in enumerate(texts):
            ids = tok(tx, add_special_tokens=False, return_tensors="pt").input_ids[:, :T].to(DEV)
            if ids.shape[1] < 2:
                per_doc.append(None); continue
            logits = model(input_ids=ids).logits[0, :-1].float()
            lp = torch.log_softmax(logits, dim=-1).gather(-1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
            nll = float(-lp.sum()); n = int(lp.numel())
            per_doc.append(round(nll / n, 5)); tot_nll += nll; tot_tok += n
    msha = fsha(os.path.join(MODEL, "model.safetensors")) if os.path.exists(os.path.join(MODEL, "model.safetensors")) else None
    rec = {"kind": "fp688_lmloss", "tag": TAG, "model": MODEL, "model_safetensors_sha256": msha, "dataset": "NeelNanda/pile-10k", "dataset_dir": ds_dir, "dataset_files": src,
           "n_docs": len(texts), "max_tokens_per_doc": T, "text_sha256": hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest(), "tokens_scored": tot_tok,
           "mean_nll_nats": tot_nll / max(1, tot_tok), "per_doc_nll": per_doc, "dtype": "bfloat16", "chat_template": False,
           "stack": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, "gpu": torch.cuda.get_device_name(0)},
           "tool_sha256": fsha(os.path.abspath(__file__)), "wall_s": round(time.time() - t0, 1), "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(rec, open(rp, "w"), indent=1, sort_keys=True)
    print("FP688-LMLOSS-DONE %s nll %.5f tokens %d docs %d model %s receipt %s" % (TAG, rec["mean_nll_nats"], tot_tok, len(texts), (msha or "hub")[:16], fsha(rp)[:16]))


if __name__ == "__main__":
    main()
