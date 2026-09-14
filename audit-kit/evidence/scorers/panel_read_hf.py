"""Pilot leg reader (HF side): the 641-item E6 panel read as full-sequence log-probabilities per candidate on a Hugging Face checkpoint at a
declared dtype, producing unrounded margins (gold minus best rival) and per-candidate log-probs, hashed. Used for L0 (base bf16 contract
read) and L4 (materialized bf16 read). Same tokenization and same summation as fp627_meter_hf.read_panel_margins; candidates scored one at a
time (batch 1) so that bf16 kernel rounding does not depend on batch composition.
Usage: python panel_read_hf.py --model ID_OR_PATH --revision REV --dtype bfloat16 --panel P --bank B --out OUT.json [--limit N]"""
import hashlib
import json
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def parse(argv):
    a = {"--model": None, "--revision": None, "--dtype": "bfloat16", "--panel": None, "--bank": None, "--out": None, "--limit": "0", "--device": "cuda"}
    i = 1
    while i < len(argv):
        if argv[i] in a and i + 1 < len(argv):
            a[argv[i]] = argv[i + 1]
            i += 2
        else:
            raise SystemExit("bad argv %r" % argv[i])
    if a.get("--out") and os.path.exists(a["--out"]):
        raise SystemExit("REFUSED-NO-CLOBBER %s" % a["--out"])       # C-714: a read of record is never overwritten
    if any(a[k] is None for k in ("--model", "--panel", "--bank", "--out")):
        raise SystemExit("--model --panel --bank --out required")
    return a


def main():
    a = parse(sys.argv)
    t0 = time.time()
    panel = json.load(open(a["--panel"], encoding="utf-8"))
    bank = {it["id"]: it for it in json.load(open(a["--bank"], encoding="utf-8"))["items"]}
    ids = [r["id"] for r in panel["items"]]
    if int(a["--limit"]) > 0:
        ids = ids[: int(a["--limit"])]
    tok = AutoTokenizer.from_pretrained(a["--model"], revision=a["--revision"])
    model = AutoModelForCausalLM.from_pretrained(a["--model"], revision=a["--revision"], dtype=getattr(torch, a["--dtype"]), low_cpu_mem_usage=True, device_map=a["--device"]).eval()
    dtypes = sorted({str(p.dtype) for p in model.parameters()})
    lps, margins = {}, {}
    with torch.no_grad():
        for i in ids:
            it = bank[i]
            row = {}
            for c in it["candidates"]:
                x = torch.tensor([tok(it["templates"][0].replace("{answer}", c), add_special_tokens=False)["input_ids"]], dtype=torch.long, device=a["--device"])
                lsm = torch.log_softmax(model(input_ids=x).logits[0, :-1].float(), dim=-1)
                row[c] = float(lsm.gather(-1, x[0, 1:].unsqueeze(-1)).sum())
            lps[i] = row
            g = row[it["answer"]]
            margins[i] = g - max(v for c, v in row.items() if c != it["answer"])
    rec = {"kind": "panel_read_hf", "model": a["--model"], "revision": a["--revision"], "dtype_requested": a["--dtype"], "param_dtypes": dtypes, "panel_sha256": fsha(a["--panel"]), "bank_sha256": fsha(a["--bank"]),
           "n": len(ids), "batch": 1, "margins": margins, "logprobs": lps, "torch": torch.__version__, "elapsed_s": round(time.time() - t0, 1), "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    b = json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()
    with open(a["--out"], "xb") as f:                                  # G-752: exclusive create; a concurrent writer fails instead of overwriting
        f.write(b)
    print("PANEL-READ-HF-DONE", hashlib.sha256(b).hexdigest()[:16], "n", len(ids), "dtypes", dtypes, "positive margins", sum(1 for v in margins.values() if v > 0))


if __name__ == "__main__":
    main()
