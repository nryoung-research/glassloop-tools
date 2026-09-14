#!/usr/bin/env python3
"""FP-658 coding-key bases (GPU; base model only; no writes to any model). Nathan 2026-09-09: "go yeah? Seems worth a check."
Builds FP-647-format key bases {str(L): {"resid": V, "mlp": V}} from the Gram of the BASE model's activations on the FP-656 coding pool
prompts (pool key from the manifest, chat-rendered with add_generation_prompt, all prompt positions), layers CB_LAYERS (28-35): "resid" =
MLP input (pre-hook on mlp), "mlp" = down_proj input; rule sqrt(eig) > CB_RULE (1e-2) x sqrt(max eig), exactly as fp653_prep.py / FP-647.
Also writes a RANDOM control basis of the SAME ranks per layer/kind (seeded Haar-orthonormal columns via QR of a Gaussian), so that removing
the same energy fraction in random directions can be told apart from removing coding directions (G-766 norm-confound lesson).
Outputs (no-clobber): CB_OUT/fp658_coding_bases.pt, CB_OUT/fp658_random_bases.pt, CB_OUT/fp658_bases_receipt.json.
Env: CB_MODEL (HF id or dir), CB_REVISION, CB_MANIFEST (fp656_pool_manifest.json), CB_POOL_KEY (default K300), CB_PAIR_FILES (comma list),
CB_OUT, CB_LAYERS (28-35), CB_RULE (1e-2), CB_SEED (20260909), CB_DEVICE (cuda)."""
import hashlib
import json
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ["CB_MODEL"]
REV = os.environ.get("CB_REVISION")
MAN = os.environ["CB_MANIFEST"]
KEY = os.environ.get("CB_POOL_KEY", "K300")
PAIRS = [p.strip() for p in os.environ["CB_PAIR_FILES"].split(",") if p.strip()]
OUT = os.environ["CB_OUT"]
LO, HI = [int(x) for x in os.environ.get("CB_LAYERS", "28-35").split("-")]
LAYERS = list(range(LO, HI + 1))
RULE = float(os.environ.get("CB_RULE", "1e-2"))
SEED = int(os.environ.get("CB_SEED", "20260909"))
DEV = os.environ.get("CB_DEVICE", "cuda")


def hlog(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    outs = {"coding": os.path.join(OUT, "fp658_coding_bases.pt"), "random": os.path.join(OUT, "fp658_random_bases.pt"), "receipt": os.path.join(OUT, "fp658_bases_receipt.json")}
    for p in outs.values():
        if os.path.exists(p):
            raise SystemExit("FP658-NO-CLOBBER %s" % p)
    man = json.load(open(MAN, encoding="utf-8"))
    ids = man["pools"][KEY]
    by_id = {}
    for pf in PAIRS:
        for it in json.load(open(pf, encoding="utf-8"))["items"]:
            by_id[it["id"]] = it
    prompts = [by_id[i]["prompt"] for i in ids]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REV)
    base = AutoModelForCausalLM.from_pretrained(MODEL, revision=REV, dtype=torch.bfloat16).to(DEV).eval()
    d_model, d_mlp = base.config.hidden_size, base.config.intermediate_size
    gram = {L: {"resid": torch.zeros(d_model, d_model, device=DEV), "mlp": torch.zeros(d_mlp, d_mlp, device=DEV)} for L in LAYERS}
    hooks = []
    for L in LAYERS:
        mlp = base.model.layers[L].mlp

        def pre_mlp(mod, args, L=L):
            x = args[0].reshape(-1, d_model).float()
            gram[L]["resid"].addmm_(x.t(), x)

        def pre_down(mod, args, L=L):
            h = args[0].reshape(-1, d_mlp).float()
            gram[L]["mlp"].addmm_(h.t(), h)
        hooks.append(mlp.register_forward_pre_hook(pre_mlp))
        hooks.append(mlp.down_proj.register_forward_pre_hook(pre_down))
    npos = 0
    t0 = time.time()
    with torch.no_grad():
        for i, pr in enumerate(prompts):
            enc = tok.apply_chat_template([{"role": "user", "content": pr}], add_generation_prompt=True, return_tensors="pt")
            ids_ = (enc.input_ids if hasattr(enc, "input_ids") else enc).to(DEV)
            base(input_ids=ids_, attention_mask=torch.ones_like(ids_), use_cache=False)
            npos += int(ids_.shape[1])
            if (i + 1) % 50 == 0:
                hlog("  %d/%d prompts, %d positions" % (i + 1, len(prompts), npos))
    for h in hooks:
        h.remove()
    g = torch.Generator().manual_seed(SEED)
    bases, rand, ranks = {}, {}, {}
    for L in LAYERS:
        bases[str(L)], rand[str(L)], ranks[str(L)] = {}, {}, {}
        for kind in ("resid", "mlp"):
            ev, U = torch.linalg.eigh(gram[L][kind])
            keep = ev.clamp_min(0).sqrt() > RULE * float(ev[-1].clamp_min(0).sqrt())
            V = U[:, keep].contiguous().cpu()
            bases[str(L)][kind] = V
            ranks[str(L)][kind] = int(V.shape[1])
            Q, _ = torch.linalg.qr(torch.randn(V.shape[0], V.shape[1], generator=g, dtype=torch.float32))
            rand[str(L)][kind] = Q.contiguous()
    torch.save(bases, outs["coding"])
    torch.save(rand, outs["random"])
    rec = {"kind": "fp658_bases_receipt", "model": MODEL, "revision": REV, "pool_key": KEY, "n_prompts": len(prompts), "positions": npos, "layers": LAYERS, "rule_rel": RULE, "seed_random": SEED,
           "ranks": ranks, "manifest_sha256": fsha(MAN), "pair_files_sha256": {os.path.basename(p): fsha(p) for p in PAIRS}, "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
           "outputs_sha256": {k: fsha(v) for k, v in outs.items() if k != "receipt"}, "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if DEV == "cuda" else DEV,
           "wall_sec": round(time.time() - t0, 1), "stamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "tool_sha256": fsha(os.path.abspath(__file__))}
    json.dump(rec, open(outs["receipt"], "w"), indent=1, sort_keys=True)
    hlog("FP658-BASES-DONE %s prompts %d positions %d ranks %s coding %s random %s" % (KEY, len(prompts), npos, ranks, rec["outputs_sha256"]["coding"][:16], rec["outputs_sha256"]["random"][:16]))


if __name__ == "__main__":
    main()
