#!/usr/bin/env python3
"""FP-678 amendment v0.1: the PLAIN FINE-TUNE baselines a practitioner would run on the same 60 lessons - the arms our machinery must beat.
  FT_KIND=lora : PEFT LoRA r 16 / alpha 32 / dropout 0.05 on q,k,v,o,gate,up,down of EVERY layer, AdamW lr 2e-4, merged into the weights at export.
  FT_KIND=full : every parameter trainable (bf16 weights, AdamW states in bf16, gradient checkpointing), AdamW lr 1e-5.
Both: 3 epochs over the 60 teach lessons, seeded shuffle per epoch, effective batch 4 (batch 1 x 4 accumulation) -> 45 optimizer steps, constant lr,
no warmup, no replay, no gate, no projection. Lesson = the exam form our writer trains on (pilot_teach.chat_form): chat-rendered closed prompt, no BOS,
loss on the fenced compact worked solution + eos only. Export = native bf16 save_pretrained + tokenizer + MANIFEST-SHA256.json (files_sha256), the
layout fp678_h2h.py (FP678_ARM=ours) and fp656_delta_norm.py read. Receipt: recipe, steps, loss-bearing tokens, per-epoch loss, wall, peak GPU.
Env: FT_KIND, FT_OUT (dir; export to FT_OUT/A_written_bf16), FT_LIMIT_LESSONS (0 = 60), FT_EPOCHS (3), FT_LR (per kind default), FT_SEED (0), FT_ACCUM (4).
"""
import glob
import hashlib
import json
import os
import random
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
CLIMBER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "climber"))
SNAP = os.environ.get("FP678_SNAPSHOT", "C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
KIND = os.environ["FT_KIND"]
OUT = os.path.abspath(os.environ["FT_OUT"])
SAVE = os.path.join(OUT, "A_written_bf16")
LIMIT = int(os.environ.get("FT_LIMIT_LESSONS", "0"))
EPOCHS = int(os.environ.get("FT_EPOCHS", "3"))
LR = float(os.environ.get("FT_LR", {"lora": "2e-4", "full": "1e-5"}[KIND]))
SEED = int(os.environ.get("FT_SEED", "0"))
ACCUM = int(os.environ.get("FT_ACCUM", "4"))
OPT = os.environ.get("FT_OPT", "adamw")          # adamw | adafactor (the full arm falls back to Adafactor only if bf16 AdamW states do not fit the 32 GB card; receipted)
FAMILY = os.path.join(CLIMBER, "vex_family.json")
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}


def hlog(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    if os.path.exists(SAVE):
        raise SystemExit("REFUSED-NO-CLOBBER %s exists" % SAVE)
    os.makedirs(OUT, exist_ok=True)
    sys.path.insert(0, CLIMBER)
    import importlib
    worked_solution = importlib.import_module("vex_teach").worked_solution
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    random.seed(SEED)
    torch.manual_seed(SEED)
    fam = json.load(open(FAMILY, encoding="utf-8"))
    tasks = fam["teach"][:LIMIT] if LIMIT else fam["teach"]
    tok = AutoTokenizer.from_pretrained(SNAP)
    model = AutoModelForCausalLM.from_pretrained(SNAP, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda")
    gen_cfg = model.generation_config
    # lessons in the exam form
    lessons = []
    for t in tasks:
        rendered = tok.apply_chat_template([{"role": "user", "content": t["prompt_closed"]}], add_generation_prompt=True, tokenize=False)
        resp = "```python\n%s\n```%s" % (worked_solution(t["ops"], compact=True), tok.eos_token)
        p_ids = tok(rendered, add_special_tokens=False)["input_ids"]
        r_ids = tok(resp, add_special_tokens=False)["input_ids"]
        ids = torch.tensor([p_ids + r_ids], dtype=torch.long)
        labels = torch.tensor([[-100] * len(p_ids) + r_ids], dtype=torch.long)
        lessons.append({"task_id": t["task_id"], "ids": ids, "labels": labels, "n_loss_tokens": len(r_ids)})
    data_sha = hashlib.sha256(json.dumps([[l["task_id"], l["ids"][0].tolist()] for l in lessons]).encode()).hexdigest()
    if KIND == "lora":
        from peft import LoraConfig, get_peft_model
        import peft
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
        lib = {"peft": peft.__version__}
    elif KIND == "full":
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        for p in model.parameters():
            p.requires_grad_(True)
        lib = {}
    else:
        raise SystemExit("FT_KIND lora|full")
    params = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in params)
    n_all = sum(p.numel() for p in model.parameters())
    if OPT == "adamw":
        opt = torch.optim.AdamW(params, lr=LR, betas=(0.9, 0.999), weight_decay=0.0)
        opt_name = "AdamW(0.9,0.999,wd 0)"
    elif OPT == "adafactor":
        from transformers.optimization import Adafactor
        opt = Adafactor(params, lr=LR, scale_parameter=False, relative_step=False, warmup_init=False, weight_decay=0.0)
        opt_name = "Adafactor(lr fixed, no relative step)"
    else:
        raise SystemExit("FT_OPT adamw|adafactor")
    hlog("FT %s lessons %d epochs %d lr %g accum %d trainable %.1fM / %.1fM" % (KIND, len(lessons), EPOCHS, LR, ACCUM, n_train / 1e6, n_all / 1e6))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    steps, tokens, ep_loss, micro = 0, 0, [], 0
    order_log = []
    for ep in range(EPOCHS):
        order = list(range(len(lessons)))
        random.shuffle(order)
        order_log.append(order)
        tot, n = 0.0, 0
        opt.zero_grad(set_to_none=True)
        for j, i in enumerate(order):
            l = lessons[i]
            out = model(input_ids=l["ids"].cuda(), labels=l["labels"].cuda())
            (out.loss / ACCUM).backward()
            tot += float(out.loss)
            n += 1
            tokens += l["n_loss_tokens"]
            micro += 1
            if micro % ACCUM == 0 or j == len(order) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
        ep_loss.append(round(tot / max(1, n), 4))
        hlog("  epoch %d/%d mean loss %.4f steps %d" % (ep + 1, EPOCHS, ep_loss[-1], steps))
    wall = time.time() - t0
    peak = int(torch.cuda.max_memory_allocated())
    model.eval()
    if KIND == "lora":
        model = model.merge_and_unload()
    model.config.use_cache = True
    model.generation_config = gen_cfg
    model.save_pretrained(SAVE, safe_serialization=True)
    tok.save_pretrained(SAVE)
    files = {os.path.basename(f): fsha(f) for f in sorted(glob.glob(os.path.join(SAVE, "*"))) if os.path.isfile(f)}
    if "model.safetensors" not in files:
        raise SystemExit("EXPORT-NOT-SINGLE-SHARD %s" % list(files))
    manifest = {"dir": SAVE, "files_sha256": files, "dtype": "bfloat16", "reload_verified": False, "mode": "FT_" + KIND, "note": "FP-678 plain fine-tune baseline export (native bf16 save_pretrained%s)" % (", LoRA merged" if KIND == "lora" else "")}
    # reload check: the saved file reloads and one late weight equals the live model's
    from safetensors import safe_open
    with safe_open(os.path.join(SAVE, "model.safetensors"), framework="pt", device="cpu") as f:
        k = "model.layers.%d.mlp.down_proj.weight" % (model.config.num_hidden_layers - 1)   # FP-680: the last layer of whatever depth (was hard-coded 35)
        live = dict(model.named_parameters())[k].detach().to("cpu")
        manifest["reload_verified"] = bool(torch.equal(f.get_tensor(k).to(live.dtype), live))
    json.dump(manifest, open(os.path.join(SAVE, "MANIFEST-SHA256.json"), "w"), indent=1, sort_keys=True)
    receipt = {"kind": "fp678_ft_receipt", "ft_kind": KIND, "recipe": {"epochs": EPOCHS, "lr": LR, "optimizer": opt_name, "effective_batch": ACCUM, "micro_batch": 1, "grad_clip": 1.0,
               "warmup": 0, "schedule": "constant", "seed": SEED, "shuffle": "seeded per epoch", "loss": "response tokens only (fenced compact worked solution + eos)", "prompt_form": "chat-rendered prompt_closed, no BOS",
               "lora": (LORA if KIND == "lora" else None), "full": ({"trainable": "all parameters", "weights": "bf16", "optimizer_states": "bf16", "gradient_checkpointing": True} if KIND == "full" else None)},
               "n_lessons": len(lessons), "lesson_ids": [l["task_id"] for l in lessons], "data_sha256": data_sha, "family_sha256": fsha(FAMILY), "epoch_order": order_log,
               "realized_dose": {"optimizer_steps": steps, "loss_bearing_tokens": tokens, "epoch_mean_loss": ep_loss}, "trainable_params": n_train, "total_params": n_all,
               "wall_s": round(wall, 1), "peak_gpu_bytes": peak, "stack": {"python": sys.version.split()[0], "torch": torch.__version__, "transformers": transformers.__version__, **lib},
               "base_snapshot": SNAP, "export": {"dir": SAVE, "model_safetensors_sha256": files["model.safetensors"], "reload_verified": manifest["reload_verified"]}, "script_sha256": fsha(os.path.abspath(__file__)),
               "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rp = os.path.join(OUT, "fp678_ft_%s_receipt.json" % KIND)
    json.dump(receipt, open(rp, "w"), indent=1, sort_keys=True)
    hlog("FP678-FT-DONE %s steps %d tokens %d loss %s wall %.0fs peak %.1f GiB export %s reload %s receipt %s" % (KIND, steps, tokens, ep_loss, wall, peak / 2 ** 30, files["model.safetensors"][:16], manifest["reload_verified"], fsha(rp)[:16]))


if __name__ == "__main__":
    main()
