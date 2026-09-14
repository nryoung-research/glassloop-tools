#!/usr/bin/env python3
# FP-667 FAMILY-SELECTABLE copy of the FP-662 in-write projection writer (7f764b27): adds FG_FAMILY_JSON / FG_FAMILY_MODULE (the pilot_teach family adapter of C-707) and records them in the receipt; with both unset the lesson construction is byte-identical to FP-662. Original header follows.
"""FP-662 IN-WRITE PROJECTION writer (copy of the FP-652 v3 candidate 32411ae4 + optional FG_PROJECT_BASES / FG_EXPECT_PROJECT_SHA: every realized step is projected out of a pinned input-side span; U mode only is used for FP-662). Original header follows.
FP-652 lesson-gate fix candidate v1 (copied writer; no live campaign modified): the pinned writer's LESSON CONSTRUCTION (pilot_teach.chat_form, rotation windows, per-lesson
AdamW reset, reservation quotas, replay draws, seeded schedule) driven through the decision_guard package's NativeBF16Shadow (HF bf16 native
model + fp32 masters) so that arms U / R / C share ONE tokenizer, dtype and decoder path.
  U  plain step: accumulate the response-only CE of the drawn sequences, optimizer.step(), sync (no guard, no gating).
  R  U + one coding-trace replay pair per step (base completion as the answer, chat form, response-only loss, own budget).
  C  R + controller.constrained_step: <= FG_MAX_ROWS margin rows on the most threatened protected prefixes, lesson-only acquisition descent,
     native finite lesson-only loss decrease (replay stays in the proposal), FULL-BANK verify_root_trace, backtracking over step scales, transaction rollback.
Env (pins are REQUIRED where marked):
  FG_MODE U|R|C; FG_MODEL_ROOT (local snapshot dir) ; FG_CLIMBER (dir with pilot_teach.py, vex_family.json, vexlib, vex_tasks);
  FG_EXPECT_FAMILY_SHA*; FG_TRACES (fp652_selected_traces.jsonl) + FG_EXPECT_TRACES_SHA* (R, C; optional in R when a pool is given);
  FP-656 pool replay (R only): FG_POOL_MANIFEST + FG_EXPECT_MANIFEST_SHA* + FG_POOL_KEY (K40|K120|K300|RND120_control) + FG_REPLAY_PAIRS (comma list of
    contrast-pair JSON files {items:[{id,prompt,answer}]}; each pinned by sha inside the manifest sources); pairs = the pool ids in manifest order,
    chat-form prompt -> base answer + eos, one draw per accepted step, own budget; FG_DG (decision_guard package dir);
  FG_OUT, FG_SAVE_DIR, FG_TAG; FG_SEED (default 20260829); recipe FG_STEPS 60, FG_LR 1e-5, FG_LEG_TOK 33943, FG_DOSE_TOK 5657, FG_MAXTOK 1024,
  FG_WINDOW_TOP (default n_layers-2); guard FG_MAX_ROWS 4, FG_DESCENT 0.05, FG_SCALES 1,.5,.25,.125, FG_MAX_CONSEC_REJECT 3, FG_MAX_NEW_TOKENS 256.
Everything attempted is counted; a rejected step counts as attempted exposure, never as dose."""
import copy
import gc
import hashlib
import json
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODE = os.environ["FG_MODE"].upper()
assert MODE in ("U", "R", "C")
MODEL_ROOT = os.environ["FG_MODEL_ROOT"]
CLIMBER = os.environ["FG_CLIMBER"]
DG = os.environ["FG_DG"]
OUT = os.environ["FG_OUT"]
SAVE_DIR = os.environ["FG_SAVE_DIR"]
TAG = os.environ.get("FG_TAG", "fp652")
SEED = int(os.environ.get("FG_SEED", "20260829"))
STEPS = int(os.environ.get("FG_STEPS", "60"))
LR = float(os.environ.get("FG_LR", "1e-5"))
LEG_TOK = int(os.environ.get("FG_LEG_TOK", "33943"))
DOSE_TOK = int(os.environ.get("FG_DOSE_TOK", "5657"))
MAXTOK = int(os.environ.get("FG_MAXTOK", "1024"))
MAX_ROWS = int(os.environ.get("FG_MAX_ROWS", "4"))
DESCENT = float(os.environ.get("FG_DESCENT", "0.05"))
SCALES = tuple(float(x) for x in os.environ.get("FG_SCALES", "1,.5,.25,.125").split(","))
MAX_CONSEC_REJECT = int(os.environ.get("FG_MAX_CONSEC_REJECT", "3"))
MAX_NEW = int(os.environ.get("FG_MAX_NEW_TOKENS", "256"))
HF_MAP = {"W_in": "up_proj", "W_gate": "gate_proj", "W_out": "down_proj"}


def hlog(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 24), b""):
            h.update(ch)
    return h.hexdigest()

WRITER_SHA_AT_START = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()  # the bytes this process loaded; a file replaced mid-run cannot rename the receipt



def pinned(path, envname):
    b = open(path, "rb").read()
    s = hashlib.sha256(b).hexdigest()
    e = os.environ.get(envname, "")
    if len(e) != 64 or e != s:
        raise SystemExit("FP652-REFUSED: %s digest %s differs from pin %s (%s)" % (path, s[:16], e[:16], envname))
    return b, s


def main():
    t0 = time.time()
    if os.path.exists(SAVE_DIR):
        raise SystemExit("FP652-NO-CLOBBER %s" % SAVE_DIR)
    os.makedirs(OUT, exist_ok=True)
    # ---- the pinned writer's lesson construction, imported with the recipe environment it expects ----
    for k in list(os.environ):
        if k.startswith("VXT_"):
            del os.environ[k]
    FAMILY_JSON = os.environ.get("FG_FAMILY_JSON", "vex_family.json")                      # FP-667: family file name inside CLIMBER (default = vex, byte-identical behaviour)
    FAMILY_MODULE = os.environ.get("FG_FAMILY_MODULE")                                       # FP-667: pilot_teach family adapter module (e.g. quill_tasks); unset = vex bindings
    os.environ.update({"VXT_VIEW": "closed", "VXT_CHAT_FORM": "only", "VXT_MAXTOK": str(MAXTOK), "VXT_FAMILY": os.path.join(CLIMBER, FAMILY_JSON)})
    if FAMILY_MODULE:
        os.environ.update({"VXT_FAMILY_MODULE": FAMILY_MODULE, "VXT_EXPECT_FAMILY_SHA": os.environ.get("FG_EXPECT_FAMILY_SHA", "")})
    sys.path.insert(0, CLIMBER)
    sys.path.insert(0, DG)
    import pilot_teach as PT  # noqa: E402
    from controller import constrained_step, MarginConstraint, FiniteCheck  # noqa: E402
    from decoder import GreedyTrace, verify_root_trace, DecoderGuardError  # noqa: E402
    from search import make_margin_callback  # noqa: E402
    from shadow import NativeBF16Shadow  # noqa: E402
    fam_bytes, FAM_SHA = pinned(os.path.join(CLIMBER, FAMILY_JSON), "FG_EXPECT_FAMILY_SHA")
    fam = json.loads(fam_bytes.decode("utf-8"))
    teach_tasks = fam["teach"]
    bad = [t["task_id"] for t in teach_tasks if not PT.VT.grade(PT.worked_solution(t["ops"]), t)[0]]
    if bad:
        raise SystemExit("FP652-REFUSED: worked solutions fail their own tests: %s" % bad[:5])
    tok = AutoTokenizer.from_pretrained(MODEL_ROOT, local_files_only=True)
    lessons = [PT.chat_form(t, tok) for t in teach_tasks]                # (rendered prompt, fenced compact solution + eos)
    N_TOTAL = len(lessons)                                                 # reservation quotas always use the FULL roster (60), so a limited smoke = the first N lessons of a real leg
    LIMIT = int(os.environ.get("FG_LIMIT_LESSONS", "0"))                  # smoke/profile only: first N lessons (receipted)
    if LIMIT:
        lessons, teach_tasks = lessons[:LIMIT], teach_tasks[:LIMIT]
    hlog("writer sha (bytes loaded) %s" % WRITER_SHA_AT_START[:16])
    hlog("lessons: %d chat-form pairs (view closed, chat-only), family %s" % (len(lessons), FAM_SHA[:16]))
    # ---- protected traces and coding replay pairs ----
    traces, pairs, TRACES_SHA = [], [], None
    POOL = None
    if MODE == "R" and os.environ.get("FG_POOL_MANIFEST"):
        mb, MAN_SHA = pinned(os.environ["FG_POOL_MANIFEST"], "FG_EXPECT_MANIFEST_SHA")
        man = json.loads(mb.decode("utf-8"))
        key = os.environ["FG_POOL_KEY"]
        ids = man["pools"][key]
        by_id = {}
        pair_files = {}
        for pf in os.environ["FG_REPLAY_PAIRS"].split(","):
            pf = pf.strip()
            pb = open(pf, "rb").read()
            psha = hashlib.sha256(pb).hexdigest()
            src = man["sources"].get(os.path.basename(pf))
            if src != psha:
                raise SystemExit("FP652-REFUSED: pair file %s sha %s != manifest %s" % (pf, psha[:16], (src or "?")[:16]))
            pair_files[os.path.basename(pf)] = psha
            for it in json.load(open(pf, encoding="utf-8"))["items"]:
                by_id[it["id"]] = it
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit("FP652-REFUSED: pool %s ids missing from pair files: %s" % (key, missing[:5]))
        for i in ids:
            it = by_id[i]
            pairs.append((tok.apply_chat_template([{"role": "user", "content": it["prompt"]}], add_generation_prompt=True, tokenize=False), it["answer"] + tok.eos_token))
        POOL = {"manifest_sha256": MAN_SHA, "key": key, "n": len(ids), "pair_files_sha256": pair_files, "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        hlog("pool replay: %s (%d pairs) from manifest %s" % (key, len(pairs), MAN_SHA[:16]))
    if MODE in ("R", "C") and (MODE == "C" or os.environ.get("FG_TRACES")):
        tb, TRACES_SHA = pinned(os.environ["FG_TRACES"], "FG_EXPECT_TRACES_SHA")
        rows = [json.loads(l) for l in tb.decode("utf-8").splitlines() if l.strip()]
        for r in rows:
            t = r["trace"]
            traces.append({"task_id": r["task_id"], "root": GreedyTrace(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in t.items()}), "completion_text": r["completion_text"]})
        bank_items = {it["task_id"]: it for it in json.load(open(os.path.join(DG, "coding_bank.json"), encoding="utf-8"))["items"]}
        if POOL is None:
            for tr in traces:
                pairs.append((tok.apply_chat_template([{"role": "user", "content": bank_items[tr["task_id"]]["prompt"]}], add_generation_prompt=True, tokenize=False), tr["completion_text"] + tok.eos_token))
        hlog("protected traces: %d (%s); coding replay pairs: %d" % (len(traces), ", ".join(x["task_id"] for x in traces), len(pairs)))
    # ---- native bf16 model + fp32 masters for the 4 rotation windows ----
    native = AutoModelForCausalLM.from_pretrained(MODEL_ROOT, local_files_only=True, dtype=torch.bfloat16).to("cuda").eval()
    ORIG_GEN = copy.deepcopy(native.generation_config)                    # the base's own config, restored for the export
    gen_config = copy.deepcopy(native.generation_config)                   # the capture's greedy config (do_sample False, pad = eos) for verification
    gen_config.do_sample = False
    gen_config.pad_token_id = tok.eos_token_id
    native.generation_config = gen_config
    nl = native.config.num_hidden_layers
    top = int(os.environ.get("FG_WINDOW_TOP", str(nl - 2)))
    ROT = [(top - 6, top - 5), (top - 4, top - 3), (top - 2, top - 1), (top, top + 1)]
    layers = sorted({L for lo, hi in ROT for L in range(lo, hi + 1)})
    params = dict(native.named_parameters())
    masters = {}
    for L in layers:
        for tl, hf in HF_MAP.items():
            masters["blocks.%d.mlp.%s" % (L, tl)] = torch.nn.Parameter(params["model.layers.%d.mlp.%s.weight" % (L, hf)].detach().T.float().contiguous().clone())
    # ---- FP-662: in-write INPUT-SIDE subspace projection (optional). After every optimizer step the realized delta of each active master is
    # projected out of the pinned span: delta <- delta - V(V^T delta), V = bases[layer][kind] (resid for W_in/W_gate, mlp for W_out; TL rows = input).
    # Exact regardless of the optimizer (AdamW's elementwise scaling is applied first, then the realized step is projected). ----
    PROJ_BASES, PROJ_SHA = None, None
    PROJ_LOG = {"steps": 0, "energy_before": 0.0, "energy_removed": 0.0}
    if os.environ.get("FG_PROJECT_BASES"):
        _pb, PROJ_SHA = pinned(os.environ["FG_PROJECT_BASES"], "FG_EXPECT_PROJECT_SHA")
        _b = torch.load(os.environ["FG_PROJECT_BASES"], map_location="cpu")
        PROJ_BASES = {}
        for n in masters:
            _L, _kind = n.split(".")[1], ("resid" if n.split(".")[3] in ("W_in", "W_gate") else "mlp")
            _V = _b[_L][_kind].float().to("cuda")
            if _V.shape[0] != masters[n].shape[0]:
                raise SystemExit("FP662-REFUSED: basis dim %d != master rows %d on %s" % (_V.shape[0], masters[n].shape[0], n))
            PROJ_BASES[n] = _V
        del _b
        hlog("FP-662 projection armed: %d masters, bases %s, ranks %s" % (len(PROJ_BASES), PROJ_SHA[:16], {n: int(v.shape[1]) for n, v in PROJ_BASES.items()}))
    shadow = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
    DEV = next(native.parameters()).device                                 # exact device object (cuda:0), as the search bridge compares devices strictly
    hlog("shadow ready: %d masters over layers %s; native bf16 on cuda; free %.1f GiB" % (len(masters), layers, torch.cuda.mem_get_info()[0] / 2**30))
    MODEL_FILES = {fn: fsha(os.path.join(MODEL_ROOT, fn)) for fn in sorted(os.listdir(MODEL_ROOT)) if fn.endswith((".safetensors", ".json"))}
    IDENTITY = "fp652:%s:%s" % (TAG, hashlib.sha256(json.dumps(MODEL_FILES, sort_keys=True).encode()).hexdigest()[:16])

    VERIFY_ORDER = sorted(traces, key=lambda t: min(t["root"].margins))
    if traces:
        hlog("verification order (min root margin ascending): %s" % ", ".join("%s(%.2f)" % (t["task_id"], min(t["root"].margins)) for t in VERIFY_ORDER))

    def full_check(label):
        """Every protected trace regenerated with the actual decoder; returns (ok, detail, generated_tokens)."""
        toks = 0
        for tr in VERIFY_ORDER:                                            # smallest root margin first: a rejection is detected before the robust traces are regenerated
            ids = torch.tensor([list(tr["root"].prompt_ids)], device="cuda")
            try:
                verify_root_trace(shadow, tr["root"], ids, torch.ones_like(ids), external_model_identity=IDENTITY)
            except DecoderGuardError as exc:
                return False, "%s: %s" % (tr["task_id"], str(exc)[:200]), toks
            toks += len(tr["root"].completion_ids)
        return True, "%s: %d traces preserved" % (label, len(traces)), toks
    if MODE == "C":
        ok, detail, _ = full_check("root")
        if not ok:
            raise SystemExit("FP652-REFUSED: root traces do not verify on the unmodified model under this decoder: %s" % detail)
        hlog("root check: " + detail)

    # ---- sequences and losses (exactly the pinned writer's accounting: response-only mean CE per sequence, divided by the number of sequences) ----
    def enc_pair(pair):
        p_ids = tok(pair[0], add_special_tokens=False)["input_ids"]
        full = tok(pair[0] + pair[1], add_special_tokens=False)["input_ids"]
        if full[:len(p_ids)] != p_ids:
            raise SystemExit("FP652-REFUSED: rendered prompt is not a token prefix of prompt+response")
        if len(full) > MAXTOK:
            raise SystemExit("FP652-REFUSED: chat-form sequence of %d tokens exceeds FG_MAXTOK %d" % (len(full), MAXTOK))
        return torch.tensor([full], device="cuda"), len(p_ids), len(full) - len(p_ids)

    def seq_loss(logits, ids, n_p):
        lp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
        tgt = ids[0, 1:]
        nll = -lp.gather(1, tgt[:, None]).squeeze(1)
        return nll[n_p - 1:].mean()
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    LEG = {"used": 0, "attempted_targets": 0, "accepted_steps": 0, "attempted_steps": 0, "rejected_steps": 0, "replay_tokens": 0, "coding_tokens": 0, "coding_draws": 0,
           "verify_tokens": 0, "verify_calls": 0, "quotas": [], "zero_lessons": [], "step_log": []}
    bills = []
    for i, lesson in enumerate(lessons):
        lo, hi = ROT[i % len(ROT)]
        active = tuple(n for n in masters if lo <= int(n.split(".")[1]) <= hi)
        act = {n: masters[n] for n in active}
        for n, p in masters.items():
            p.requires_grad_(n in active)
        opt = torch.optim.AdamW([masters[n] for n in active], lr=LR, weight_decay=0.0)
        remaining = max(1, N_TOTAL - i)
        quota = (LEG_TOK - LEG["used"]) // remaining
        LEG["quotas"].append(quota)
        tok_used, steps, losses, consec_rej = 0, 0, [], 0
        priors = lessons[:i]
        for _ in range(STEPS):
            if (DOSE_TOK and tok_used >= DOSE_TOK) or LEG["used"] >= LEG_TOK:
                break
            seqs = [lesson]
            if priors:
                seqs.append(rng.choice(priors))
            encs = [enc_pair(s) for s in seqs]
            need = sum(e[2] for e in encs)
            if tok_used + need > quota or LEG["used"] + need > LEG_TOK:
                break
            coding = None
            if MODE in ("R", "C") and pairs:
                coding = enc_pair(rng.choice(pairs))

            def functional(ids, attention_mask):
                return shadow.functional_forward(active, input_ids=ids, attention_mask=attention_mask, use_cache=False)

            def teaching():
                total = 0.0
                for ids, n_p, _ in encs:
                    total = total + seq_loss(functional(ids, torch.ones_like(ids)).logits, ids, n_p) / len(encs)
                return total

            def optimization_loss():
                # Replay shapes the proposed optimizer direction; acquisition
                # and its native finite gate remain the lesson objective alone.
                total = teaching()
                if coding is not None:
                    ids, n_p, _ = coding
                    total = total + seq_loss(functional(ids, torch.ones_like(ids)).logits, ids, n_p)
                return total

            def finite_loss():
                with torch.no_grad():
                    total = 0.0
                    for ids, n_p, _ in encs:
                        total += float(seq_loss(shadow.forward(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits, ids, n_p)) / len(encs)
                    return total
            LEG["attempted_steps"] += 1
            LEG["attempted_targets"] += need
            if MODE in ("U", "R"):
                opt.zero_grad(set_to_none=True)
                loss = optimization_loss()
                loss.backward()
                prev = ({n: masters[n].detach().clone() for n in active} if PROJ_BASES else None)
                opt.step()
                if PROJ_BASES:
                    with torch.no_grad():
                        for n in active:
                            d = masters[n] - prev[n]
                            V = PROJ_BASES[n]
                            comp = V @ (V.t() @ d)
                            PROJ_LOG["energy_before"] += float((d ** 2).sum())
                            PROJ_LOG["energy_removed"] += float((comp ** 2).sum())
                            masters[n].sub_(comp)
                        PROJ_LOG["steps"] += 1
                    del prev
                shadow.sync_from_masters()
                shadow.mark_accepted()
                accepted, detail = True, "plain step"
                losses.append(float(loss))
            else:
                # threatened rows: teacher-forced raw margins over every protected prefix, smallest slack first, at most 2 per trace
                cands = []
                with torch.no_grad():
                    for tr in traces:
                        root = tr["root"]
                        ids = torch.tensor([list(root.prompt_ids) + list(root.completion_ids)], device="cuda")
                        lg = shadow.forward(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits[0].float()
                        P0 = len(root.prompt_ids)
                        for pos, tokid in enumerate(root.completion_ids):
                            row = lg[P0 - 1 + pos]
                            win = row[tokid].item()
                            row2 = row.clone()
                            row2[tokid] = -float("inf")
                            cands.append((win - row2.max().item() - root.floors[pos], tr, pos))
                cands.sort(key=lambda x: x[0])
                rows, per = [], {}
                for slack, tr, pos in cands:
                    if per.get(tr["task_id"], 0) >= 2:
                        continue
                    rows.append(MarginConstraint("%s:%d" % (tr["task_id"], pos), make_margin_callback(shadow, tr["root"], pos, functional, device=DEV), tr["root"].floors[pos]))
                    per[tr["task_id"]] = per.get(tr["task_id"], 0) + 1
                    if len(rows) >= MAX_ROWS:
                        break

                def check():
                    ok, detail_, toks = full_check("step")
                    LEG["verify_tokens"] += toks
                    LEG["verify_calls"] += 1
                    return FiniteCheck(bool(ok), bool(shadow.changed_from_accepted()), detail_)
                res = shadow.run_step(constrained_step, act, opt, teaching, finite_teaching_loss=finite_loss, optimization_loss=optimization_loss, constraints=rows, finite_check=check,
                                      descent_fraction=DESCENT, step_scales=SCALES, capture_cuda_devices=(0,))
                accepted, detail = bool(res.accepted), res.reason
                LEG["step_log"].append({"lesson": i, "accepted": accepted, "reason": res.reason, "attempts": res.attempts, "scale": res.step_scale, "before": res.teaching_before, "after": res.teaching_after,
                                        "proposal_norm": res.proposal_norm, "corrected_norm": res.corrected_norm, "rows": [c.name for c in rows], "slacks": [round(c[0], 4) for c in cands[:MAX_ROWS]]})
                if accepted:
                    losses.append(float(res.teaching_after))
            if accepted:
                tok_used += need
                LEG["used"] += need
                LEG["accepted_steps"] += 1
                LEG["replay_tokens"] += (encs[1][2] if len(encs) > 1 else 0)
                if coding is not None:
                    LEG["coding_tokens"] += coding[2]
                    LEG["coding_draws"] += 1
                steps += 1
                consec_rej = 0
            else:
                LEG["rejected_steps"] += 1
                consec_rej += 1
                if consec_rej >= MAX_CONSEC_REJECT:
                    break
        for n, p in masters.items():
            p.requires_grad_(False)
        del opt
        gc.collect()
        torch.cuda.empty_cache()
        if steps == 0:
            LEG["zero_lessons"].append(teach_tasks[i]["task_id"])
        bills.append({"task_id": teach_tasks[i]["task_id"], "window": [lo, hi], "loss_first": (round(losses[0], 4) if losses else None), "loss_last": (round(losses[-1], 4) if losses else None),
                      "tokens_consumed": tok_used, "steps_taken": steps, "rejected_steps_in_lesson": sum(1 for s in LEG["step_log"] if s["lesson"] == i and not s["accepted"]) if MODE == "C" else 0})
        if (i + 1) % 10 == 0:
            hlog("  lesson %d/%d: leg targets %d, accepted %d / attempted %d, verify tokens %d" % (i + 1, len(lessons), LEG["used"], LEG["accepted_steps"], LEG["attempted_steps"], LEG["verify_tokens"]))
    # ---- final full-bank check on the settled state, then export ----
    final_ok, final_detail = (None, "not applicable (mode %s)" % MODE)
    if MODE in ("R", "C") and traces:
        final_ok, final_detail, _ = full_check("final")
        hlog("final protected-trace check: %s %s" % (final_ok, final_detail))
    shadow.sync_from_masters()
    os.makedirs(SAVE_DIR, exist_ok=False)
    torch.save({n: p.detach().cpu().float().clone() for n, p in masters.items()}, os.path.join(SAVE_DIR, "fp32_master_windows.pt"))
    native.generation_config = ORIG_GEN                                    # export carries the base's generation_config.json (save_pretrained validates it)
    native.save_pretrained(SAVE_DIR, safe_serialization=True)
    native.generation_config = gen_config
    tok.save_pretrained(SAVE_DIR)
    re = AutoModelForCausalLM.from_pretrained(SAVE_DIR, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    sd = re.state_dict()
    for n, p in masters.items():
        L, tl = int(n.split(".")[1]), n.split(".")[3]
        if not torch.equal(sd["model.layers.%d.mlp.%s.weight" % (L, HF_MAP[tl])].to(torch.bfloat16), p.detach().T.contiguous().to(torch.bfloat16).cpu()):
            raise SystemExit("FP652-RELOAD-REFUSED on %s" % n)
    del re, sd
    files = {fn: fsha(os.path.join(SAVE_DIR, fn)) for fn in sorted(os.listdir(SAVE_DIR)) if os.path.isfile(os.path.join(SAVE_DIR, fn))}
    manifest = {"dir": SAVE_DIR, "files_sha256": files, "dtype": "bfloat16", "reload_verified": True, "mode": MODE, "note": "FP-652 guard writer export: native bf16 saved from the shadow-synced masters; fp32 masters alongside (TL names, [in,out] orientation)"}
    json.dump(manifest, open(os.path.join(SAVE_DIR, "MANIFEST-SHA256.json"), "w"), indent=1, sort_keys=True)
    receipt = {"kind": "fp652_guard_writer_receipt", "writer_variant": ("in-write-projection-v1" if PROJ_BASES else "lesson-gate-fix-v1"), "family_json": FAMILY_JSON, "family_module": FAMILY_MODULE, "projection": ({"bases": os.environ.get("FG_PROJECT_BASES"), "bases_sha256": PROJ_SHA, "side": "input (TL rows)", "rule": "after each optimizer step: delta <- delta - V(V^T delta) on every active master", "steps": PROJ_LOG["steps"], "energy_before_total": PROJ_LOG["energy_before"], "energy_removed_total": PROJ_LOG["energy_removed"], "fraction_removed": PROJ_LOG["energy_removed"] / max(PROJ_LOG["energy_before"], 1e-30)} if PROJ_BASES else None), "objective_contract": {"proposal": "mean current/prior lesson CE + coding replay CE", "acquisition_gradient": "mean current/prior lesson CE only", "finite_acquisition_gate": "native BF16 mean current/prior lesson CE only"}, "mode": MODE, "tag": TAG, "seed": SEED, "limit_lessons": LIMIT or None, "model_root": MODEL_ROOT, "model_files_sha256": MODEL_FILES, "identity": IDENTITY, "family_sha256": FAM_SHA,
               "traces_sha256": TRACES_SHA, "pool_replay": POOL, "protected_traces": [x["task_id"] for x in traces], "generation_config": json.loads(gen_config.to_json_string()) if hasattr(gen_config, "to_json_string") else str(gen_config),
               "recipe": {"steps_cap": STEPS, "lr": LR, "leg_tok": LEG_TOK, "dose_tok": DOSE_TOK, "maxtok": MAXTOK, "rotation": ROT, "reservation": True, "replay": True, "chat_form": "only", "view": "closed"},
               "guard": ({"max_rows": MAX_ROWS, "descent_fraction": DESCENT, "step_scales": list(SCALES), "max_consecutive_rejections": MAX_CONSEC_REJECT, "row_selection": "teacher-forced raw-logit slack (margin - floor), smallest first, <= 2 rows per trace; constraint values themselves use the processed-score callback",
                          "verification": "full protected bank regenerated with the actual decoder after every candidate scale (verify_root_trace); order = min root margin ascending (fail-fast; acceptance still requires all traces)", "verify_order": [t["task_id"] for t in VERIFY_ORDER]} if MODE == "C" else None),
               "bills": bills, "realized_dose": {"leg_tok_used": LEG["used"], "attempted_targets": LEG["attempted_targets"], "accepted_steps": LEG["accepted_steps"], "attempted_steps": LEG["attempted_steps"], "rejected_steps": LEG["rejected_steps"],
                                                 "replay_loss_bearing_tokens": LEG["replay_tokens"], "coding_replay": {"draws": LEG["coding_draws"], "loss_bearing_tokens": LEG["coding_tokens"], "rule": "one coding-trace pair per step, own budget, never counted in leg_tok_used"},
                                                 "verify_calls": LEG["verify_calls"], "verify_generated_tokens": LEG["verify_tokens"], "reserved_quotas": LEG["quotas"], "zero_token_lessons": LEG["zero_lessons"]},
               "step_log": LEG["step_log"], "final_protected_check": {"ok": final_ok, "detail": final_detail}, "materialized_bf16": manifest, "reload_verified": True,
               "runtime": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "gpu": torch.cuda.get_device_name(0)}, "pilot_teach_sha256": fsha(os.path.join(CLIMBER, "pilot_teach.py")),
               "dg_sha256": {fn: fsha(os.path.join(DG, fn)) for fn in ("controller.py", "shadow.py", "decoder.py", "search.py", "transaction.py", "gram_projector.py")}, "writer_sha256": WRITER_SHA_AT_START,
               "wall_sec": round(time.time() - t0, 1), "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    rp = os.path.join(OUT, "fp652_%s_receipt.json" % TAG)
    if os.path.exists(rp):
        raise SystemExit("FP652-NO-CLOBBER %s" % rp)
    json.dump(receipt, open(rp, "w"), indent=1, sort_keys=True)
    hlog("FP652-WRITER-DONE mode %s: accepted %d / attempted %d steps, leg targets %d, verify tokens %d, final check %s; receipt %s" % (MODE, LEG["accepted_steps"], LEG["attempted_steps"], LEG["used"], LEG["verify_tokens"], final_ok, fsha(rp)[:16]))


if __name__ == "__main__":
    main()
