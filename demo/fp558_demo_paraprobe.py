# -*- coding: utf-8 -*-
"""Teaching-generalization probe (prereg 4opD): quiz the 20 authored
paraphrases on BASE and on the demo act-7 learned state, demo match
rule. Publish-whichever-it-reads."""
import json, re, sys, time

PROBE = "fp558_demo_paraprobe.json"
CKPT = r"E:\fp558\demo_v3\ckpts\demo_act7.pt"
OUT = r"E:\fp558\demo_v3\panel641\paraprobe_report.json"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
TOK = 24

def hlog(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def main():
    import torch
    torch.set_grad_enabled(False)
    from transformer_lens import HookedTransformer
    probe = json.load(open(PROBE, encoding="utf-8"))
    model = HookedTransformer.from_pretrained(MODEL, device="cuda",
                                              dtype=torch.float32)
    model.eval(); tok = model.tokenizer

    def ask(q):
        ids = model.to_tokens(q); out = ids
        for _ in range(TOK):
            nx = model(out)[0, -1].argmax().item()
            out = torch.cat([out, torch.tensor([[nx]], device=out.device)], 1)
        return tok.decode(out[0, ids.shape[1]:].tolist())

    def norm(s): return s.casefold().replace("$", "").replace(",", "")

    def run(tag):
        rows = []
        for it in probe["items"]:
            als = [norm(a) for a in it["answers"]]
            # direct question (the trained string) for the same-state contrast
            for kind, q in [("direct", it["question"])] + [
                    ("para%d" % i, p) for i, p in enumerate(it["paraphrases"])]:
                g = ask(q)
                hit = any(a in norm(g) for a in als)
                rows.append({"id": it["id"], "kind": kind, "hit": hit,
                             "gen": g})
                hlog("  [%s] %s/%s %s" % (tag, it["id"], kind,
                                          "HIT" if hit else "miss"))
        para = [r for r in rows if r["kind"] != "direct"]
        direct = [r for r in rows if r["kind"] == "direct"]
        return {"rows": rows,
                "para_rate": round(sum(r["hit"] for r in para) / len(para), 4),
                "direct_rate": round(sum(r["hit"] for r in direct) / len(direct), 4)}

    hlog("BASE pass"); base = run("base")
    sd = torch.load(CKPT, map_location="cuda")
    model.load_state_dict(sd, strict=False); del sd
    torch.cuda.empty_cache()
    hlog("ACT7 pass"); act7 = run("act7")
    rep = {"prereg": "4opD", "probe_sha_note": "fp558_demo_paraprobe.json",
           "match_rule": probe["match_rule"], "authored": probe["authored"],
           "base": {k: base[k] for k in ("para_rate", "direct_rate")},
           "act7": {k: act7[k] for k in ("para_rate", "direct_rate")},
           "rows": {"base": base["rows"], "act7": act7["rows"]},
           "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=1)
    print(json.dumps({k: rep[k] for k in ("base", "act7")}, indent=1))
    print("PARAPROBE-DONE", flush=True)

if __name__ == "__main__":
    main()
