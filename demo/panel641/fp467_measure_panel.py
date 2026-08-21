"""
FP-467 panel measurement (seal fp467_SEAL.md §2) — one arm per invocation.

Scores every panel item's forced-choice margin on one checkpoint (or the pristine base).
The margin function is fp465_measure.py's VERBATIM (templates[0], full-candidate log-prob
sums, no_grad) so panel margins are comparable to the FP-465 chain of trust. Writes
{arm, checkpoint, n, margins: {id: margin}} + sha256. No flip logic here — flips are the
analyzer's job, always against the panel's own base arm.
"""
import json, time, hashlib, argparse, re
import torch


def hlog(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--loader", default="fp32", choices=["fp32", "bf16_np"],
                    help="fp501b: bf16_np = from_pretrained_no_processing + bfloat16")
    ap.add_argument("--checkpoint", default=None, help="edited .pt; absent = score the base")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--arm", required=True, help="arm name written into the record, e.g. base/d400/c400")
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    panel = json.load(open(A.panel, encoding="utf-8"))
    bank = {it["id"]: it for it in json.load(open(A.bank, encoding="utf-8"))["items"]}
    items = [bank[r["id"]] for r in panel["items"]]
    hlog(f"arm {A.arm}: {len(items)} panel items")

    torch.set_grad_enabled(False)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformer_lens import HookedTransformer
    if A.loader == "bf16_np":
        model = HookedTransformer.from_pretrained_no_processing(A.base, device=dev,
                                                                dtype=torch.bfloat16)
    else:
        model = HookedTransformer.from_pretrained(A.base, device=dev, dtype=torch.float32)
    if A.checkpoint:
        sd = torch.load(A.checkpoint, map_location=dev)
        missing = model.load_state_dict(sd, strict=False)
        del sd
        # 4okB finding #3: strict=False with the return discarded can silently score the
        # pristine base. unexpected_keys>0 or every model key missing = wrong file, fail loud.
        if len(missing.unexpected_keys) > 0:
            raise RuntimeError(f"checkpoint has {len(missing.unexpected_keys)} unexpected keys")
        hlog(f"edited checkpoint loaded ({len(missing.missing_keys)} keys kept from base)")
    model.eval()

    def margin(it):
        tpl = it["templates"][0]
        lps = {}
        for c in it["candidates"]:
            tt = model.to_tokens(tpl.replace("{answer}", c))
            lsm = model(tt)[0][:-1].log_softmax(-1)
            tgt = tt[0, 1:]
            lps[c] = float(lsm[torch.arange(len(tgt), device=tt.device), tgt].sum())
        a = lps[it["answer"]]
        b = max(v for c, v in lps.items() if c != it["answer"])
        return a - b, lps

    # ---- OPEN-SET READ (standard since 4okQ/4okR, 2026-08-07). The forced-choice margin sees
    # 16 of 151,665 logits; measured under-report factor 1.79x, and on some frames the closed
    # read finds ZERO damage where the open read finds real loss. This adds ONE forward per
    # item (~6% cost) and does NOT touch the margin computation above, which stays
    # byte-identical so every prior measurement remains comparable.
    TAIL = re.compile(r"[\s\.\,\!\?\"\'’”]*\Z")
    tk = model.tokenizer
    _tid_cache = {}

    def tid(s):
        if s not in _tid_cache:
            _tid_cache[s] = tk(" " + s, add_special_tokens=False)["input_ids"][0]
        return _tid_cache[s]

    def open_set(it):
        """None when the item admits no next-token probe (non-terminal slot or colliding
        candidate first-tokens) -- skipped items are counted, never silently dropped."""
        tpl = it["templates"][0]
        i = tpl.find("{answer}")
        if i < 0 or not TAIL.fullmatch(tpl[i + len("{answer}"):]):
            return None, "non_terminal"
        ids = [tid(c) for c in it["candidates"]]
        if len(set(ids)) != len(ids):
            return None, "collision"
        lp = model(model.to_tokens(tpl[:i].rstrip()))[0, -1].log_softmax(-1)
        a = tid(it["answer"])
        return dict(lp_ans=round(float(lp[a]), 6),
                    rank_ans=int((lp > lp[a]).sum().item()) + 1,
                    top1_outside=bool(int(lp.argmax().item()) not in ids),
                    mass_in_set=round(float(torch.exp(
                        lp[torch.tensor(ids, device=lp.device)]).sum()), 6)), None

    margins, all_lps, os_rows, t0 = {}, {}, {}, time.time()
    os_skip = {"non_terminal": 0, "collision": 0}
    for n, it in enumerate(items):
        m, lps = margin(it)
        margins[it["id"]] = m
        # seal §12: per-candidate log-probs make flip DESTINATIONS readable, which is the
        # intrusion diagnostic (a pumped payload target as destination vs an arbitrary wrong one)
        all_lps[it["id"]] = {k: round(v, 6) for k, v in lps.items()}
        rec, why = open_set(it)
        if rec is not None:
            os_rows[it["id"]] = rec
        else:
            os_skip[why] += 1
        if (n + 1) % 200 == 0:
            r = (time.time() - t0) / (n + 1)
            hlog(f"{n+1}/{len(items)} eta {(len(items)-n-1)*r/60:.0f} min")
    hlog(f"open-set read: {len(os_rows)} scored, skipped {os_skip}")

    out = dict(arm=A.arm, checkpoint=A.checkpoint, base=A.base,
               panel_sha=open(A.panel + ".sha256").read().strip() if
               __import__("os").path.exists(A.panel + ".sha256") else None,
               n=len(margins), margins=margins, lps=all_lps,
               open_set=os_rows, open_set_skipped=os_skip)
    txt = json.dumps(out, indent=1)
    open(A.out, "w", encoding="utf-8").write(txt)
    open(A.out + ".sha256", "w").write(hashlib.sha256(txt.encode()).hexdigest() + "\n")
    hlog(f"wrote {A.out}")


if __name__ == "__main__":
    main()
