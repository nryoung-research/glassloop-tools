# -*- coding: utf-8 -*-
"""panel-bill: measure a model over a probe panel, then bill the damage
between two measurements.

Ported from the Glass Loop program's sealed panel instrument
(fp467_measure_panel.py) and its billing arithmetic (fp540/fp543/fp547).

`measure` scores every panel item two ways on one model:

  CLOSED (margin)  forced-choice margin in nats: full-candidate log-prob sum
                   for each candidate substituted into the item's first
                   template; margin = lp(answer) - max(lp(distractor)).
                   Byte-comparable across runs of this tool with the same
                   backend/tokenizer.
  OPEN-SET         one extra forward per item at the position before the
                   answer slot: {lp_ans, rank_ans, top1_outside,
                   mass_in_set} over the FULL vocabulary. Standard in the
                   source program because the closed margin reads only the
                   candidates' logits (16 of 151,665 in the source setup)
                   and was measured to under-report damage ~1.79x. Items
                   whose template has no terminal answer slot, or whose
                   candidates share a first token, are SKIPPED WITH A
                   COUNTED REASON, never silently dropped.

`bill` compares two measurement files (a reference/base and a current arm):

  base-positives   items the reference got right (margin > 0)
  casualties       base-positive items whose current margin < 0, listed by
                   name with per-item deficit = max(0, -margin) in nats
  deficit sum      total nats below zero across casualties
  open-set losses  items that were rank 1 in the reference and rank > 1 now
  field stats      mean and p95 of |delta margin| over shared items

No telemetry. Deterministic given a deterministic backend. Selftest runs on
CPU with no downloads and no heavy imports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

# Verbatim from the source instrument: the answer slot must be terminal up to
# trailing whitespace/punctuation for the open-set probe to be well-posed.
TAIL = re.compile(r"[\s\.\,\!\?\"\'’”]*\Z")


def hlog(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# --------------------------------------------------------------- protocol

class ModelAdapter(Protocol):
    """Same adapter shape as the sibling meter-check tool (duplicated so each
    tool stays standalone)."""

    def to_tokens(self, text: str) -> List[int]: ...
    def first_token_id(self, text: str) -> int: ...
    def sequence_logprobs(self, ids: Sequence[int]) -> List[float]: ...
    def next_logprobs(self, ids: Sequence[int]) -> Sequence[float]: ...


def log_softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - lse for x in logits]


# ------------------------------------------------------------ measurement

def margin(model: ModelAdapter, it: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """Forced-choice margin, faithful to the source: templates[0], candidate
    substituted for {answer}, log-prob summed over the FULL token sequence
    (prompt tokens included — they cancel in the margin up to tokenization
    shifts), margin = answer minus best non-answer."""
    tpl = it["templates"][0]
    lps: Dict[str, float] = {}
    for c in it["candidates"]:
        tt = model.to_tokens(tpl.replace("{answer}", c))
        lps[c] = float(sum(model.sequence_logprobs(tt)))
    a = lps[it["answer"]]
    b = max(v for c, v in lps.items() if c != it["answer"])
    return a - b, lps


def open_set(model: ModelAdapter, it: Dict[str, Any]):
    """Open-vocabulary read at the answer slot. Returns (record, None) or
    (None, reason) where reason is 'non_terminal' or 'collision'."""
    tpl = it["templates"][0]
    i = tpl.find("{answer}")
    if i < 0 or not TAIL.fullmatch(tpl[i + len("{answer}"):]):
        return None, "non_terminal"
    ids = [model.first_token_id(c) for c in it["candidates"]]
    if len(set(ids)) != len(ids):
        return None, "collision"
    lp = model.next_logprobs(model.to_tokens(tpl[:i].rstrip()))
    a = model.first_token_id(it["answer"])
    la = float(lp[a])
    if isinstance(lp, list):
        rank = 1 + sum(1 for v in lp if v > la)
        top1 = max(range(len(lp)), key=lp.__getitem__)
    else:  # torch tensor / numpy array fast path
        rank = 1 + int((lp > la).sum())
        top1 = int(lp.argmax())
    mass = sum(math.exp(float(lp[j])) for j in ids)
    return dict(lp_ans=round(la, 6),
                rank_ans=rank,
                top1_outside=bool(top1 not in ids),
                mass_in_set=round(mass, 6)), None


def measure_items(model: ModelAdapter, items: Sequence[Dict[str, Any]],
                  arm: str, checkpoint: Optional[str] = None,
                  base: Optional[str] = None) -> Dict[str, Any]:
    margins: Dict[str, float] = {}
    all_lps: Dict[str, Dict[str, float]] = {}
    os_rows: Dict[str, Dict[str, Any]] = {}
    os_skip = {"non_terminal": 0, "collision": 0}
    t0 = time.time()
    for n, it in enumerate(items):
        m, lps = margin(model, it)
        margins[it["id"]] = m
        # per-candidate log-probs make flip DESTINATIONS readable (which
        # wrong answer won), the source program's intrusion diagnostic
        all_lps[it["id"]] = {k: round(v, 6) for k, v in lps.items()}
        rec, why = open_set(model, it)
        if rec is not None:
            os_rows[it["id"]] = rec
        else:
            os_skip[why] += 1
        if (n + 1) % 200 == 0:
            r = (time.time() - t0) / (n + 1)
            hlog(f"{n + 1}/{len(items)} eta {(len(items) - n - 1) * r / 60:.0f} min")
    hlog(f"open-set read: {len(os_rows)} scored, skipped {os_skip}")
    return dict(arm=arm, checkpoint=checkpoint, base=base, n=len(margins),
                margins=margins, lps=all_lps, open_set=os_rows,
                open_set_skipped=os_skip)


# ----------------------------------------------------------------- billing

def bill(base_doc: Dict[str, Any], cur_doc: Dict[str, Any]) -> Dict[str, Any]:
    """Damage bill of `cur` against `base`. Arithmetic as used throughout the
    source program:

      casualty      base margin > 0 and current margin < 0 (strict; an
                    exactly-zero margin is neither correct nor a casualty
                    and carries zero deficit)
      deficit       max(0, -current margin) in nats (fp544/fp547 convention)
      open-set loss base rank_ans == 1 and current rank_ans > 1; items
                    missing from the current open-set read are not counted
      p95(|dm|)     sorted |delta|[int(n * 0.95)] (fp540 convention)
    """
    bm: Dict[str, float] = base_doc["margins"]
    cm: Dict[str, float] = cur_doc["margins"]
    shared = [i for i in bm if i in cm]
    only_base = len(bm) - len(shared)
    only_cur = len(cm) - len(shared)

    base_pos = [i for i in shared if bm[i] > 0]
    cas = sorted(i for i in base_pos if cm[i] < 0)
    cas_rows = [dict(id=i, base_margin=round(bm[i], 6),
                     current_margin=round(cm[i], 6),
                     deficit_nats=round(max(0.0, -cm[i]), 6)) for i in cas]
    total_deficit = sum(max(0.0, -cm[i]) for i in base_pos)

    ab = sorted(abs(cm[i] - bm[i]) for i in shared)
    field = dict(mean_abs_dm=round(sum(ab) / len(ab), 4) if ab else None,
                 p95_abs_dm=round(ab[int(len(ab) * 0.95)], 4) if ab else None)

    b_os = base_doc.get("open_set", {})
    c_os = cur_doc.get("open_set", {})
    os_losses = []
    for i, rr in b_os.items():
        c = c_os.get(i)
        if c is not None and rr.get("rank_ans") == 1 \
                and c.get("rank_ans", 1) > 1:
            os_losses.append(dict(id=i, new_rank=c["rank_ans"]))

    return dict(
        base_arm=base_doc.get("arm"), current_arm=cur_doc.get("arm"),
        n_shared=len(shared), n_only_base=only_base, n_only_current=only_cur,
        n_base_positives=len(base_pos),
        n_casualties=len(cas), casualties=cas_rows,
        total_deficit_nats=round(total_deficit, 6),
        field=field,
        n_openset_rank1_losses=len(os_losses),
        openset_rank1_losses=os_losses,
        note=("closed-set margins are a LOWER BOUND on damage; the open-set "
              "channel exists because the closed read under-reported ~1.79x "
              "in the source program"),
    )


def print_bill(b: Dict[str, Any]) -> None:
    print(f"bill: {b['current_arm']} vs {b['base_arm']}")
    print(f"  shared items      {b['n_shared']} "
          f"(base-only {b['n_only_base']}, current-only {b['n_only_current']})")
    print(f"  base-positives    {b['n_base_positives']}")
    print(f"  casualties        {b['n_casualties']}")
    for r in b["casualties"]:
        print(f"    {r['id']}: {r['base_margin']:+.3f} -> "
              f"{r['current_margin']:+.3f}  deficit {r['deficit_nats']:.3f} nats")
    print(f"  total deficit     {b['total_deficit_nats']} nats")
    print(f"  |dm| field        mean {b['field']['mean_abs_dm']}  "
          f"p95 {b['field']['p95_abs_dm']}")
    print(f"  open-set rank-1 losses  {b['n_openset_rank1_losses']}")
    for r in b["openset_rank1_losses"]:
        print(f"    {r['id']}: rank 1 -> {r['new_rank']}")


# --------------------------------------------------------------- mock model

class MockModel:
    """Pure-python reference adapter with hand-built logits (see the sibling
    meter-check tool for the same class with generation support)."""

    def __init__(self, vocab: Sequence[str],
                 table: Optional[Dict[tuple, List[float]]] = None,
                 default: Optional[List[float]] = None):
        self.vocab = list(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.table = dict(table or {})
        self.default = list(default) if default is not None \
            else [0.0] * len(self.vocab)

    def to_tokens(self, text: str) -> List[int]:
        return [self.index[w] for w in text.split()]

    def first_token_id(self, text: str) -> int:
        return self.index[(" " + text).split()[0]]

    def _logits(self, ids: Sequence[int]) -> List[float]:
        words = tuple(self.vocab[i] for i in ids)
        if words in self.table:
            return self.table[words]
        if (words[-1],) in self.table:
            return self.table[(words[-1],)]
        return self.default

    def sequence_logprobs(self, ids: Sequence[int]) -> List[float]:
        out = []
        for t in range(len(ids) - 1):
            dist = log_softmax(self._logits(ids[:t + 1]))
            out.append(dist[ids[t + 1]])
        return out

    def next_logprobs(self, ids: Sequence[int]) -> List[float]:
        return log_softmax(self._logits(ids))


# --------------------------------------------------- transformer_lens adapter

class TransformerLensAdapter:
    """Real-model adapter; torch + transformer_lens imported lazily.

    loader='bf16_np' = from_pretrained_no_processing + bfloat16. USE THIS FOR
    LARGE MODELS: the naive fp32 from_pretrained path materializes ~4x the
    weight bytes during processing and has OOM-killed a host in the source
    program (documented incident).

    Checkpoints load with strict=False but fail loud on unexpected keys:
    with the return value discarded, strict=False can silently score the
    pristine base model (source program finding #3, 4okB).
    """

    def __init__(self, model_name: str, checkpoint: Optional[str] = None,
                 device: Optional[str] = None, loader: str = "fp32"):
        import torch
        from transformer_lens import HookedTransformer
        torch.set_grad_enabled(False)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if loader == "bf16_np":
            self.model = HookedTransformer.from_pretrained_no_processing(
                model_name, device=self.device, dtype=torch.bfloat16)
        else:
            self.model = HookedTransformer.from_pretrained(
                model_name, device=self.device, dtype=torch.float32)
        if checkpoint:
            sd = torch.load(checkpoint, map_location=self.device)
            missing = self.model.load_state_dict(sd, strict=False)
            del sd
            if len(missing.unexpected_keys) > 0:
                raise RuntimeError(
                    f"checkpoint has {len(missing.unexpected_keys)} "
                    f"unexpected keys")
            hlog(f"edited checkpoint loaded "
                 f"({len(missing.missing_keys)} keys kept from base)")
        self.model.eval()
        self._tid_cache: Dict[str, int] = {}

    def to_tokens(self, text: str) -> List[int]:
        return self.model.to_tokens(text)[0].tolist()

    def first_token_id(self, text: str) -> int:
        if text not in self._tid_cache:
            self._tid_cache[text] = self.model.tokenizer(
                " " + text, add_special_tokens=False)["input_ids"][0]
        return self._tid_cache[text]

    def sequence_logprobs(self, ids: Sequence[int]) -> List[float]:
        tt = self.torch.tensor([list(ids)], device=self.device)
        lsm = self.model(tt)[0][:-1].log_softmax(-1)
        tgt = tt[0, 1:]
        lp = lsm[self.torch.arange(len(tgt), device=tt.device), tgt]
        return [float(x) for x in lp]

    def next_logprobs(self, ids: Sequence[int]):
        tt = self.torch.tensor([list(ids)], device=self.device)
        return self.model(tt)[0, -1].log_softmax(-1)


# --------------------------------------------------------------------- CLI

def _load_items(panel_path: str, bank_path: Optional[str]) -> List[Dict[str, Any]]:
    panel = json.load(open(panel_path, encoding="utf-8"))
    rows = panel["items"] if isinstance(panel, dict) else panel
    if bank_path:
        bank = {it["id"]: it for it in
                json.load(open(bank_path, encoding="utf-8"))["items"]}
        return [bank[r["id"]] for r in rows]
    return list(rows)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def _build_adapter(A) -> ModelAdapter:
    if bool(A.adapter) == bool(A.model):
        raise SystemExit("give exactly one of --adapter / --model")
    if A.adapter:
        mod_name, _, attr = A.adapter.partition(":")
        if not attr:
            raise SystemExit("--adapter must be 'module:callable'")
        import importlib
        return getattr(importlib.import_module(mod_name), attr)()
    return TransformerLensAdapter(A.model, checkpoint=A.checkpoint,
                                  device=A.device, loader=A.loader)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Panel measurement + damage billing in nats.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="model + panel -> measurement json")
    m.add_argument("--panel", required=True,
                   help="json with items [{id, templates, candidates, "
                        "answer}], or id-rows resolved against --bank")
    m.add_argument("--bank", default=None,
                   help="optional item bank json {items: [...]} keyed by id")
    m.add_argument("--arm", required=True,
                   help="arm name written into the record, e.g. base/edited")
    m.add_argument("--out", required=True)
    m.add_argument("--adapter", default=None,
                   help="module:callable returning a ModelAdapter")
    m.add_argument("--model", default=None,
                   help="transformer_lens model name")
    m.add_argument("--checkpoint", default=None,
                   help="edited .pt overlaid on --model; absent = base")
    m.add_argument("--loader", default="fp32", choices=["fp32", "bf16_np"],
                   help="bf16_np = from_pretrained_no_processing + bfloat16 "
                        "(REQUIRED for large models; naive fp32 loading "
                        "spikes ~4x and can OOM-kill the host)")
    m.add_argument("--device", default=None)

    b = sub.add_parser("bill", help="two measurement jsons -> damage bill")
    b.add_argument("--base", required=True, help="reference measurement json")
    b.add_argument("--current", required=True, help="arm measurement json")
    b.add_argument("--out", default=None)

    A = ap.parse_args(argv)
    if A.cmd == "measure":
        items = _load_items(A.panel, A.bank)
        hlog(f"arm {A.arm}: {len(items)} panel items")
        model = _build_adapter(A)
        out = measure_items(model, items, arm=A.arm, checkpoint=A.checkpoint,
                            base=A.model or A.adapter)
        out["panel_sha256"] = _sha256_file(A.panel)
        txt = json.dumps(out, indent=1)
        open(A.out, "w", encoding="utf-8").write(txt)
        open(A.out + ".sha256", "w").write(
            hashlib.sha256(txt.encode()).hexdigest() + "\n")
        hlog(f"wrote {A.out}")
    else:
        base_doc = json.load(open(A.base, encoding="utf-8"))
        cur_doc = json.load(open(A.current, encoding="utf-8"))
        rep = bill(base_doc, cur_doc)
        print_bill(rep)
        if A.out:
            json.dump(rep, open(A.out, "w", encoding="utf-8"), indent=1)
            print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
