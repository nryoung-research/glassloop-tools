# -*- coding: utf-8 -*-
"""meter-check: is your eval's scorer lying to you?

Three-channel corrected-meter rescore, ported from the Glass Loop program's
FP-555 full rescore (fp555_fullrescore.py). For every item
{prompt, answer, answer_set} it scores the SAME model three ways:

  FIRST-TOKEN  argmax of the next-token distribution after the prompt equals
               the first token of " " + answer. This is the legacy channel and
               the known-artifact-prone one: it confuses "did not emit the
               answer token first" with "does not know the answer".
  FC           forced choice over the answer_set by FULL-ANSWER log-prob: for
               each candidate, the sum of per-token log-probs of the candidate
               continuation after the prompt; hit iff the true answer scores
               highest.
  OPEN         greedy decode of N tokens (default 12); hit iff the answer
               string appears (case-insensitive) in the generation.

The artifact signature is FIRST-TOKEN = fail while FC or OPEN = pass. In the
source program, applying this corrected meter to a sealed population of
refused verdicts flipped 193 of 227 historical cells.

Model access goes through a tiny adapter protocol (ModelAdapter below) so the
scoring arithmetic is backend-free. A MockModel reference implementation is
included (used by selftest.py); a TransformerLensAdapter is included for real
models and imports torch/transformer_lens lazily.

No telemetry. Deterministic (greedy decode, no sampling). Selftest runs on
CPU with no model downloads and no heavy imports.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any, Dict, List, Optional, Protocol, Sequence


# --------------------------------------------------------------- protocol

class ModelAdapter(Protocol):
    """What a model must provide. All ids are ints into the model's vocab.

    to_tokens(text)          full-context tokenization (include BOS here if
                             your model expects one).
    to_string(ids)           decode a list of token ids to text.
    first_token_id(text)     first token id of " " + text, tokenized WITHOUT
                             special tokens (the source program's convention
                             for answer targets).
    sequence_logprobs(ids)   list of length len(ids)-1; element t is the
                             log-prob of ids[t+1] given ids[:t+1].
    next_logprobs(ids)       full-vocab log-softmax at the last position.
                             May be a python list, numpy array, or torch
                             tensor (anything indexable with an argmax-able
                             fast path is fine).
    """

    def to_tokens(self, text: str) -> List[int]: ...
    def to_string(self, ids: Sequence[int]) -> str: ...
    def first_token_id(self, text: str) -> int: ...
    def sequence_logprobs(self, ids: Sequence[int]) -> List[float]: ...
    def next_logprobs(self, ids: Sequence[int]) -> Sequence[float]: ...


def _argmax(xs: Sequence[float]) -> int:
    """First-max-wins argmax; uses tensor/ndarray .argmax() when available
    (torch/numpy also return the first maximal index)."""
    am = getattr(xs, "argmax", None)
    if am is not None:
        return int(am())
    return max(range(len(xs)), key=xs.__getitem__)


def log_softmax(logits: Sequence[float]) -> List[float]:
    m = max(logits)
    lse = m + math.log(sum(math.exp(x - m) for x in logits))
    return [x - lse for x in logits]


# --------------------------------------------------------------- channels

def score_item(model: ModelAdapter, prompt: str, answer: str,
               answer_set: Sequence[str], n_open: int = 12) -> Dict[str, Any]:
    """Score one item on all three channels. Faithful port of the FP-555
    arithmetic:

      FIRST-TOKEN: hit = argmax(next_logprobs(prompt)) == first token of
                   " " + answer (editor holdout convention).
      FC:          fa_lp(prompt, ans) = sum of target log-probs of
                   to_tokens(prompt + " " + ans) from position npre-1 on,
                   npre = len(to_tokens(prompt)); hit = argmax over the
                   answer_set equals the answer (first-max-wins on ties, in
                   answer_set order).
      OPEN:        greedy n_open tokens appended one at a time; hit =
                   answer.lower() in decoded-generation.lower().
    """
    if answer not in answer_set:
        raise ValueError(f"answer {answer!r} not in answer_set {answer_set!r}")
    ids = list(model.to_tokens(prompt))
    npre = len(ids)

    # FIRST-TOKEN (legacy channel)
    nl = model.next_logprobs(ids)
    first_hit = int(_argmax(nl) == model.first_token_id(answer))

    # FC (full-answer forced choice)
    scores: Dict[str, float] = {}
    for cand in answer_set:
        tt = list(model.to_tokens(prompt + " " + cand))
        lp = model.sequence_logprobs(tt)
        scores[cand] = float(sum(lp[npre - 1:]))
    fc_hit = int(max(scores, key=scores.get) == answer)  # type: ignore[arg-type]

    # OPEN (greedy-N containment)
    out = list(ids)
    for _ in range(n_open):
        out.append(_argmax(model.next_logprobs(out)))
    gen = model.to_string(out[npre:])
    open_hit = int(answer.lower() in gen.lower())

    return dict(first_token=first_hit, fc=fc_hit, open=open_hit,
                fc_scores={k: round(v, 6) for k, v in scores.items()},
                generation=gen)


def run_items(model: ModelAdapter, items: Sequence[Dict[str, Any]],
              n_open: int = 12, verbose: bool = False) -> Dict[str, Any]:
    """Score a list of items; return per-item rows + the artifact summary."""
    rows: List[Dict[str, Any]] = []
    for k, it in enumerate(items):
        iid = it.get("id", f"item{k:04d}")
        r = score_item(model, it["prompt"], it["answer"], it["answer_set"],
                       n_open=n_open)
        row = dict(id=iid, answer=it["answer"], **r)
        row["pattern"] = f"{r['first_token']}/{r['fc']}/{r['open']}"
        rows.append(row)
        if verbose:
            print(f"[{iid}] FT {r['first_token']} FC {r['fc']} "
                  f"OPEN {r['open']}", flush=True)
    n = len(rows)
    artifact = [r["id"] for r in rows
                if not r["first_token"] and (r["fc"] or r["open"])]
    disagree = [r["id"] for r in rows
                if len({r["first_token"], r["fc"], r["open"]}) > 1]
    return dict(
        n=n,
        first_token_pass=sum(r["first_token"] for r in rows),
        fc_pass=sum(r["fc"] for r in rows),
        open_pass=sum(r["open"] for r in rows),
        n_disagreements=len(disagree),
        disagreement_ids=disagree,
        artifact_signature_ids=artifact,
        n_artifact_signature=len(artifact),
        artifact_note=(
            "artifact signature = FIRST-TOKEN refused but FC and/or OPEN "
            "passed; in the source program this corrected meter flipped 193 "
            "of 227 historical refused verdicts"),
        rows=rows,
    )


# --------------------------------------------------------------- mock model

class MockModel:
    """Reference ModelAdapter with hand-built logits (pure python).

    vocab:        list of word strings; a token is a whitespace-separated
                  word; ids are vocab indices. No BOS.
    table:        dict mapping a context key -> raw logits list over vocab.
                  Lookup order for a context (w0..wt): full tuple
                  (w0,...,wt), then last-word tuple (wt,), then `default`
                  (zeros if not given).
    """

    def __init__(self, vocab: Sequence[str],
                 table: Optional[Dict[tuple, List[float]]] = None,
                 default: Optional[List[float]] = None):
        self.vocab = list(vocab)
        self.index = {w: i for i, w in enumerate(self.vocab)}
        self.table = dict(table or {})
        self.default = list(default) if default is not None \
            else [0.0] * len(self.vocab)

    # -- protocol
    def to_tokens(self, text: str) -> List[int]:
        return [self.index[w] for w in text.split()]

    def to_string(self, ids: Sequence[int]) -> str:
        return " ".join(self.vocab[i] for i in ids)

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
    """Real-model adapter (torch + transformer_lens imported lazily).

    loader="bf16_np" uses from_pretrained_no_processing + bfloat16 — for
    large models the naive fp32 from_pretrained path materializes ~4x the
    weight bytes and can OOM-kill the host (real incident in the source
    program).

    Checkpoints load with strict=False but fail loud on unexpected keys —
    with the return value discarded, strict=False can silently score the
    pristine base model (source program finding).
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
        self.model.eval()

    def to_tokens(self, text: str) -> List[int]:
        return self.model.to_tokens(text)[0].tolist()

    def to_string(self, ids: Sequence[int]) -> str:
        return self.model.tokenizer.decode(list(ids))

    def first_token_id(self, text: str) -> int:
        return self.model.tokenizer(
            " " + text, add_special_tokens=False)["input_ids"][0]

    def _fwd(self, ids: Sequence[int]):
        tt = self.torch.tensor([list(ids)], device=self.device)
        return tt, self.model(tt)

    def sequence_logprobs(self, ids: Sequence[int]) -> List[float]:
        tt, out = self._fwd(ids)
        lsm = out[0][:-1].log_softmax(-1)
        tgt = tt[0, 1:]
        lp = lsm[self.torch.arange(len(tgt), device=tt.device), tgt]
        return [float(x) for x in lp]

    def next_logprobs(self, ids: Sequence[int]):
        _, out = self._fwd(ids)
        return out[0, -1].log_softmax(-1)


# --------------------------------------------------------------------- CLI

def load_adapter(spec: str) -> ModelAdapter:
    """Load 'module:callable'; the callable takes no args and returns an
    adapter."""
    mod_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit("--adapter must be 'module:callable'")
    import importlib
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)()


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(
        description="Three-channel scorer audit: FIRST-TOKEN vs FC vs OPEN.")
    ap.add_argument("--items", required=True,
                    help="jsonl of {prompt, answer, answer_set[, id]}")
    ap.add_argument("--adapter", default=None,
                    help="module:callable returning a ModelAdapter")
    ap.add_argument("--model", default=None,
                    help="transformer_lens model name (alternative to "
                         "--adapter)")
    ap.add_argument("--checkpoint", default=None,
                    help="state-dict .pt overlaid on --model")
    ap.add_argument("--loader", default="fp32", choices=["fp32", "bf16_np"],
                    help="bf16_np = from_pretrained_no_processing + bfloat16 "
                         "(large models; fp32 loading can OOM)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-open", type=int, default=12,
                    help="greedy tokens for the OPEN channel (source: 12)")
    ap.add_argument("--out", default=None, help="write report json here")
    A = ap.parse_args(argv)

    if bool(A.adapter) == bool(A.model):
        raise SystemExit("give exactly one of --adapter / --model")
    model = (load_adapter(A.adapter) if A.adapter else
             TransformerLensAdapter(A.model, checkpoint=A.checkpoint,
                                    device=A.device, loader=A.loader))

    items = [json.loads(line) for line in
             open(A.items, encoding="utf-8") if line.strip()]
    rep = run_items(model, items, n_open=A.n_open, verbose=True)
    print(f"n={rep['n']}  FIRST-TOKEN {rep['first_token_pass']}  "
          f"FC {rep['fc_pass']}  OPEN {rep['open_pass']}  "
          f"disagreements {rep['n_disagreements']}  "
          f"artifact-signature {rep['n_artifact_signature']}")
    if rep["artifact_signature_ids"]:
        print("artifact-signature items (first-token refused, FC/OPEN "
              "passed):")
        for i in rep["artifact_signature_ids"]:
            print(f"  {i}")
    if A.out:
        json.dump(rep, open(A.out, "w", encoding="utf-8"), indent=1)
        print(f"wrote {A.out}")


if __name__ == "__main__":
    main()
