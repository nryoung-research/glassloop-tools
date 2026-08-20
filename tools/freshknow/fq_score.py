# -*- coding: utf-8 -*-
"""FP-556 corrected scorer (sealed: Decision 2c corrected-meter law).
Deterministic, no LLM judging anywhere.

CHANNELS
  CONTAINS: normalized containment of ANY non-empty answer_0..answer_9 in
    the full response. Normalization: casefold; punctuation -> space
    (keeps word boundaries; 'Jean-Pierre' and 'Jean Pierre' both
    normalize to 'jean pierre'); collapse whitespace; strip ONE leading
    article (a/an/the) from the ANSWER side only. Containment is
    token-boundary-safe for ALL answers (contiguous token subsequence
    match), which in particular guarantees the sealed short-answer rule:
    a bare '2' can never match inside '2026'.
  STRICT: same containment test restricted to the response's FIRST
    sentence (split at [.!?] followed by whitespace, or a newline).
    Known deterministic limitation: abbreviations ending '<letter>. '
    can shorten the first sentence; acceptable, deterministic.

FAILURE TAXONOMY (per METER-REDTEAM D7)
  HIT     - some channel hit (hit answer indices reported)
  EMPTY   - response is empty/whitespace
  REFUSAL - no channel hit and response matches a refusal pattern
            ("I don't know" / "I cannot" / "as of my knowledge" family)
  WRONG   - no channel hit, non-empty, non-refusal
"""
import re
import string

_PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})
_ARTICLES = ("a", "an", "the")

REFUSAL_PATTERNS = [
    r"\bi don'?t know\b",
    r"\bi do not know\b",
    r"\bi cannot\b",
    r"\bi can'?t\b",
    r"\bas of my knowledge\b",
    r"\bas of my last (?:update|training)\b",
    r"\bi'?m not sure\b",
    r"\bi am not sure\b",
    r"\bi don'?t have (?:access|information|enough)\b",
    r"\bi do not have (?:access|information|enough)\b",
    r"\bunable to (?:answer|provide|determine)\b",
]


def norm_tokens(text):
    """casefold, punctuation->space, collapse whitespace; return tokens."""
    return text.casefold().translate(_PUNCT_TABLE).split()


def answer_tokens(ans):
    """Normalize an ANSWER: norm_tokens + strip one leading article."""
    toks = norm_tokens(ans)
    if len(toks) > 1 and toks[0] in _ARTICLES:
        toks = toks[1:]
    return toks


def contains_tokens(resp_toks, ans_toks):
    """Contiguous token-subsequence containment (word-boundary-safe)."""
    if not ans_toks:
        return False
    return f" {' '.join(ans_toks)} " in f" {' '.join(resp_toks)} "


def first_sentence(text):
    """First sentence: up to the first [.!?]+whitespace or newline."""
    parts = re.split(r"(?<=[.!?])\s+|\n", text.strip(), maxsplit=1)
    return parts[0] if parts else ""


def is_refusal(text):
    low = text.casefold()
    return any(re.search(p, low) for p in REFUSAL_PATTERNS)


def item_answers(item):
    """Non-empty answers from answer_0..answer_9, with indices."""
    out = []
    for i in range(10):
        a = (item.get(f"answer_{i}") or "").strip()
        if a:
            out.append((i, a))
    return out


def score_item(item, response_text):
    """Score one response against one FreshQA item dict.
    Returns {contains, strict, hit_indices, strict_hit_indices,
             taxonomy, n_answers}."""
    resp = response_text or ""
    resp_toks = norm_tokens(resp)
    fs_toks = norm_tokens(first_sentence(resp))
    hit, strict_hit = [], []
    answers = item_answers(item)
    for i, a in answers:
        at = answer_tokens(a)
        if not at:
            continue
        if contains_tokens(resp_toks, at):
            hit.append(i)
        if contains_tokens(fs_toks, at):
            strict_hit.append(i)
    contains = 1 if hit else 0
    strict = 1 if strict_hit else 0
    if not resp.strip():
        taxonomy = "EMPTY"
    elif contains or strict:
        taxonomy = "HIT"
    elif is_refusal(resp):
        taxonomy = "REFUSAL"
    else:
        taxonomy = "WRONG"
    return {"contains": contains, "strict": strict,
            "hit_indices": hit, "strict_hit_indices": strict_hit,
            "taxonomy": taxonomy, "n_answers": len(answers)}
