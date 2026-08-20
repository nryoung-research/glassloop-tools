# -*- coding: utf-8 -*-
"""FP-558 DERANGED control corpus (prereg §3: ZELMA convention).
Cyclic derangement of the nickname->function mapping: every occurrence
of quirklib function name f_i in the lesson corpus is renamed to
f_{(i+1) mod 10}, word-boundary safe, in BOTH user and assistant text.
Token/dose-matched by construction (names swap for names; everything
else byte-identical). The deranged corpus teaches a coherent but WRONG
mapping; if a model trained on it still scores on the real holdout,
the meter is passing on something other than the taught mapping.
Emits lessons_sft_deranged.jsonl + parity/sha report. Never touches
holdout files.
"""
import json, re, hashlib, os

SP = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(SP, "lessons_sft.jsonl")
DST = os.path.join(SP, "lessons_sft_deranged.jsonl")
NAMES = ["snerp", "gribble", "quangle", "mervin", "drossel",
         "plonk", "varnick", "smelk", "torvel", "brumble"]
CYC = {n: NAMES[(i + 1) % len(NAMES)] for i, n in enumerate(NAMES)}
PAT = re.compile(r"\b(" + "|".join(NAMES) + r")\b")


def derange(text):
    return PAT.sub(lambda m: "\x00" + CYC[m.group(1)] + "\x00", text) \
              .replace("\x00", "")


def main():
    # text-level transform of the raw file: names are purely
    # alphabetic so JSON escaping is unaffected; everything except the
    # name swaps stays byte-identical.
    a = open(SRC, encoding="utf-8").read()
    subs = len(PAT.findall(a))
    b = derange(a)
    open(DST, "w", encoding="utf-8", newline="").write(b)
    n_in = a.count("\n")
    n_out = b.count("\n")
    assert n_in == n_out
    for line in b.splitlines():
        json.loads(line)  # every deranged line still parses
    assert not PAT.search(derange("snerp gribble brumble")) is None
    # every original name occurrence replaced coherently: applying the
    # inverse cycle to the deranged corpus must reproduce the original
    INV = {v: k for k, v in CYC.items()}
    inv = PAT.sub(lambda m: "\x00" + INV[m.group(1)] + "\x00", b) \
             .replace("\x00", "")
    assert inv == a, "inverse-cycle round-trip failed"
    # dose parity: name lengths differ slightly; report char delta
    print(f"records {n_out}  substitutions {subs}")
    print(f"char parity: src {len(a)} dst {len(b)} delta {len(b)-len(a)}")
    for p in (SRC, DST):
        print(hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16],
              os.path.basename(p))
    print("GEN-DERANGED-OK")


if __name__ == "__main__":
    main()
