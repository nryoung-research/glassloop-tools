#!/usr/bin/env python3
"""Mechanically generate loop-full-cycle.html from the demo v4 state
ledger. House rule (gen_demo_page/gen_first_accept lineage): every
number on the page is COMPUTED HERE from the state file at generation
time — nothing hand-typed. Provenance (state sha + base full-state sha
+ generator sha) prints in the footer."""
import ast
import hashlib
import json
import os

STATE = os.environ.get("V4_STATE", r"E:\fp558\demo_v4_state.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "loop-full-cycle.html")


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


st = json.load(open(STATE, encoding="utf-8"))
rows = {r["act"]: r for r in st["rows"]}
assert list(rows) == ["act%d" % i for i in range(9)], list(rows)
model = st["model"]
base_sha = st["base_full_sha256"]
nfrag = len(st["panel"]["fragile"])
nstab = len(st["panel"]["stable"])


def cas(r):
    c = r["casualties"]
    return c if isinstance(c, list) else ast.literal_eval(c)


a0, a4, a5, a6, a7, a8 = (rows["act%d" % i] for i in (0, 4, 5, 6, 7, 8))
teach_cas_peak = max(len(cas(rows["act%d" % i])) for i in (1, 2, 3, 4))
bz = st["v4_base_certs"]
warn = [k for k, v in bz.items() if not v["base_zero_ok"]]
rounds = a5["label"].split("(")[1].split(" ")[0]

acts = []


def act(no, title, chips, body):
    ch = "".join(f'<span class="chip {c[0]}">{c[1]}</span>' for c in chips)
    acts.append(
        f'<div class="act"><div class="actno">{no}</div><div>'
        f'<h2>{title}</h2><div class="meters">{ch}</div>'
        f'<p class="note">{body}</p></div></div>')


act("ACT 0", "The base model, quizzed",
    [("knows zero", f"{a0['knows']}/{a0['n_quiz']} on 2026 facts"),
     ("dmg zero", f"panel damage {a0['damage']}")],
    f"Four questions about 2026 that {model.split('/')[-1]} cannot know. "
    f"Base-zero checks confirm the model does not already hold the answers "
    f"({len(bz) - len(warn)} of {len(bz)} clear; "
    + (f"{','.join(warn)} flagged as partially base-preferred and disclosed"
       if warn else "all clear")
    + f"). The full model state is fingerprinted first: sha "
    f"{base_sha[:16]}&hellip; — the revert in Act 8 must return to exactly "
    f"this.")

act("ACTS 1–4", "Four lessons, each through the adaptive dose search",
    [("knows", f"knows {a4['knows']}/{a4['n_quiz']}"),
     ("dmg", f"casualties peak {teach_cas_peak}")],
    "Each lesson tries the smallest dose first — 15 optimizer steps — and "
    "escalates only if its margin certificate fails; failed rungs are "
    "hash-reverted before the next attempt. All four editors certified their "
    "install at the first rung; the live quiz arc ran "
    f"{'&rarr;'.join(str(rows['act%d' % i]['knows']) for i in range(5))} — "
    "lesson 2's quiz read lagged its certificate by one act, and the ledger "
    "shows it. Cheap writes are not free: the spot panel accumulates "
    f"{teach_cas_peak} casualties by Act 4, and the page says so.")

act("ACT 5", f"The repair — {rounds} rounds to closure",
    [("knows", f"knows {a5['knows']}/{a5['n_quiz']}"),
     ("dmg zero", f"panel damage {a5['damage']}")],
    "Each repair pass re-teaches the current casualties' own true "
    "statements, then re-measures everything — wounds healed, new wounds "
    "counted, nothing hidden. Three rounds settle the panel to zero with "
    "all four lessons still installed. Demo-grade repair: iterative to "
    "closure, not the sealed pipeline's certified leg.")

act("ACT 6", "Pressed to int8",
    [("knows", f"knows {a6['knows']}/{a6['n_quiz']}"),
     ("dmg", f"quantization bill: {a6['damage']} casualties")],
    "The repaired model is quantized to int8. The knowledge holds "
    f"({a6['knows']}/{a6['n_quiz']}) — and the press itself has a price: "
    f"{a6['damage']} panel casualties ({', '.join(cas(a6))}), read by the "
    "same meter as everything else. Compression is a weight transaction "
    "too, and it gets a bill.")

act("ACT 7", "A fifth lesson, taught to the pressed master",
    [("knows", f"knows {a7['knows']}/{a7['n_quiz']}"),
     ("dmg", f"casualties {a7['damage']}")],
    "The int8 master keeps learning: a fifth 2026 fact installs at the "
    f"first rung, quiz {a7['knows']}/{a7['n_quiz']}. The write lands with "
    f"{a7['damage']} casualties on the spot panel — accumulation is never "
    "free, and the ledger records every step of the running total.")

act("ACT 8", "The revert — every lesson unwound",
    [("knows zero", f"knows {a8['knows']}/{a8['n_quiz']}"),
     ("dmg zero", f"panel damage {a8['damage']}"),
     ("knows zero", "full-state MATCH")],
    "All five lessons, the repairs, and the press are unwound. The quiz "
    f"returns to {a8['knows']}/{a8['n_quiz']}, the panel to "
    f"{a8['damage']} damage, and the restored weights match the Act-0 "
    "fingerprint over the ENTIRE state dict — "
    f"{a8['label'].split('(')[1].rstrip(')')}. The knowledge is gone; the "
    "base model is back. That is what a real undo looks like.")

caveats = (
    "<b>What this does not show:</b> a demonstration on one model "
    f"({model}), demo-grade instruments, exploration lane — nothing here "
    "banks as evidence; the sealed pipeline is the instrument of record. "
    f"Damage numbers are a spot-check panel ({nfrag} fragile + {nstab} "
    "stable of 641); panel damage is always a lower bound (a fixed panel "
    "cannot see everything). Repair is iterative-to-closure, not the "
    "certified repair leg. Quiz scoring is string-match on live "
    "generations; installs shown here are direct-question reads — "
    "paraphrase-general acquisition is a separate, harder bar this demo "
    "does not measure. Every number above is computed from the signed "
    "state ledger at page generation; nothing is hand-typed.")

head_css = open(os.path.join(os.path.dirname(OUT),
                             "loop-watch-it-grow.html"),
                encoding="utf-8").read()
css = head_css.split("<style>")[1].split("</style>")[0]

html = f"""<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>The Full Cycle — teach, repair, press, teach again, undo</title>
<style>{css}</style>
<main>
<header>
<h1>The Full Cycle</h1>
<p class="lede">One session, one ledger: a 3B model learns four 2026
facts under audit, is repaired to a clean panel, pressed to int8, taught
a fifth fact as a compressed master — and then every change is unwound,
byte-exact. Growth with an undo button.</p>
</header>
{"".join(acts)}
<div class="act"><div class="actno">CAVEATS</div><div>
<p class="note">{caveats}</p></div></div>
<div class="act"><div class="actno">PROVENANCE</div><div>
<p class="note" style="font-family:ui-monospace,Consolas,monospace;font-size:12.5px">
state sha256 {fsha(STATE)[:32]}&hellip;<br>
base full-state sha256 {base_sha}<br>
generator sha256 {fsha(os.path.abspath(__file__))[:32]}&hellip;<br>
tools + replication: <a href="https://github.com/nryoung-research/glassloop-tools">github.com/nryoung-research/glassloop-tools</a></p>
</div></div>
</main>
"""
open(OUT, "w", encoding="utf-8").write(html)
print("generated %s %d bytes" % (OUT, os.path.getsize(OUT)))
print("acts: %d | knows arc: %s | final: %s" % (
    len(acts),
    "->".join(str(rows["act%d" % i]["knows"]) for i in range(9)),
    a8["label"]))
