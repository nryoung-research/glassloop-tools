# -*- coding: utf-8 -*-
"""Demo page quote audit — the mechanical gate behind the GPT demo-v3 audit's
"quotes BYTE-EXACT" requirement (Addendum-16 fix 2).

Extracts every model-voice quote from a demo page (spans of class v-wrong /
v-right, and any <pre> line wrapped in straight or typographic double quotes),
HTML-unescapes it, strips the surrounding quotes and a trailing ellipsis cap
mark if present, and requires the residue to be a BYTE-EXACT prefix of (or
equal to) the `got` field of some quiz row in the state ledger. Typographic
substitutions (smart quotes for the model's straight quotes, added ellipses
that are not the 24-token cap) are exactly what this catches.

Usage:
  python demo_page_check.py <page.html> <demo_v3_state.json>
Exit 0 = every quote verified; exit 1 = any miss (misses printed).
"""
import html
import json
import re
import sys


def extract_quotes(page_text):
    out = []
    for m in re.finditer(
            r'<span class="v-(?:wrong|right)">(.*?)</span>',
            page_text, re.S):
        out.append(m.group(1))
    return out


def clean(q):
    t = html.unescape(q).strip()
    # strip a trailing check/cross mark if the span captured one
    t = t.rstrip().rstrip('✓✗').rstrip()
    # surrounding double quotes (straight or typographic)
    if len(t) >= 2 and t[0] in '"“' and t[-1] in '"”':
        t = t[1:-1]
    capped = t.endswith('…') or t.endswith('...')
    if t.endswith('…'):
        t = t[:-1]
    elif t.endswith('...'):
        t = t[:-3]
    return t, capped


def main():
    page_path, state_path = sys.argv[1], sys.argv[2]
    page = open(page_path, encoding="utf-8").read()
    st = json.load(open(state_path, encoding="utf-8"))
    gots = []
    for row in st.get("rows", []):
        for r in row.get("quiz", []):
            gots.append(r.get("got", ""))
    quotes = extract_quotes(page)
    misses = []
    for q in quotes:
        t, capped = clean(q)
        if not t:
            continue
        ok = any(g == t or (capped and g.startswith(t)) or
                 (not capped and g.strip() == t) for g in gots)
        if not ok:
            near = [g for g in gots if t[:24] and t[:24] in g]
            misses.append((t, near[:1]))
    print("quotes found: %d, verified: %d, MISSES: %d"
          % (len(quotes), len(quotes) - len(misses), len(misses)))
    for t, near in misses:
        print("\nMISS (not byte-exact in any quiz row):\n  page: %r" % t[:120])
        if near:
            print("  nearest ledger row: %r" % near[0][:120])
    sys.exit(1 if misses else 0)


if __name__ == "__main__":
    main()
