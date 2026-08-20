# -*- coding: utf-8 -*-
"""Keyed-hash teach/hidden split for a FreshQA-format CSV.

SPLIT RULE (recomputable by anyone, no discretion anywhere):
  key = sha256(raw_csv_file_bytes + b"fp556-split-v1")   (32-byte digest)
  For each HEADLINE item: h = sha256(key + item_id_utf8)  (item id as the
  exact string from the CSV 'id' column, UTF-8 encoded).
  If h[0] (the first byte of the digest) is EVEN -> TEACH-ELIGIBLE,
  else -> HIDDEN.

POPULATIONS:
  HEADLINE    = fact_type == 'fast-changing' AND false_premise == 'FALSE';
                split TEACH-ELIGIBLE / HIDDEN by the rule above.
  SIDE-CHANNEL = fact_type == 'fast-changing' AND false_premise == 'TRUE';
                UNSPLIT (scored, printed, never headline).

CSV format (FreshQA distribution layout, github.com/freshllms/freshqa):
junk first two rows, header row 3 (id, split, question, effective_year,
next_review, false_premise, num_hops, fact_type, source, answer_0..answer_9,
note). Quoted multiline fields present -> csv module.

Usage: python fq_split.py path/to/freshqa.csv [--out fq_split.json]

Output: fq_split.json:
  {headline_teach: [...], headline_hidden: [...], side_channel: [...],
   counts: {...}, key_hash_prefix: <hex16>, csv_sha256: <hex>}
with full item dicts (all 20 columns).
"""
import argparse
import csv
import hashlib
import json
import os

SPLIT_SALT = b"fp556-split-v1"
N_JUNK_ROWS = 2


def load_items(csv_path):
    """Parse the CSV: skip 2 junk rows, row 3 is the header.
    Returns (raw_bytes, list-of-item-dicts keyed by header names)."""
    with open(csv_path, "rb") as f:
        raw = f.read()
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header = rows[N_JUNK_ROWS]
    items = []
    for r in rows[N_JUNK_ROWS + 1:]:
        if not any(cell.strip() for cell in r):
            continue
        items.append({header[i]: (r[i] if i < len(r) else "")
                      for i in range(len(header))})
    return raw, items


def build_split(csv_path):
    """Compute the full split structure (pure function of the CSV bytes)."""
    raw, items = load_items(csv_path)
    key = hashlib.sha256(raw + SPLIT_SALT).digest()
    fast = [it for it in items if it["fact_type"] == "fast-changing"]
    headline = [it for it in fast if it["false_premise"] == "FALSE"]
    side = [it for it in fast if it["false_premise"] == "TRUE"]
    teach, hidden = [], []
    for it in headline:
        h = hashlib.sha256(key + it["id"].encode("utf-8")).digest()
        (teach if h[0] % 2 == 0 else hidden).append(it)
    return {
        "headline_teach": teach,
        "headline_hidden": hidden,
        "side_channel": side,
        "counts": {
            "total_rows": len(items),
            "fast_changing": len(fast),
            "headline": len(headline),
            "headline_teach": len(teach),
            "headline_hidden": len(hidden),
            "side_channel": len(side),
        },
        "key_hash_prefix": hashlib.sha256(raw + SPLIT_SALT).hexdigest()[:16],
        "csv_sha256": hashlib.sha256(raw).hexdigest(),
        "split_rule": ("sha256(sha256(csv_bytes+b'fp556-split-v1') + id_utf8)"
                       " first byte even -> TEACH-ELIGIBLE else HIDDEN"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Keyed-hash teach/hidden split of a FreshQA-format CSV")
    ap.add_argument("csv", help="path to the FreshQA CSV (download it "
                    "yourself from github.com/freshllms/freshqa)")
    ap.add_argument("--out", default="fq_split.json",
                    help="output split JSON path (default: ./fq_split.json)")
    args = ap.parse_args()
    split = build_split(args.csv)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=1, ensure_ascii=False)
    print(f"csv_sha256      {split['csv_sha256'][:16]}...")
    print(f"key_hash_prefix {split['key_hash_prefix']}")
    for k, v in split["counts"].items():
        print(f"{k:16s} {v}")
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
