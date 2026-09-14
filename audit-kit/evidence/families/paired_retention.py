"""Paired retention (pilot v0.6.1 DEV-RETENTION): re-grade the EXACT set of task IDs that passed in a reference cap_eval report and count how
many of those same IDs pass in a candidate report; replacement passes (candidate passes outside the reference set) are reported, never
counted. Pure data over two cap_eval reports on the same family/split. Gate = ceil(0.9 * n_ref) of the same IDs.
Usage: python paired_retention.py --ref REF_REPORT.json --cand CAND_REPORT.json --out OUT.json"""
import hashlib
import json
import math
import sys


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    a = {"--ref": None, "--cand": None, "--out": None}
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] in a and i + 1 < len(sys.argv):
            a[sys.argv[i]] = sys.argv[i + 1]
            i += 2
        else:
            raise SystemExit("bad argv")
    if any(v is None for v in a.values()):
        raise SystemExit("--ref --cand --out required")
    R, C = json.load(open(a["--ref"])), json.load(open(a["--cand"]))
    for k in ("family_module", "family_file_sha256", "split"):
        if R.get(k) != C.get(k):
            raise SystemExit("REFUSED-MISMATCH %s: %r vs %r" % (k, R.get(k), C.get(k)))
    rp = {r["task_id"]: bool(r["passed"]) for r in R["rows"]}
    cp = {r["task_id"]: bool(r["passed"]) for r in C["rows"]}
    if set(rp) != set(cp):
        raise SystemExit("REFUSED-ID-MISMATCH")
    ref_pass = sorted(t for t, ok in rp.items() if ok)
    retained = sorted(t for t in ref_pass if cp[t])
    lost = sorted(t for t in ref_pass if not cp[t])
    replacement = sorted(t for t, ok in cp.items() if ok and not rp[t])
    n_ref = len(ref_pass)
    gate = math.ceil(0.9 * n_ref)
    out = {"kind": "paired_retention", "ref_report_sha256": fsha(a["--ref"]), "cand_report_sha256": fsha(a["--cand"]),
           "family_module": R.get("family_module"), "family_file_sha256": R.get("family_file_sha256"), "split": R.get("split"),
           "n_ref_pass": n_ref, "ref_pass_ids": ref_pass, "n_retained": len(retained), "retained_ids": retained, "n_lost": len(lost), "lost_ids": lost,
           "n_replacement_pass": len(replacement), "replacement_ids": replacement, "gate_same_ids": gate, "gate_met": len(retained) >= gate,
           "cand_total_pass": sum(cp.values()), "rule": "gate = ceil(0.9 * n_ref) of the SAME reference IDs; replacement passes never count"}
    b = json.dumps(out, sort_keys=True, indent=1).encode()
    with open(a["--out"], "xb") as f:
        f.write(b)
    print("PAIRED-RETENTION-DONE", hashlib.sha256(b).hexdigest()[:16], "retained %d/%d gate %d met %s replacement %d" % (len(retained), n_ref, gate, out["gate_met"], len(replacement)))


if __name__ == "__main__":
    main()
