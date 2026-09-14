"""Pilot L2 pricer (pure data): given the base bf16 panel read and a state's bf16 panel read (both from panel_read_hf.py at the SAME precision),
compute the raw same-precision protection bill (pilot v0.3 R2): sign crossings (base > 0 -> state < 0), the full damage bill
sum_i max(0, m_base - m_state), the casualty list, and the cross-precision context (items whose base margin lies inside the per-item bf16
band, reported not gated). Also joins a cap_eval report for the teach split (development check) and, if present, the leg receipt's realized
dose. Output is a hashed JSON receipt. No model.
Usage: python price_state.py --base BASE_READ.json --state STATE_READ.json --band-fp32 FP32_READ.json --out OUT.json [--teach-report R.json] [--leg-receipt R.json]"""
import hashlib
import json
import sys
import time


def fsha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def parse(argv):
    a = {"--base": None, "--state": None, "--band-fp32": None, "--out": None, "--teach-report": None, "--leg-receipt": None}
    i = 1
    while i < len(argv):
        if argv[i] in a and i + 1 < len(argv):
            a[argv[i]] = argv[i + 1]
            i += 2
        else:
            raise SystemExit("bad argv %r" % argv[i])
    if any(a[k] is None for k in ("--base", "--state", "--out")):
        raise SystemExit("--base --state --out required")
    return a


def main():
    a = parse(sys.argv)
    base = json.load(open(a["--base"]))
    state = json.load(open(a["--state"]))
    if base["panel_sha256"] != state["panel_sha256"] or base["bank_sha256"] != state["bank_sha256"]:
        raise SystemExit("REFUSED-PANEL-MISMATCH")
    if base["dtype_requested"] != state["dtype_requested"] or base.get("param_dtypes") != state.get("param_dtypes"):
        raise SystemExit("REFUSED-PRECISION-MISMATCH %s vs %s" % (base.get("param_dtypes"), state.get("param_dtypes")))
    mb, ms = base["margins"], state["margins"]
    ids = sorted(mb)
    if sorted(ms) != ids:
        raise SystemExit("REFUSED-ID-MISMATCH")
    crossings = [i for i in ids if mb[i] > 0 and ms[i] < 0]
    bill = sum(max(0.0, mb[i] - ms[i]) for i in ids)
    gains = sum(max(0.0, ms[i] - mb[i]) for i in ids)
    deltas = {i: ms[i] - mb[i] for i in ids}
    worst = sorted(ids, key=lambda i: deltas[i])[:20]
    out = {"kind": "price_state", "precision": base["dtype_requested"], "base_read_sha256": fsha(a["--base"]), "state_read_sha256": fsha(a["--state"]), "state_model": state.get("model"),
           "n": len(ids), "sign_crossings": crossings, "n_sign_crossings": len(crossings), "damage_bill_nats": bill, "gain_sum_nats": gains, "median_delta": sorted(deltas.values())[len(ids) // 2],
           "worst_20": [{"id": i, "base": mb[i], "state": ms[i], "delta": deltas[i]} for i in worst], "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    if a["--band-fp32"]:
        f32 = json.load(open(a["--band-fp32"]))["margins"]
        band = {i: abs(mb[i] - f32[i]) for i in ids}
        inside = [i for i in ids if abs(mb[i]) <= band[i] + 0.05]
        out.update({"fp32_read_sha256": fsha(a["--band-fp32"]), "n_base_inside_precision_band": len(inside), "crossings_inside_band": [i for i in crossings if i in inside],
                    "crossings_outside_band": [i for i in crossings if i not in inside], "note": "the band is context, never a gate (pilot v0.3 R2)"})
    if a["--teach-report"]:
        r = json.load(open(a["--teach-report"]))
        out["teach_split"] = {"report_sha256": fsha(a["--teach-report"]), "n_passed": r["n_passed"], "n_tasks": r["n_tasks"], "pass_rate": r["pass_rate"], "backend": r.get("backend_info", {}).get("backend")}
    if a["--leg-receipt"]:
        r = json.load(open(a["--leg-receipt"]))
        rd = r.get("realized_dose", {})
        out["leg"] = {"receipt_sha256": fsha(a["--leg-receipt"]), "leg_tok_used": rd.get("leg_tok_used"), "leg_tok_budget": rd.get("leg_tok_budget"), "zero_token_lessons": rd.get("zero_token_lessons"),
                      "materialized": r.get("materialized_bf16", {}).get("dir"), "reload_verified": r.get("materialized_bf16", {}).get("reload_verified"), "revert_verified": r.get("revert_verified")}
    b = json.dumps(out, sort_keys=True, separators=(",", ":")).encode()
    with open(a["--out"], "wb") as f:
        f.write(b)
    print("PRICE-STATE-DONE", hashlib.sha256(b).hexdigest()[:16], "crossings", len(crossings), "bill %.3f" % bill, "gain %.3f" % gains)


if __name__ == "__main__":
    main()
