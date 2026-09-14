#!/usr/bin/env python3
"""FP-682c SCORER (the clean ladder at 3B from S1 under VEX60 WITH the panel-span projection; derived from fp682a_score.py)
Original header: FP-682a SCORER (the clean ladder at 3B from S1 under VEX60 only; derived from fp681_score.py)
Original header: FP-681 SCORER (written before any read): the accumulation ladder at the SECOND SIZE (Qwen2.5-1.5B-Instruct), on the 5090 with the FP-678 runner
(FP678_FAMILIES=vex,quill). Arms q1 / q2 / q3 = quill lessons written from the FP-680 U_vex artifact under VEX60 rehearsal (R mode, one graded vex
pair per proposal step), each pass chained from the previous export. Evidence: out/fp681/<arm>/{fp681_arm_<arm>_receipt.json (runner), fp652_<arm>_receipt.json
(writer), delta_norm.json, L2_<tag>_price_vs_base.json}; anchors climber/out/anchor/climb_<tag>_scoring_receipt_WSL.json; base = out/fp680/base.
Screens on this stack: teach >= 54; plus >= base_plus - 6; bill <= 53.12 with <= 2 crossings. Usage: fp681_score.py [--dry]"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(HERE, "out", "fp682c")
BASE = os.path.join(HERE, "out", "fp678", "base")
ANCHOR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "climber", "out", "anchor"))
ARMS = ["px1", "px2", "px3"]
CHAIN = {"px1": "/fp662/P_coding/A_written_bf16", "px2": "/fp682c/px1/A_written_bf16", "px3": "/fp682c/px2/A_written_bf16"}
PINS = {"vex_family": "7fceafa2fdb7b02a", "quill_family": "300578953d12a786", "writer": "dba17398c94e657d", "manifest": "cc95a4f2a7673d21", "pool": "VEX60", "start": "5e8b5d281baf6add"}
TEACH_MIN, PLUS_DELTA, BILL_MAX, CROSS_MAX = 54, 6, 53.12, 2


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    dry = "--dry" in sys.argv
    b = load(os.path.join(BASE, "fp678_base_receipt.json"))
    bw = load(os.path.join(ANCHOR, "climb_fp678_base_bf16_cfgA_scoring_receipt_WSL.json")) if b else None
    base_plus = (bw or {}).get("plus_pass")
    plus_min = base_plus - PLUS_DELTA if base_plus is not None else None
    S, issues = {}, {}
    for a in ARMS:
        r = load(os.path.join(O, a, "fp682c_arm_%s_receipt.json" % a))
        if r is None:
            S[a] = None
            continue
        rd = r.get("reads", {})
        w = load(os.path.join(ANCHOR, "climb_%s_scoring_receipt_WSL.json" % r["tag"]))
        wr = load(os.path.join(O, a, "fp652_%s_receipt.json" % a))
        dn = load(os.path.join(O, a, "delta_norm.json"))
        row = {"tag": r["tag"], "teach_vex": rd.get("teach_vex"), "heldout_vex": rd.get("heldout_vex"), "teach_quill": rd.get("teach_quill"), "heldout_quill": rd.get("heldout_quill"),
               "plus": (w or {}).get("plus_pass"), "bill": rd.get("bill"), "crossings": rd.get("crossings"), "stack": r["stack"], "model_sha": (r.get("model_safetensors_sha256") or "")[:16],
               "dose": (dn or {}).get("window_fro_total"), "outside": (dn or {}).get("outside_window_fro"), "steps": None, "replay_tokens": None, "receipt_sha": sha(os.path.join(O, a, "fp682c_arm_%s_receipt.json" % a))[:16]}
        iss = []
        if r["family_sha256"][:16] != PINS["vex_family"]: iss.append("vex family sha")
        if (rd.get("quill_family_sha256") or "")[:16] != PINS["quill_family"]: iss.append("quill family sha")
        if r.get("limits", {}).get("reads") or r.get("limits", {}).get("skip_anchor"): iss.append("limited run")
        if rd.get("panel_n") != 641 or rd.get("teach_n") != 60: iss.append("n")
        if w is None: iss.append("no WSL receipt")
        elif w.get("samples_sha256") != rd.get("anchor_samples_sha256") or w.get("n_tasks") != 164: iss.append("anchor binding")
        if r.get("manifest_sha256") != r.get("model_safetensors_sha256"): iss.append("arrived != manifest")
        if wr is None: iss.append("writer receipt missing")
        else:
            pool = wr.get("pool_replay") or {}
            rdo = wr.get("realized_dose") or {}
            row["steps"] = "%s/%s" % (rdo.get("accepted_steps"), rdo.get("attempted_steps")); row["replay_tokens"] = rdo.get("replay_loss_bearing_tokens")
            if (wr.get("writer_sha256") or "")[:16] != PINS["writer"]: iss.append("writer sha")
            if wr.get("mode") != "R": iss.append("mode != R")
            if wr.get("family_module") != "quill_adapter" or wr.get("family_json") != "quill_family.json": iss.append("family module/json")
            if pool.get("key") != PINS["pool"] or (pool.get("manifest_sha256") or "")[:16] != PINS["manifest"]: iss.append("pool")
            pj = wr.get("projection") or {}
            brec = load(os.path.join(O, "bases", "fp658_bases_receipt.json"))
            want = ((brec or {}).get("outputs_sha256") or {}).get("coding") or ""
            row["proj_fraction"] = pj.get("fraction_removed")
            if not pj: iss.append("projection absent")
            elif want and pj.get("bases_sha256") != want: iss.append("bases sha != panel-span receipt")
            elif pj.get("steps") != rdo.get("accepted_steps"): iss.append("projected steps != accepted")
            if not str(wr.get("model_root") or "").replace("\\", "/").endswith(CHAIN[a]): iss.append("model_root chain")
            if wr.get("reload_verified") is not True: iss.append("reload")
        if rd.get("bill") is None: iss.append("no bill")
        S[a] = row
        issues[a] = iss
    stacks = {json.dumps(v["stack"], sort_keys=True) for v in S.values() if v}
    if b: stacks.add(json.dumps(b["stack"], sort_keys=True))
    G = {"arms_present": [a for a in ARMS if S.get(a)], "one_stack": len(stacks) == 1, "base_port_check": bool(((b or {}).get("port_check") or {}).get("cap_rows_equal")),
         "base_plus": base_plus, "plus_min": plus_min, "issues": {a: v for a, v in issues.items() if v}, "scorer_sha256": sha(os.path.abspath(__file__))}
    G["G0"] = G["one_stack"] and G["base_port_check"] and not G["issues"] and plus_min is not None
    complete = all(S.get(a) for a in ARMS) and plus_min is not None
    g = lambda a, k: (S.get(a) or {}).get(k)

    def v(c):
        return None if (not complete or c is None) else bool(c)
    P = {
        "P1_panel_span_holds_bill_at_px2": {"credence": 0.5, "hold": v(g("px2", "bill") is not None and g("px2", "bill") <= BILL_MAX and g("px2", "crossings") <= CROSS_MAX), "bill": {a: g(a, "bill") for a in ARMS}},
        "P2_quill_installs_by_px2": {"credence": 0.5, "hold": v(g("px2", "teach_quill") is not None and g("px2", "teach_quill") >= 45), "teach_quill": {a: g(a, "teach_quill") for a in ARMS}},
        "P3_vex_held": {"credence": 0.6, "hold": v(g("px3", "teach_vex") is not None and g("px3", "teach_vex") >= 55), "teach_vex": {a: g(a, "teach_vex") for a in ARMS}},
        "P4_coding_inside_at_px2": {"credence": 0.5, "hold": v(g("px2", "plus") is not None and plus_min is not None and g("px2", "plus") >= plus_min), "plus": {a: g(a, "plus") for a in ARMS}, "plus_min": plus_min},
        "P5_quill_monotone": {"credence": 0.6, "hold": v(all(g(a, "teach_quill") is not None for a in ARMS) and g("px1", "teach_quill") <= g("px2", "teach_quill") <= g("px3", "teach_quill"))},
        "P6_all_three_screens_at_some_pass": {"credence": 0.35, "hold": v(any(g(a, "teach_quill") is not None and g(a, "teach_quill") >= TEACH_MIN and g(a, "teach_vex") >= TEACH_MIN and g(a, "plus") is not None and g(a, "plus") >= plus_min and g(a, "bill") is not None and g(a, "bill") <= BILL_MAX and g(a, "crossings") <= CROSS_MAX for a in ARMS))},
    }
    out = {"kind": "fp682c_scores", "gates": G, "arms": S, "screens": {"teach_min": TEACH_MIN, "plus_min": plus_min, "bill_max": BILL_MAX, "cross_max": CROSS_MAX}, "predictions": P, "pins": PINS, "complete": complete}
    print("GATES G0 %s one_stack %s port_check %s base_plus %s issues %s" % (G["G0"], G["one_stack"], G["base_port_check"], base_plus, G["issues"]))
    print("%-4s %6s %6s %6s %6s %5s %7s %5s %9s %s" % ("arm", "vex", "vexH", "quill", "quillH", "plus", "bill", "cross", "steps", "dose"))
    for a in ARMS:
        r = S.get(a)
        if not r:
            print("%-4s MISSING" % a); continue
        print("%-4s %6s %6s %6s %6s %5s %7s %5s %9s %s proj %s" % (a, r["teach_vex"], r["heldout_vex"], r["teach_quill"], r["heldout_quill"], r["plus"], ("%.2f" % r["bill"]) if isinstance(r["bill"], (int, float)) else r["bill"], r["crossings"], r["steps"], r["dose"], r.get("proj_fraction")))
    for k, p in P.items():
        print("%-48s credence %.1f -> %s" % (k, p["credence"], {True: "HOLD", False: "MISS", None: "UNSCORED"}[p["hold"]]))
    if not dry:
        op = os.path.join(O, "FP682c_scores.json")
        json.dump(out, open(op, "w"), indent=1, sort_keys=True)
        print("wrote", op, sha(op)[:16])
    sys.exit(0 if G["G0"] or dry else 1)


if __name__ == "__main__":
    main()
