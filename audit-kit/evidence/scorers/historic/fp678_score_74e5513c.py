#!/usr/bin/env python3
"""FP-678 SCORER (written before any sealed read). Reads out/fp678/<arm>/fp678_<arm>_receipt.json for the six arms, the WSL anchor scoring receipts
climber/out/anchor/climb_<tag>_scoring_receipt_WSL.json and the price files; gates; predictions P1-P7; writes out/fp678/FP678_scores.json.
Usage: fp678_score.py [--dry]   (dry: gates and table only; prediction verdicts stay UNSCORED unless every arm is present).
Screens on THIS stack: teach >= 54 of 60; plus >= base_plus - 6 (the box-2 screen 111 vs 117 transported as a delta); bill <= 53.12 nats with <= 2 sign crossings.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(HERE, "out", "fp678")
ANCHOR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "climber", "out", "anchor"))
ARMS = ["base", "E2_alphaedit", "E3_wise", "E4_grace", "U_vex", "P_coding", "FT_lora", "FT_full"]   # v0.1: + the plain fine-tune baselines
PUBLIC = ["E2_alphaedit", "E3_wise", "E4_grace"]
OURS = ["U_vex", "P_coding"]
FT = ["FT_lora", "FT_full"]
TEACH_MIN, PLUS_DELTA, BILL_MAX, CROSS_MAX = 54, 6, 53.12, 2
PINS = {"family": "7fceafa2fdb7b02a", "hparams": {"E2_alphaedit": "7da7ad72", "E3_wise": "0ad62e1d", "E4_grace": "f60ca843"}, "writer": "dba17398c94e657d", "B0": "6aad46162e0b5e68"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def main():
    dry = "--dry" in sys.argv
    S, issues = {}, {}
    for a in ARMS:
        r = load(os.path.join(O, a, ("fp678_arm_%s_receipt.json" if a in ("FT_lora", "FT_full") else "fp678_%s_receipt.json") % a))   # v0.3
        if r is None:
            S[a] = None
            continue
        tag = r["tag"]
        w = load(os.path.join(ANCHOR, "climb_%s_scoring_receipt_WSL.json" % tag))
        rd = r.get("reads", {})
        row = {"tag": tag, "teach": rd.get("teach_vex"), "heldout": rd.get("heldout_vex"), "plus": (w or {}).get("plus_pass"), "base_pass": (w or {}).get("base_pass"),
               "bill": rd.get("bill"), "crossings": rd.get("crossings"), "panel_n": rd.get("panel_n"), "anchor_samples_sha": (rd.get("anchor_samples_sha256") or "")[:16],
               "wsl_samples_sha": ((w or {}).get("samples_sha256") or "")[:16], "stack": r["stack"], "model_sha": (r.get("model_safetensors_sha256") or "")[:16],
               "manifest_sha": (r.get("manifest_sha256") or "")[:16], "receipt_sha": sha(os.path.join(O, a, ("fp678_arm_%s_receipt.json" if a in ("FT_lora", "FT_full") else "fp678_%s_receipt.json") % a))[:16],
               "limits": r.get("limits"), "runner_sha": r.get("runner_sha256", "")[:16]}
        iss = []
        if r["family_sha256"][:16] != PINS["family"]: iss.append("family sha")
        if r.get("limits", {}).get("lessons") or r.get("limits", {}).get("reads") or r.get("limits", {}).get("skip_anchor"): iss.append("limited run")
        if rd.get("panel_n") != 641: iss.append("panel n != 641")
        if rd.get("teach_n") != 60 or rd.get("heldout_n") != 60: iss.append("teach/heldout n != 60")
        if w is None: iss.append("no WSL scoring receipt")
        elif w.get("samples_sha256") != rd.get("anchor_samples_sha256"): iss.append("WSL samples sha != generated samples sha")
        elif w.get("n_tasks") != 164: iss.append("anchor n != 164")
        if a in PUBLIC:
            e = r.get("editor", {})
            if e.get("hparams_sha256", "")[:8] != PINS["hparams"][a]: iss.append("hparams sha")
            if e.get("n_requests") != 60: iss.append("n_requests != 60")
            row["dose"] = e.get("dose_total_fro") or (e.get("state") or {}).get("side_delta_fro") or (e.get("state") or {}).get("n_keys")
            row["deployable_as"] = e.get("deployable_as")
            row["edit_wall_s"] = e.get("wall_s")
            if a == "E2_alphaedit" and not e.get("P_preexisting"): iss.append("P rebuilt in-run")
            if a in ("E3_wise", "E4_grace") and not e.get("original_weight_sha256"): iss.append("original weight sha missing")
        if a in OURS:
            wr = load(os.path.join(O, a, "fp652_%s_receipt.json" % a))
            if wr is None: iss.append("writer receipt missing")
            else:
                if wr.get("writer_sha256", "")[:16] != PINS["writer"]: iss.append("writer sha")
                row["steps"] = "%s/%s" % ((wr.get("realized_dose") or {}).get("accepted_steps"), (wr.get("realized_dose") or {}).get("attempted_steps"))
                pj = wr.get("projection") or {}
                row["projection"] = bool(pj)
                if a == "P_coding" and (pj.get("bases_sha256") or "")[:16] != PINS["B0"]: iss.append("bases != B0")
                if a == "U_vex" and pj: iss.append("U_vex projected")
                row["dose"] = None   # filled from delta_norm.json below
            if r.get("manifest_sha256") != r.get("model_safetensors_sha256"): iss.append("arrived != manifest")
            dn = load(os.path.join(O, a, "delta_norm.json"))
            if dn: row["dose"] = dn.get("window_fro_total"); row["outside_window_fro"] = dn.get("outside_window_fro")
        if a in FT:
            fr = load(os.path.join(O, a, "fp678_train_%s_receipt.json" % a.split("_")[1]))   # v0.3: renamed (fp678_ft_lora == fp678_FT_lora on a case-insensitive filesystem)
            if fr is None: iss.append("ft receipt missing")
            else:
                rc = fr.get("recipe", {})
                if fr.get("n_lessons") != 60 or rc.get("epochs") != 3 or rc.get("effective_batch") != 4: iss.append("ft recipe != 60 lessons / 3 epochs / batch 4")
                if a == "FT_lora" and (rc.get("lora") or {}).get("r") != 16: iss.append("lora r != 16")
                if a == "FT_full" and rc.get("lora"): iss.append("full arm has lora")
                if not (fr.get("export") or {}).get("reload_verified"): iss.append("reload not verified")
                if fr.get("family_sha256", "")[:16] != PINS["family"]: iss.append("ft family sha")
                row["steps"] = "%s" % (fr.get("realized_dose") or {}).get("optimizer_steps"); row["loss_tokens"] = (fr.get("realized_dose") or {}).get("loss_bearing_tokens"); row["epoch_loss"] = (fr.get("realized_dose") or {}).get("epoch_mean_loss")
            if r.get("manifest_sha256") != r.get("model_safetensors_sha256"): iss.append("arrived != manifest")
            dn = load(os.path.join(O, a, "delta_norm.json"))
            if dn: row["dose"] = dn.get("window_fro_total"); row["outside_window_fro"] = dn.get("outside_window_fro")
        if a != "base" and rd.get("bill") is None: iss.append("no bill")
        S[a] = row
        issues[a] = iss
    base = S.get("base")
    stacks = {json.dumps(v["stack"], sort_keys=True) for v in S.values() if v}
    G = {"arms_present": [a for a in ARMS if S.get(a)], "one_stack": len(stacks) == 1, "stack": (json.loads(next(iter(stacks))) if len(stacks) == 1 else sorted(stacks)),
         "port_check_pass": bool(((load(os.path.join(O, "base", "fp678_base_receipt.json")) or {}).get("port_check") or {}).get("cap_rows_equal")) and
                            bool(((load(os.path.join(O, "base", "fp678_base_receipt.json")) or {}).get("port_check") or {}).get("panel_items_equal")),
         "issues": {a: v for a, v in issues.items() if v}, "scorer_sha256": sha(os.path.abspath(__file__))}
    G["G0"] = G["one_stack"] and G["port_check_pass"] and not G["issues"] and base is not None and base.get("plus") is not None
    plus_min = (base["plus"] - PLUS_DELTA) if base and base.get("plus") is not None else None

    def screens(a):
        r = S.get(a)
        if not r or r["teach"] is None or r["plus"] is None or r["bill"] is None: return None
        return {"teach": r["teach"] >= TEACH_MIN, "plus": r["plus"] >= plus_min, "bill": r["bill"] <= BILL_MAX and r["crossings"] <= CROSS_MAX}
    SC = {a: screens(a) for a in ARMS if a != "base"}
    allp = {a: (all(v.values()) if v else None) for a, v in SC.items()}
    complete = all(S.get(a) for a in ARMS) and plus_min is not None

    def v(cond):
        return None if (not complete or cond is None) else bool(cond)
    g = lambda a, k: (S.get(a) or {}).get(k)
    P = {
        "P1_no_public_editor_passes_all_three": {"credence": 0.8, "hold": v(complete and not any(allp[a] for a in PUBLIC)), "detail": {a: allp[a] for a in PUBLIC}},
        "P2_alphaedit_teach_below_20": {"credence": 0.7, "hold": v(g("E2_alphaedit", "teach") is not None and g("E2_alphaedit", "teach") < 20), "teach": g("E2_alphaedit", "teach")},
        "P3_wise_teach_ge_30_heldout_le_10": {"credence": 0.5, "hold": v(g("E3_wise", "teach") is not None and g("E3_wise", "teach") >= 30 and g("E3_wise", "heldout") <= 10), "teach": g("E3_wise", "teach"), "heldout": g("E3_wise", "heldout")},
        "P4_grace_teach_ge_30_heldout_le_5": {"credence": 0.6, "hold": v(g("E4_grace", "teach") is not None and g("E4_grace", "teach") >= 30 and g("E4_grace", "heldout") <= 5), "teach": g("E4_grace", "teach"), "heldout": g("E4_grace", "heldout")},
        "P5_side_memory_bills_zero_crossings": {"credence": 0.7, "hold": v(g("E3_wise", "crossings") is not None and g("E3_wise", "crossings") == 0 and g("E4_grace", "crossings") == 0), "wise": g("E3_wise", "crossings"), "grace": g("E4_grace", "crossings")},
        "P6_alphaedit_plus_below_screen": {"credence": 0.6, "hold": v(g("E2_alphaedit", "plus") is not None and plus_min is not None and g("E2_alphaedit", "plus") < plus_min), "plus": g("E2_alphaedit", "plus"), "plus_min": plus_min},
        "P7_our_P_coding_passes_all_three_same_day": {"credence": 0.6, "hold": v(allp.get("P_coding")), "screens": SC.get("P_coding")},
        "P9_lora_installs_but_fails_plus_or_bill": {"credence": 0.7, "hold": v(g("FT_lora", "teach") is not None and SC.get("FT_lora") is not None and g("FT_lora", "teach") >= TEACH_MIN and not (SC["FT_lora"]["plus"] and SC["FT_lora"]["bill"])), "teach": g("FT_lora", "teach"), "screens": SC.get("FT_lora")},
        "P10_full_ft_fails_plus_or_bill": {"credence": 0.7, "hold": v(SC.get("FT_full") is not None and not (SC["FT_full"]["plus"] and SC["FT_full"]["bill"])), "teach": g("FT_full", "teach"), "screens": SC.get("FT_full")},
        "P11_no_plain_baseline_passes_all_three": {"credence": 0.7, "hold": v(complete and not any(allp[a] for a in FT)), "detail": {a: allp.get(a) for a in FT}},
        "P8_our_P_coding_teach_ge_every_public_editor_within_plus_screen": {"credence": 0.7,
            "hold": v(g("P_coding", "teach") is not None and all(g(a, "teach") is not None and (g("P_coding", "teach") >= g(a, "teach") or not (SC[a] or {}).get("plus")) for a in PUBLIC)),
            "teach": {a: g(a, "teach") for a in ["P_coding"] + PUBLIC}},
    }
    out = {"kind": "fp678_scores", "gates": G, "arms": S, "screens": {"teach_min": TEACH_MIN, "plus_min": plus_min, "bill_max": BILL_MAX, "cross_max": CROSS_MAX, "per_arm": SC, "all_three": allp},
           "predictions": P, "pins": PINS, "complete": complete}
    print("GATES G0 %s one_stack %s port_check %s issues %s" % (G["G0"], G["one_stack"], G["port_check_pass"], G["issues"]))
    print("%-13s %5s %7s %5s %7s %5s %-18s %s" % ("arm", "teach", "heldout", "plus", "bill", "cross", "screens", "dose"))
    for a in ARMS:
        r = S.get(a)
        if not r:
            print("%-13s MISSING" % a); continue
        sc = SC.get(a)
        print("%-13s %5s %7s %5s %7s %5s %-18s %s" % (a, r["teach"], r["heldout"], r["plus"], ("%.2f" % r["bill"]) if isinstance(r["bill"], (int, float)) else r["bill"], r["crossings"],
                                                        ("T%s P%s B%s" % tuple(int(bool(sc[k])) for k in ("teach", "plus", "bill"))) if sc else "-", r.get("dose")))
    for k, p in P.items():
        print("%-62s credence %.1f -> %s" % (k, p["credence"], {True: "HOLD", False: "MISS", None: "UNSCORED"}[p["hold"]]))
    if not dry:
        op = os.path.join(O, "FP678_scores.json")
        json.dump(out, open(op, "w"), indent=1, sort_keys=True)
        print("wrote", op, sha(op)[:16])
    sys.exit(0 if G["G0"] or dry else 1)


if __name__ == "__main__":
    main()
