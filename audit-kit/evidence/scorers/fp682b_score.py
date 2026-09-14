"""FP-682b scorer (panel-span projection ladder at 3B from rq under VEX60; derived from fp679b_score.py)
Original header: FP-679b scorer (the VEX60-only pool at 3B: the missing cell between MIX_VC (rq2/rq3) and no pool (u2); matches the FP-681 design at 1.5B)
Original header: FP-679 scorer (written BEFORE any read; sha in the prereg FROZEN line): the quill DOSE LADDER under replay.
Arms (box-2 writes from the FP-677 rq artifact = quill lessons once under the vex+coding pool; box-2 reads, same stack as the base read):
rq2 = quill lessons again from rq, R mode, no projection, pool MIX_VC (the second pass); rq3 = quill lessons again from rq2 (the third pass);
u2 = quill lessons again from rq, U mode, NO pool (the replay-off comparator at the second pass). References: rq (C-855) quill 6 / 11, vex 58 / 54,
plus 105, lost 20, bill 48.60 / 1; S1 vex 60 / coding 112. Evidence layout (fp679_fetch.sh): out/box2/L2_<arm>_price_vs_base_box2.json,
cap_{vex,quill}_<arm>_bf16_{teach,heldout}_report.json, paired_vex_S1_to_<arm>.json, out/box2/fp679/<arm>/{fp652_<arm>_receipt.json, delta_norm.json,
MANIFEST-SHA256.json}, out/box2/fp679/run.log, climber/out/anchor/climb_<arm>_*. Binding per arm: writer dba17398; mode per arm; quill_adapter /
quill_family.json 30057895; pool MIX_VC with manifest cc95a4f2 (rq2, rq3) or no pool (u2); no projection; model_root chain (rq2, u2 from rq;
rq3 from rq2); reload; ARRIVED == manifest; same-stack; anchor bindings; ARM DONE ledger. Reports; enacts nothing."""
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "tools", "pilot"))
import fp656_score_v6b as V6  # noqa: E402
import fp658b_score_v2 as B  # noqa: E402

O = os.path.join(ROOT, "tools", "pilot", "out", "box2")
F = os.path.join(O, "fp682b")
PINS = {"writer": "dba17398c94e657d", "quill_family": "300578953d12a786", "manifest": "cc95a4f2a7673d21", "rq_export": "ed2f987fffd76d34"}
ARMS = ["pq2", "pq3"]
SPEC = {"pq2": {"mode": "R", "pool": "VEX60", "root": "/rq/A_written_bf16"},
        "pq3": {"mode": "R", "pool": "VEX60", "root": "/pq2/A_written_bf16"}}
REF = {"vq2": {"teach_quill": 54, "teach_vex": 60, "plus": 109, "bill": 77.73, "crossings": 1}, "vq3": {"teach_quill": 60, "teach_vex": 60, "plus": 104, "bill": 86.43, "crossings": 1}}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def jload(p):
    return V6.load(p) if os.path.exists(p) else None


def arm(n, runlog):
    r = {"arm": n, "issues": []}
    sp = SPEC[n]
    try:
        B.common(n, r)
        d8 = os.path.join(F, n)
        rp = os.path.join(d8, "fp652_%s_receipt.json" % n)
        if os.path.exists(rp):
            d = V6.load(rp)
            pj = d.get("projection") or {}
            rd = d.get("realized_dose") or {}
            pool = d.get("pool_replay") or {}
            r.update(writer_sha=(d.get("writer_sha256") or "")[:16], variant=d.get("writer_variant"), mode=d.get("mode"), module=d.get("family_module"), family_json=d.get("family_json"), family_sha=(d.get("family_sha256") or "")[:16],
                     accepted=rd.get("accepted_steps"), attempted=rd.get("attempted_steps"), leg_tok=rd.get("leg_tok_used"), replay_tokens=rd.get("replay_loss_bearing_tokens"), pool_key=pool.get("key"), pool_manifest=(pool.get("manifest_sha256") or "")[:16],
                     pool_n=pool.get("n"), model_root=d.get("model_root"), reload_verified=d.get("reload_verified"), model_sha_manifest=B.manifest_model_sha(d), runtime=d.get("runtime"),
                     bills=[(b.get("task_id"), b.get("loss_first"), b.get("loss_last")) for b in (d.get("bills") or [])])
            if r["writer_sha"] != PINS["writer"]: r["issues"].append("writer sha != pin")
            if r["mode"] != sp["mode"]: r["issues"].append("mode != %s" % sp["mode"])
            if r["module"] != "quill_adapter" or r["family_json"] != "quill_family.json": r["issues"].append("family module/json")
            if r["family_sha"] != PINS["quill_family"]: r["issues"].append("family sha")
            if sp["pool"] is None:
                if pool: r["issues"].append("pool present in U arm")
            else:
                if r["pool_key"] != sp["pool"]: r["issues"].append("replay pool key")
                if r["pool_manifest"] != PINS["manifest"][:16]: r["issues"].append("replay manifest sha")
            bases_rec = jload(os.path.join(F, "bases", "fp658_bases_receipt.json"))
            want = ((bases_rec or {}).get("outputs_sha256") or {}).get("coding") or ""   # fp658 receipt: outputs_sha256["coding"] = the span bases file
            r.update(bases_sha=(pj.get("bases_sha256") or "")[:16] if pj else None, proj_steps=pj.get("steps") if pj else None, proj_fraction=pj.get("fraction_removed") if pj else None)
            if not pj: r["issues"].append("projection absent")
            elif want and pj.get("bases_sha256") != want: r["issues"].append("bases sha != panel-span receipt")
            elif r["proj_steps"] is None or r["proj_steps"] != r["accepted"]: r["issues"].append("projected steps != accepted")
            if not str(r.get("model_root") or "").replace("\\", "/").endswith(sp["root"]): r["issues"].append("model_root != %s" % sp["root"])
            if r["reload_verified"] is not True: r["issues"].append("reload")
        else:
            r["issues"].append("no writer receipt")
        v, why = V6.norm_join(d8, r.get("model_sha_manifest"), r.get("sidecar_model_sha"))
        if v is not None:
            r["delta_norm"] = v
        else:
            r["issues"].append(why)
        al = dict(re.findall(r"ARRIVED (\w+) model\.safetensors (\w+) \(== manifest\)", runlog))
        if not (r.get("model_sha_manifest") and al.get(n, "") == r.get("model_sha_manifest")): r["issues"].append("ARRIVED line != manifest")
        for fam in ("vex", "quill"):
            for split in ("teach", "heldout"):
                q = jload(os.path.join(O, "cap_%s_%s_bf16_%s_report.json" % (fam, n, split)))
                key = ("teach_%s" % fam) if split == "teach" else ("heldout_%s" % fam)
                if q is not None:
                    r[key] = q.get("n_passed")
        if r.get("teach_quill") is None: r["issues"].append("no quill teach report")
        if "bill" in r and r.get("same_stack") is not True: r["issues"].append("stack")
        if "teach" in r and r.get("teach_valid") is not True: r["issues"].append("teach rows")
        if "anchor_box2" in r:
            if not r.get("sidecar_model_sha") or r.get("sidecar_model_sha") != r.get("model_sha_manifest"): r["issues"].append("anchor model bytes != export manifest")
            if r.get("receipt_join") is not True: r["issues"].append("eval/samples bytes != scoring receipt digests")
            if r.get("eval_valid") is not True: r["issues"].append("eval invalid (%s)" % r.get("eval_reason"))
            if r.get("eval_join") is not True: r["issues"].append("eval join")
            if r.get("plus_matches_receipt") is not True: r["issues"].append("paired plus != receipt plus_pass")
    except V6.Malformed as e:
        r["issues"].append("malformed evidence: %s" % e)
    r["bound"] = not r["issues"]
    return r


def main():
    rl = os.path.join(F, "run.log")
    txt = open(rl, encoding="utf-8", errors="replace").read() if os.path.exists(rl) else ""
    S = {n: arm(n, txt) for n in ARMS}
    G = {"arms_done": [a for a, *_ in re.findall(r"ARM (\w+) DONE teach_vex=(\S+) heldout_vex=(\S+) teach_quill=(\S+) heldout_quill=(\S+)", txt)]}
    hd = re.search(r"FP682b READS START pq2 (\w+) pq3 (\w+) bases (\w+) vex_family (\w+) quill_family (\w+) rq (\w+) queue (\w+) scorer_ready (\S+)", txt)
    G["scorer_ready_sha"] = hd.group(8) if hd else None
    G["scorer_ready_matches_this_file"] = (G["scorer_ready_sha"] is not None and sha(os.path.abspath(__file__)).startswith(G["scorer_ready_sha"][:16]))
    G["G0"] = bool(hd and hd.group(5).startswith(PINS["quill_family"]) and hd.group(6).startswith(PINS["rq_export"]))
    P = {}

    def need(names, fields):
        out = []
        if G["G0"] is not True:
            out.append("G0")
        for n in names:
            if n not in G["arms_done"]:
                out.append("ledger(%s not ARM DONE)" % n)
            for f in fields:
                if S[n].get(f) is None:
                    out.append("%s %s" % (n, f))
            if not S[n]["bound"]:
                out.append("binding(%s: %s)" % (n, "; ".join(S[n]["issues"])[:80]))
        return out

    def score(key, cond, needed):
        P[key] = ("UNSCORED (%s)" % "; ".join(needed)) if needed else ("HOLD" if cond else "MISS")

    def T(n): return S[n].get("teach", -1)
    def TQ(n): return S[n].get("teach_quill", -1)
    def PL(n): return S[n].get("anchor_box2", -10**6)
    def BL(n): return S[n].get("bill", 1e9)
    def CR(n): return S[n].get("crossings", 99)
    score("P1 the panel span holds the bill: bill(pq2) <= 60 (vq2 77.73)", BL("pq2") <= 60.0, need(["pq2"], ["bill"]))
    score("P2 the panel span does not block quill: teach_quill(pq2) >= 45 (vq2 54)", TQ("pq2") >= 45, need(["pq2"], ["teach_quill"]))
    score("P3 vex held: teach_vex(pq2) >= 55 and teach_vex(pq3) >= 55", T("pq2") >= 55 and T("pq3") >= 55, need(["pq2", "pq3"], ["teach"]))
    score("P4 coding kept: plus(pq2) >= 105 (vq2 109)", PL("pq2") >= 105, need(["pq2"], ["anchor_box2"]))
    score("P5 crossings <= 2 at both passes", CR("pq2") <= 2 and CR("pq3") <= 2, need(["pq2", "pq3"], ["bill"]))
    score("P6 the third pass lands inside ALL three screens with both families >= 54: quill(pq3) >= 54, vex(pq3) >= 54, plus(pq3) >= 108, bill(pq3) <= 53.12, crossings <= 2", TQ("pq3") >= 54 and T("pq3") >= 54 and PL("pq3") >= 108 and BL("pq3") <= 53.12 and CR("pq3") <= 2, need(["pq3"], ["teach_quill", "teach", "anchor_box2", "bill"]))
    out = {"kind": "fp682b_scores", "pins": PINS, "gates": G, "arms": S, "references": REF, "predictions": P, "scorer_sha256": sha(os.path.abspath(__file__))}
    op = os.path.join(os.path.dirname(O), "FP682b_scores.json")
    json.dump(out, open(op, "w"), indent=1, sort_keys=True)
    print("GATES", {k: G[k] for k in ("G0", "scorer_ready_matches_this_file", "arms_done")})
    for k, v in P.items():
        print(v, "|", k)
    for n in ARMS:
        r = S[n]
        print("%-3s mode %s pool %s proj %s/%s steps %s/%s leg_tok %s replay_tok %s | vex %s/%s | quill %s/%s | bill %s cross %s plus %s paired %s dnorm %s | bound %s %s" % (n, r.get("mode"), r.get("pool_key"), r.get("bases_sha"), r.get("proj_fraction"), r.get("accepted"), r.get("attempted"), r.get("leg_tok"), r.get("replay_tokens"), r.get("teach"), r.get("heldout_vex"), r.get("teach_quill"), r.get("heldout_quill"), r.get("bill"), r.get("crossings"), r.get("anchor_box2"), r.get("paired"), r.get("delta_norm"), r["bound"], ("" if r["bound"] else r["issues"])))
    print("wrote", op, sha(op)[:16])


if __name__ == "__main__":
    main()
