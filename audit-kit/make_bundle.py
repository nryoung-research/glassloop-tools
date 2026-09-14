#!/usr/bin/env python3
"""Builds the evidence bundle for verify.py from the pilot out/ trees (copies files; never modifies them). Run from anywhere.
Bundle layout: evidence/{preregs, scorers, fp678, fp682b, fp682c, fp688, families, panel}/..., evidence/bundle.json, evidence/SHA256SUMS.
Arms included: FP-678 (base, E2, E3, E4, U_vex, P_coding, FT_lora, FT_full; 5090 stack), FP-682b (pq2, pq3; box-2 stack, base = the box-2 3B read),
FP-682c (px1, px2, px3; 5090 stack), FP-688 (the held-out-loss receipts)."""
import glob
import hashlib
import json
import os
import shutil
import time

ROOT = r"C:\Users\nryou\OneDrive - luxkbgallery.com\AI research"
FP616 = os.path.join(ROOT, "flagship", "fp616")
PILOT = os.path.join(FP616, "tools", "pilot")
OUT = os.path.join(PILOT, "out")
CLIMBER = os.path.join(ROOT, "flagship", "climber")
ANCHOR = os.path.join(CLIMBER, "out", "anchor")
GLASS = os.path.join(ROOT, "flagship", "glassloop")
DRAFTS = os.path.join(ROOT, "drafts")
KIT = os.path.dirname(os.path.abspath(__file__))
EV = os.path.join(KIT, "evidence")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load(p):
    return json.load(open(p, encoding="utf-8"))


HERE = os.path.dirname(os.path.abspath(__file__))


def put(src, rel):
    dst = os.path.join(EV, rel)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.exists(src):
        print("  MISSING", src)
        return None
    shutil.copyfile(src, dst)
    return rel.replace("\\", "/")


def main():
    for attempt in range(6):   # OneDrive / indexer handles can hold a directory open for a moment; retry rather than leave a half-deleted tree
        if not os.path.exists(EV):
            break
        try:
            shutil.rmtree(EV)
        except PermissionError:
            time.sleep(5)
    if os.path.exists(EV):
        raise SystemExit("EVIDENCE-DIR-LOCKED %s" % EV)
    os.makedirs(EV)
    B = {"title": "Auditable skill installation: head-to-head (FP-678) and the two-family artifacts (FP-682b, FP-682c) with the fourth screen (FP-688)", "generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "preregs": [], "arms": [], "tables": {}}
    # preregs + scorers + tools
    for f, sha16 in [("FP678-PUBLIC-EDITORS-HEAD-TO-HEAD-PREREG-v0-2026-09-11.md", None), ("FP682B-PANEL-SPAN-LADDER-3B-PREREG-v0-2026-09-12.md", None), ("FP682C-CLEAN-PANEL-SPAN-LADDER-3B-PREREG-v0-2026-09-12.md", None), ("FP688-GENERAL-LOSS-FOURTH-SCREEN-PREREG-v0-2026-09-12.md", None), ("FP700-OUT-OF-SPAN-PANEL-PREREG-v0-2026-09-13.md", None)]:
        rel = put(os.path.join(DRAFTS, f), "preregs/" + f)
        try:
            import subprocess
            g = subprocess.run(["git", "log", "--diff-filter=A", "--format=%H %cI", "--", os.path.join("drafts", f)], cwd=ROOT, capture_output=True, text=True).stdout.strip().splitlines()
            first = g[-1].split() if g else [None, None]
        except Exception:
            first = [None, None]
        B["preregs"].append({"file": rel, "sha16": sha(os.path.join(EV, rel))[:16], "git_first_commit": first[0], "git_first_commit_date": first[1],
                             "note": "sha of the prereg file as bundled (the FROZEN lines inside quote the scorer / chain / runner shas); git_first_commit = the commit that added the file in the public repository (the ordering evidence, checkable against the repository once released)"})
    for f in glob.glob(os.path.join(PILOT, "historic", "*.py")):
        put(f, "scorers/historic/" + os.path.basename(f))   # the exact runner / scorer versions each chain log names (recovered from git by sha)
    put(os.path.join(FP616, "tools", "supervisor", "fp_supervisor.py"), "scorers/fp_supervisor.py")
    for f in ["fp678_h2h.py", "fp678_h2h_v2.py", "fp678_score.py", "fp678_score_v2.py", "fp678_ft.py", "fp682b_score.py", "fp682c_score.py", "fp688_lmloss.py", "price_state.py", "panel_read_hf.py", "fp656_delta_norm.py", "fp658_coding_bases.py", "fp667_family_writer.py"]:
        put(os.path.join(PILOT, f), "scorers/" + f)
    for f in glob.glob(os.path.join(PILOT, "decision_guard", "*.py")):
        put(f, "scorers/decision_guard/" + os.path.basename(f))
    for f in ["fp678_chain.sh", "fp678b_chain.sh", "fp678c_chain.sh", "fp678d_chain.sh", "fp678_chain_v2.sh", "fp682b_box2_chain.sh", "fp682c_chain.sh"]:
        put(os.path.join("C:/fp43", f), "chains/" + f)
    for f in glob.glob(os.path.join(PILOT, "env_freeze", "*.txt")):
        put(f, "env/" + os.path.basename(f))
    for f in ["cap_eval.py", "climb_harness.py", "score_anchor_wsl.sh", "vex_tasks.py", "vex_teach.py", "vexlib.py", "vex_family.json", "quill_tasks.py", "quill_adapter.py", "quilllib.py", "quill_family.json", "fp677_manifest.json", "fp675_vex_pairs.json", "fp677_quill_pairs.json", "fp682b_panel_pairs.json", "fp682b_manifest.json"]:
        put(os.path.join(CLIMBER, f), "families/" + f)
    put(os.path.join("C:/fp43", "score_anchor_wsl.sh"), "families/score_anchor_wsl.sh")
    for f in ["pilot_teach.py", "paired_retention.py"]:
        put(os.path.join(CLIMBER, f), "families/" + f)
    put(os.path.join(GLASS, "fp467_panel_v2.json"), "panel/fp467_panel_v2.json")
    put(os.path.join(GLASS, "fp454_bank.json"), "panel/fp454_bank.json")
    put(os.path.join(GLASS, "fp700_panel_oos.json"), "panel/fp700_panel_oos.json")

    def anchor_files(tag):
        d = {}
        for suf, key in [("samples.jsonl", "samples"), ("receipt.json", "gen_receipt"), ("scoring_receipt_WSL.json", "scoring_receipt"), ("samples_eval_results_WSL.json", "eval_results"), ("runtime_sidecar.json", "sidecar")]:
            p = os.path.join(ANCHOR, "climb_%s_%s" % (tag, suf))
            if os.path.exists(p):
                d[key] = put(p, "anchors/climb_%s_%s" % (tag, suf))
        return d

    # ---- FP-678 (5090 stack): v1 = the first run (code shas only); v2 = the code-storing rerun (same machine, same stack, every teach row regradable) ----
    def build_fp678(dname, pfx, title):
        f678 = os.path.join(OUT, dname)
        base_panel = put(os.path.join(f678, "base", "L2_%s_base_bf16_cfgA_panel.json" % pfx), "%s/base/L2_%s_base_bf16_cfgA_panel.json" % (dname, pfx))
        scores678 = load(os.path.join(f678, "FP678_scores.json"))
        put(os.path.join(f678, "FP678_scores.json"), "%s/FP678_scores.json" % dname)
        put(os.path.join(f678, "chain.log"), "%s/chain.log" % dname)
        arms = ["base", "E2_alphaedit", "E3_wise", "E4_grace", "U_vex", "P_coding", "FT_lora", "FT_full"]
        for a in arms:
            tag = "%s_%s_bf16_cfgA" % (pfx, a)
            rname = ("fp678_arm_%s_receipt.json" % a) if a.startswith("FT_") else ("fp678_%s_receipt.json" % a)
            rec = load(os.path.join(f678, a, rname))
            arm = {"name": "%s/%s" % (dname, a), "experiment": "FP-678" if dname == "fp678" else "FP-678 v2 (code-storing rerun)", "stack": rec["stack"], "receipt": put(os.path.join(f678, a, rname), "%s/%s/%s" % (dname, a, rname))}
            row = scores678["arms"].get(a) or {}
            arm.update({"teach": row.get("teach"), "heldout": row.get("heldout"), "plus": row.get("plus")})
            if a != "base":
                arm.update({"bill": row.get("bill"), "crossings": row.get("crossings"), "panel": put(os.path.join(f678, a, "L2_%s_panel.json" % tag), "%s/%s/L2_%s_panel.json" % (dname, a, tag)), "base_panel": base_panel,
                            "price": put(os.path.join(f678, a, "L2_%s_price_vs_base.json" % tag), "%s/%s/L2_%s_price_vs_base.json" % (dname, a, tag))})
            arm["cap_reports"] = [{"family": "vex", "split": sp, "file": put(os.path.join(f678, a, "cap_vex_%s_%s_report.json" % (tag, sp)), "%s/%s/cap_vex_%s_%s_report.json" % (dname, a, tag, sp))} for sp in ("teach", "heldout")]
            af = anchor_files(tag)
            if "samples" in af:
                arm.update({"samples": af["samples"], "samples_sha256": sha(os.path.join(EV, af["samples"])), "anchor_files": af})
            for extra in ["fp652_%s_receipt.json" % a, "delta_norm.json", "fp678_train_%s_receipt.json" % a.replace("FT_", "")]:
                p = os.path.join(f678, a, extra)
                if os.path.exists(p):
                    put(p, "%s/%s/%s" % (dname, a, extra))
            B["arms"].append(arm)
        B["tables"][title] = ["%s/%s" % (dname, a) for a in arms]
        return base_panel

    base_panel = build_fp678("fp678", "fp678", "FP-678 head-to-head, first run 2026-09-11 (Qwen2.5-3B, 5090 stack; code shas only; screens teach>=54, plus>=106, bill<=53.12 & <=2 crossings)")
    build_fp678("fp678v2", "fp678v2", "FP-678 v2: the same eight arms rerun 2026-09-12 with the generated code stored (same machine and stack; every teach row regraded by step 4)")

    # ---- FP-682c (5090 stack, shares the FP-678 base) ----
    f682c = os.path.join(OUT, "fp682c")
    s682c = load(os.path.join(f682c, "FP682c_scores.json"))
    put(os.path.join(f682c, "FP682c_scores.json"), "fp682c/FP682c_scores.json"); put(os.path.join(f682c, "chain.log"), "fp682c/chain.log")
    put(os.path.join(f682c, "bases", "fp658_bases_receipt.json"), "fp682c/bases/fp658_bases_receipt.json")
    for a in ["px1", "px2", "px3"]:
        tag = "fp682c_%s_bf16_cfgA" % a
        row = s682c["arms"][a]
        arm = {"name": "fp682c/" + a, "experiment": "FP-682c", "stack": row["stack"], "receipt": put(os.path.join(f682c, a, "fp682c_arm_%s_receipt.json" % a), "fp682c/%s/fp682c_arm_%s_receipt.json" % (a, a)),
               "teach": row["teach_vex"], "heldout": row["heldout_vex"], "teach_quill": row["teach_quill"], "heldout_quill": row["heldout_quill"], "plus": row["plus"], "bill": row["bill"], "crossings": row["crossings"],
               "panel": put(os.path.join(f682c, a, "L2_%s_panel.json" % tag), "fp682c/%s/L2_%s_panel.json" % (a, tag)), "base_panel": base_panel,
               "price": put(os.path.join(f682c, a, "L2_%s_price_vs_base.json" % tag), "fp682c/%s/L2_%s_price_vs_base.json" % (a, tag)),
               "cap_reports": [{"family": fam, "split": sp, "file": put(os.path.join(f682c, a, "cap_%s_%s_%s_report.json" % (fam, tag, sp)), "fp682c/%s/cap_%s_%s_%s_report.json" % (a, fam, tag, sp))} for fam in ("vex", "quill") for sp in ("teach", "heldout")]}
        af = anchor_files(tag)
        if "samples" in af:
            arm.update({"samples": af["samples"], "samples_sha256": sha(os.path.join(EV, af["samples"])), "anchor_files": af})
        for extra in ["fp652_%s_receipt.json" % a, "delta_norm.json"]:
            put(os.path.join(f682c, a, extra), "fp682c/%s/%s" % (a, extra))
        B["arms"].append(arm)
    # px3 re-read with the code-storing runner (2026-09-12, tag fp682c_px3v2): same export, same base panel; teach/held-out/bill taken from its own reports
    a, tag = "px3v2", "fp682c_px3v2_bf16_cfgA"
    if os.path.exists(os.path.join(f682c, a, "fp678_ours_receipt.json")):
        rec = load(os.path.join(f682c, a, "fp678_ours_receipt.json"))
        caps = {(fam, sp): load(os.path.join(f682c, a, "cap_%s_%s_%s_report.json" % (fam, tag, sp))) for fam in ("vex", "quill") for sp in ("teach", "heldout")}
        price = load(os.path.join(f682c, a, "L2_%s_price_vs_base.json" % tag))
        arm = {"name": "fp682c/" + a, "experiment": "FP-682c (px3 re-read, code stored)", "stack": rec["stack"], "receipt": put(os.path.join(f682c, a, "fp678_ours_receipt.json"), "fp682c/%s/fp678_ours_receipt.json" % a),
               "teach": caps[("vex", "teach")]["n_passed"], "heldout": caps[("vex", "heldout")]["n_passed"], "teach_quill": caps[("quill", "teach")]["n_passed"], "heldout_quill": caps[("quill", "heldout")]["n_passed"],
               "bill": price["damage_bill_nats"], "crossings": price["n_sign_crossings"],
               "panel": put(os.path.join(f682c, a, "L2_%s_panel.json" % tag), "fp682c/%s/L2_%s_panel.json" % (a, tag)), "base_panel": base_panel,
               "price": put(os.path.join(f682c, a, "L2_%s_price_vs_base.json" % tag), "fp682c/%s/L2_%s_price_vs_base.json" % (a, tag)),
               "cap_reports": [{"family": fam, "split": sp, "file": put(os.path.join(f682c, a, "cap_%s_%s_%s_report.json" % (fam, tag, sp)), "fp682c/%s/cap_%s_%s_%s_report.json" % (a, fam, tag, sp))} for fam in ("vex", "quill") for sp in ("teach", "heldout")]}
        af = anchor_files(tag)
        if "samples" in af:
            arm.update({"samples": af["samples"], "samples_sha256": sha(os.path.join(EV, af["samples"])), "anchor_files": af, "plus": load(os.path.join(EV, af["scoring_receipt"]))["plus_pass"]})
        B["arms"].append(arm)
    B["tables"]["FP-682c clean ladder + panel span (S1 -> px1 -> px2 -> px3; 5090 stack; px3v2 = px3 re-read with code stored)"] = ["fp682c/" + a for a in ["px1", "px2", "px3", "px3v2"]]

    # ---- FP-682b (box-2 stack; base = the box-2 3B read L0_base_bf16_box2) ----
    bx = os.path.join(OUT, "box2")
    base2 = put(os.path.join(bx, "L0_base_bf16_box2.json"), "fp682b/L0_base_bf16_box2.json")
    afb = anchor_files("base_bf16_cfgA_box2")
    armb = {"name": "fp682b/base", "experiment": "FP-682b", "stack": {"box": "box2 (DGX Spark GB10)", "torch": "2.13.0+cu130", "transformers": "5.13.1"}, "panel": base2,
            "note": "the box-2 base read: the panel every FP-682b bill is measured against and the HumanEval+ base (117) the FP-682b plus screen refers to"}
    if "samples" in afb:
        armb.update({"samples": afb["samples"], "samples_sha256": sha(os.path.join(EV, afb["samples"])), "anchor_files": afb, "plus": load(os.path.join(EV, afb["scoring_receipt"]))["plus_pass"]})
    B["arms"].append(armb)
    s682b = load(os.path.join(OUT, "FP682b_scores.json"))
    put(os.path.join(OUT, "FP682b_scores.json"), "fp682b/FP682b_scores.json"); put(os.path.join(bx, "fp682b", "run.log"), "fp682b/run.log")
    put(os.path.join(bx, "fp682b", "bases", "fp658_bases_receipt.json"), "fp682b/bases/fp658_bases_receipt.json")
    for a in ["pq2", "pq3"]:
        row = s682b["arms"][a]
        tag = "%s_bf16_cfgA_box2" % a
        arm = {"name": "fp682b/" + a, "experiment": "FP-682b", "stack": {"box": "box2 (DGX Spark GB10)", "torch": "2.13.0+cu130", "transformers": "5.13.1"},
               "receipt": put(os.path.join(bx, "fp682b", a, "fp652_%s_receipt.json" % a), "fp682b/%s/fp652_%s_receipt.json" % (a, a)),
               "teach": row.get("teach"), "heldout": row.get("heldout_vex"), "teach_quill": row.get("teach_quill"), "heldout_quill": row.get("heldout_quill"), "plus": row.get("anchor_box2"), "bill": row.get("bill"), "crossings": row.get("crossings"),
               "panel": put(os.path.join(bx, "L2_%s_bf16_panel.json" % a), "fp682b/%s/L2_%s_bf16_panel.json" % (a, a)), "base_panel": base2,
               "price": put(os.path.join(bx, "L2_%s_price_vs_base_box2.json" % a), "fp682b/%s/L2_%s_price_vs_base_box2.json" % (a, a)),
               "cap_reports": [{"family": fam, "split": sp, "file": put(os.path.join(bx, "cap_%s_%s_bf16_%s_report.json" % (fam, a, sp)), "fp682b/%s/cap_%s_%s_bf16_%s_report.json" % (a, fam, a, sp))} for fam in ("vex", "quill") for sp in ("teach", "heldout")]}
        af = anchor_files(tag)
        if "samples" in af:
            arm.update({"samples": af["samples"], "samples_sha256": sha(os.path.join(EV, af["samples"])), "anchor_files": af})
        for extra in ["delta_norm.json", "MANIFEST-SHA256.json"]:
            put(os.path.join(bx, "fp682b", a, extra), "fp682b/%s/%s" % (a, extra))
        B["arms"].append(arm)
    B["tables"]["FP-682b panel-span ladder (rq -> pq2 -> pq3; box-2 stack; screens teach>=54, plus>=108, bill<=53.12 & <=2 crossings)"] = ["fp682b/" + a for a in ["base", "pq2", "pq3"]]

    # ---- FP-700 (5090 stack): the out-of-span panel (641 fresh bank facts, overlap 0 with the protected panel) ----
    f700 = os.path.join(OUT, "fp700")
    put(os.path.join(f700, "chain.log"), "fp700/chain.log"); put(os.path.join(f700, "chain.sh"), "chains/fp700_chain.sh")
    base_oos = put(os.path.join(f700, "L0_base_oos_bf16.json"), "fp700/L0_base_oos_bf16.json")
    B["arms"].append({"name": "fp700/base_oos", "experiment": "FP-700", "stack": {"gpu": "NVIDIA GeForce RTX 5090", "torch": "2.7.1+cu128"}, "panel": None, "note": "the base read on the out-of-span panel; every FP-700 bill is measured against it"})
    B["arms"][-1].pop("panel")
    for a, src, bp, note in [("S1_oos", "L2_S1_oos_bf16.json", base_oos, "S1 (the lineage root, coding-span-projected vex installer) on the out-of-span panel"), ("px3_oos", "L2_px3_oos_bf16.json", base_oos, "px3 (the two-family artifact) on the out-of-span panel"), ("S1_in", "L2_S1_in_bf16.json", base_panel, "S1 on the protected panel (like-for-like pair for S1_oos)")]:
        pf = "L2_%s_price_oos.json" % a.split("_")[0] if a.endswith("_oos") else "L2_S1_price_in.json"
        pr = load(os.path.join(f700, pf))
        B["arms"].append({"name": "fp700/" + a, "experiment": "FP-700", "stack": {"gpu": "NVIDIA GeForce RTX 5090", "torch": "2.7.1+cu128"}, "note": note,
                          "panel": put(os.path.join(f700, src), "fp700/" + src), "base_panel": bp, "price": put(os.path.join(f700, pf), "fp700/" + pf),
                          "bill": pr["damage_bill_nats"], "crossings": pr["n_sign_crossings"], "gain": pr.get("gain_sum_nats")})
    B["tables"]["FP-700 out-of-span panel (641 fresh bank facts never used to build any span; same base, same stack; no screen, a transfer test of the bill)"] = ["fp700/" + a for a in ["S1_in", "S1_oos", "px3_oos"]]
        # ---- FP-688 ----
    for p in glob.glob(os.path.join(OUT, "fp688", "*.json")) + glob.glob(os.path.join(OUT, "fp688", "box2", "*.json")):
        put(p, "fp688/" + os.path.relpath(p, os.path.join(OUT, "fp688")))
    B["fp688"] = {"note": "held-out pile-10k NLL receipts; deltas in FP688_scores.json; the 200 documents are re-derivable from the hub dataset snapshot named in each receipt"}
    S688 = load(os.path.join(OUT, "fp688", "FP688_scores.json"))
    recs = {}
    for p in glob.glob(os.path.join(EV, "fp688", "*_lmloss.json")) + glob.glob(os.path.join(EV, "fp688", "box2", "*_lmloss.json")):
        recs[sha(p)[:16]] = os.path.relpath(p, EV).replace("\\", "/")
    rows = {}
    for name, row in S688["rows"].items():
        rows[name] = {"receipt": recs.get(row["receipt"]), "base_receipt": recs.get(row["base_receipt"]), "dnll": row["delta"], "box": row["box"]}
    B["lmloss"] = {"scores": "fp688/FP688_scores.json", "rows": rows, "receipts": recs}

    for arm in B["arms"]:
        if "price" in arm and os.path.exists(os.path.join(EV, arm["price"])):
            pr = load(os.path.join(EV, arm["price"]))
            if "gain_sum_nats" in pr:
                arm["gain"] = pr["gain_sum_nats"]
    runs = []
    for rel in sorted(glob.glob(os.path.join(EV, "**", "*.log"), recursive=True)):
        for line in open(rel, encoding="utf-8", errors="replace"):
            if "CHAIN START" in line:
                runs.append({"log": os.path.relpath(rel, EV).replace("\\", "/"), "line": line.strip()[:400]})
    B["runs"] = runs
    B["code_bindings"] = [{"first": "fp678/%s" % a, "second": "fp678v2/%s" % a, "expect_identical": a in ("base", "U_vex", "P_coding", "FT_lora", "FT_full")} for a in ["base", "E2_alphaedit", "E3_wise", "E4_grace", "U_vex", "P_coding", "FT_lora", "FT_full"]] + [{"first": "fp682c/px3", "second": "fp682c/px3v2", "expect_identical": True}]
    B["pins_note"] = ("Every chain log's CHAIN START line names the runner / scorer / writer / chain shas that produced its evidence; the FP-678 first run "
                      "used runner 0f32a0eb and scorers 24883964 (v0), a8213689 (v0.1), 74e5513c (v0.3), all shipped under scorers/historic/; the FP-678 "
                      "pre-registration's frozen line names the v0 pair and its amendments v0.1-v0.3 (in the same file) name the later scorers. scorers/fp678_h2h.py "
                      "is the FP-682c-era runner (e570bce2, families extension) and scorers/fp678_h2h_v2.py the code-storing runner (de956806) used by the v2 rerun and px3v2.")
    nfiles = sum(1 for dp, _, fs in os.walk(EV) for f in fs if f not in ("SHA256SUMS", "bundle.json")) + 1   # + bundle.json itself
    B["inventory"] = {"files": nfiles, "arms": len(B["arms"]), "preregs": len(B["preregs"]), "note": "verify.py step 1 refuses a bundle whose SHA256SUMS, arm list or prereg list is shorter than this (G-846: empty inventories must not pass)"}
    json.dump(B, open(os.path.join(EV, "bundle.json"), "w"), indent=1, sort_keys=True)
    lines = []
    for dp, dns, fns in os.walk(EV):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            if fn in ("SHA256SUMS",):
                continue
            p = os.path.join(dp, fn)
            lines.append("%s  %s" % (sha(p), os.path.relpath(p, EV).replace("\\", "/")))
    open(os.path.join(EV, "SHA256SUMS"), "w", encoding="utf-8").write("\n".join(sorted(lines)) + "\n")
    total = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(EV) for f in fs)
    print("bundle: %d files, %.1f MB, %d arms" % (len(lines), total / 1e6, len(B["arms"])))


if __name__ == "__main__":
    main()
