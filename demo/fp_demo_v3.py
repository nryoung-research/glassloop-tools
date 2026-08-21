# -*- coding: utf-8 -*-
"""WATCH IT GROW (demo v3, non-experiment): one unbroken run on
Qwen2.5-3B-Instruct fp32 (TF32 off) in which the model LEARNS LIVE.

  ACT 0  base model quizzed on 4 FreshQA headline-teach items (expect 0/4)
  ACT 1..4  each item TAUGHT LIVE via a gl_edit2m.py subprocess, chained on
            the previous act's checkpoint; re-quiz all 4 after each lesson
            (KNOWS grows) and re-measure a 40-item spot-check panel
            (DAMAGE builds, casualties named)
  ACT 5  DEMO-GRADE REPAIR, ITERATIVE TO CLOSURE: healing takes rounds;
         each round is priced too. Each pass re-teaches the current
         casualties' own true statements (fp454_bank templates, low lr),
         the panel is re-measured, new wounds join the target list, and
         the loop stops at zero (relative to the act-0 baseline) or after
         4 rounds with the honest residual. Round-by-round mini-scoreboard
         (healed / new wounds / remaining).
  ACT 6  PRESS to int8: per-channel symmetric RTN (quant_bill arithmetic,
         imported) over attention+MLP weights, EVERY leaf preserved.
  ACT 7  GROW AGAIN, ON THE FP32 MASTER: the 5th lesson is installed on
         the repaired master (act-5 state) -- the staircase's real
         architecture: the master copy grows; pressings are releases.
         (--bonus additionally ATTEMPTS growth chained from the int8
         checkpoint; rehearsal 1 showed the editor subprocess dies with a
         Windows access violation there, so the attempt is non-fatal and
         off the main flow.)
  ACT 8  VERIFIED REVERT: snapshot union of all touched leaves (attention
         included, the v2 fix), hash-exact; quiz returns to 0/5.

SCOREBOARD after every act: KNOWS x/N, DAMAGE y items (named), one line of
narration. Machine-parsable SCOREBOARD| lines + a state file at
E:\\fp558\\demo_v3_state.json holding one row per act under "rows" with an
"acts" mirror list; every save is read back and verified. Nothing here
banks; the sealed 641-item panel remains the instrument of record.

Spot-check panel (rehearsal-2 cut): 32 FRAGILE items -- the whole
0.05 < base margin < 0.6 band (22 items), topped up with the next-lowest
margins above 0.6 -- plus 8 STABLE items (margin > 1.0) as contrast,
seed 3, deterministic.

Resume/reuse: readouts (quiz + panel) are ALWAYS re-run live; TRAINING is
skipped when an act's checkpoint already exists and nothing upstream was
retrained this run (checkpoints from rehearsal 1 stay valid -- lessons do
not depend on the panel). --fresh retrains everything live: use it for the
real performance.

CLI:  --prep       run lesson prep only (gen_demo_lessons)
      --dry        parse specs + print plan, no model, no GPU
      (default)    the full live run (GPU box only)
      --seed N     lesson-prep seed (default 1)
      --fresh      retrain every act live (ignore existing checkpoints)
      --bonus      also attempt growth chained from the int8 checkpoint
"""
import argparse
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

GL = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, GL)

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUT = r"E:\fp558"
DEMO = os.path.join(OUT, "demo_v3")
CKPTS = os.path.join(DEMO, "ckpts")
LESSONS_DIR = os.path.join(DEMO, "lessons")
STATE_PATH = os.path.join(OUT, "demo_v3_state.json")  # per spec, exact path
EDITOR = os.path.join(GL, "gl_edit2m.py")
BANK_PATH = os.path.join(GL, "fp454_bank.json")
MEAS_BASE = r"E:\fp539\meas_base.json"
QB_DIR = r"C:\Users\nryou\glassloop-tools\tools\quant-bill"
PRESSED = os.path.join(CKPTS, "demo_act6_pressed.pt")

PANEL_SEED = 3
FRAG_LO, FRAG_HI, FRAG_N = 0.05, 0.6, 32   # fragile band + top-up
STAB_BAR, STAB_N = 1.0, 8                  # stable contrast members
QUIZ_TOKENS = 24
REPAIR_LR, REPAIR_STEPS = 2e-05, 200  # low-lr targeted re-teach (demo-grade)
REPAIR_MAX_ROUNDS = 4


def banner(s):
    print("\n" + "=" * 64 + "\n  %s\n" % s + "=" * 64, flush=True)


def quiz_match(answer, got):
    """Casefolded substring test, tolerant of currency/format symbols:
    '$' and ',' are stripped from BOTH sides, so '$19.99' matches '19.99'
    and '1,429' matches '1429' (rehearsal-1 fix)."""
    na = answer.casefold().replace("$", "").replace(",", "").strip()
    ng = got.casefold().replace("$", "").replace(",", "")
    return bool(na) and na in ng


# ---------------------------------------------------------------- prep/plan

def ensure_lessons(seed, regenerate=False):
    import gen_demo_lessons
    man_path = os.path.join(LESSONS_DIR, "demo_manifest.json")
    if regenerate or not os.path.exists(man_path):
        return gen_demo_lessons.generate(seed=seed, out_dir=LESSONS_DIR)
    man = json.load(open(man_path, encoding="utf-8"))
    if man.get("seed") != seed:
        print("NOTE: existing manifest was generated with seed %s "
              "(requested %s); regenerating." % (man.get("seed"), seed))
        return gen_demo_lessons.generate(seed=seed, out_dir=LESSONS_DIR)
    return man


def build_panel():
    """32 fragile + 8 stable, deterministic (seed 3).
    Fragile = the whole (FRAG_LO, FRAG_HI) base-margin band, topped up with
    the next-lowest margins above the band if it holds fewer than FRAG_N.
    Stable = seeded sample of items with base margin > STAB_BAR."""
    mb = json.load(open(MEAS_BASE, encoding="utf-8"))
    margins = mb["margins"]
    rng = random.Random(PANEL_SEED)
    band = sorted((k for k, v in margins.items() if FRAG_LO < v < FRAG_HI),
                  key=lambda k: (margins[k], k))
    if len(band) >= FRAG_N:
        frag = sorted(rng.sample(sorted(band), FRAG_N))
    else:
        extra = sorted((k for k, v in margins.items() if v >= FRAG_HI),
                       key=lambda k: (margins[k], k))
        frag = sorted(band + extra[:FRAG_N - len(band)])
    stab_pool = [k for k in sorted(margins) if margins[k] > STAB_BAR
                 and k not in set(frag)]
    stab = sorted(rng.sample(stab_pool, STAB_N))
    return frag, stab, len(margins)


def new_state(frag, stab):
    return {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": MODEL_NAME, "panel_seed": PANEL_SEED,
            "panel": {"fragile": frag, "stable": stab},
            "quiz_tokens": QUIZ_TOKENS,
            "match_rule": "casefold; '$' and ',' stripped from both sides",
            "acts": [], "rows": [], "ever_casualties": []}


def save_state(st):
    """Write, then READ BACK and verify (rehearsal-1 fix: the per-act append
    must provably persist). 'acts' is a top-level mirror of rows[].act so
    the file is self-evident to any probe."""
    st["acts"] = [r["act"] for r in st["rows"]]
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    tmp = STATE_PATH + ".tmp"
    json.dump(st, open(tmp, "w", encoding="utf-8"), indent=1)
    os.replace(tmp, STATE_PATH)
    chk = json.load(open(STATE_PATH, encoding="utf-8"))
    if chk.get("acts") != st["acts"]:
        raise SystemExit("REFUSED: state file did not persist (%s has acts "
                         "%s, expected %s)" % (STATE_PATH, chk.get("acts"),
                                               st["acts"]))
    print("  [state] %d row(s) persisted and verified -> %s"
          % (len(st["rows"]), STATE_PATH), flush=True)


def validate_spec(path):
    spec = json.load(open(path, encoding="utf-8"))
    need = ("name", "type", "items", "fact_template", "probe_template",
            "fact_texts", "holdout_probes", "slack_frac", "steps", "lr",
            "layers", "seed")
    missing = [k for k in need if k not in spec]
    if missing:
        raise SystemExit("REFUSED: spec %s missing %s" % (path, missing))
    return spec


def build_repair_spec(casualty_ids, bank):
    """Demo-grade repair: re-teach the casualties' own true statements
    (fp454_bank templates instantiated with the true answer) at low lr."""
    fact_texts, items = [], []
    for cid in casualty_ids:
        it = bank[cid]
        a = it["answer"]
        for tpl in it["templates"]:
            t = tpl.replace("{answer}", a)
            if t not in fact_texts:
                fact_texts.append(t)
        pre = None
        for tpl in it["templates"]:
            head = tpl.split("{answer}")[0].strip()
            if head:
                pre = head
                break
        if pre:
            items.append([pre, a])
    if not items:
        return None
    return {"name": "DEMO_R5", "type": "knowledge_edit", "items": items,
            "fact_template": "{n} {t}.", "probe_template": "{n}",
            "fact_texts": fact_texts, "holdout_probes": [],
            "slack_frac": 0.125, "steps": REPAIR_STEPS, "lr": REPAIR_LR,
            "layers": [28, 35], "seed": 999, "task_bar": 0.85,
            "casualty_ids": sorted(casualty_ids)}  # inert; reuse check


# ------------------------------------------------------------------ dry run

def dry_run(A):
    banner("WATCH IT GROW v3 -- DRY RUN (no model, no GPU)")
    man = ensure_lessons(A.seed)
    print("\nlesson manifest (seed %d):" % man["seed"])
    for it in man["items"]:
        print("  [%s] id=%s  answer_0=%r" % (it["role"], it["id"],
                                             it["answer_0"]))
        print("       q: %s" % it["question"])
    print("\nspec check:")
    for it in man["items"]:
        spec = validate_spec(it["spec_file"])
        print("  %s: items=%d fact_texts=%d holdout=%d steps=%d lr=%g "
              "layers=%s slack=%g" % (spec["name"], len(spec["items"]),
                                      len(spec["fact_texts"]),
                                      len(spec["holdout_probes"]),
                                      spec["steps"], spec["lr"],
                                      spec["layers"], spec["slack_frac"]))
        print("       fact_text[0]: %r" % spec["fact_texts"][0])
        print("       holdout[1]:   %r -> %r" % tuple(
            spec["holdout_probes"][1]))

    frag, stab, n_full = build_panel()
    print("\nspot-check panel: %d fragile + %d stable of %d sealed-panel "
          "items, seed %d" % (len(frag), len(stab), n_full, PANEL_SEED))
    print("  fragile = whole (%.2f, %.2f) base-margin band + next-lowest "
          "top-up; stable = margin > %.1f" % (FRAG_LO, FRAG_HI, STAB_BAR))
    print("  fragile first 8: %s" % ", ".join(frag[:8]))
    print("  stable 8:        %s" % ", ".join(stab))
    bank = {i["id"] for i in
            json.load(open(BANK_PATH, encoding="utf-8"))["items"]}
    not_in_bank = [p for p in frag + stab if p not in bank]
    if not_in_bank:
        raise SystemExit("REFUSED: panel items missing from fp454_bank: %s"
                         % not_in_bank)
    print("  bank coverage: %d/%d panel items present in fp454_bank"
          % (len(frag) + len(stab), len(frag) + len(stab)))

    if not os.path.exists(EDITOR):
        raise SystemExit("REFUSED: editor not found at %s" % EDITOR)
    sys.path.insert(0, QB_DIR)
    import quant_bill  # top-level import is torch-free; CLI path is heavy
    q, st = quant_bill.quantize_tensor([[1.0, -2.0], [0.5, 4.0]], 8)
    print("\nquant_bill import OK (selftest: 2x2 int8 rmse %.2g); leaves: %s"
          % (st["rmse"], ",".join(quant_bill.DEFAULT_LEAVES)))

    print("\nquiz: greedy %d tokens; match = casefold substring with '$' "
          "and ',' stripped both sides" % QUIZ_TOKENS)
    assert quiz_match("$19.99", "The answer is 19.99. The Netflix")
    assert quiz_match("1,429", "there were 1429 tornadoes")
    assert not quiz_match("$19.99", "The answer is 19.98")
    print("  match selftest OK ('$19.99' ~ '...is 19.99...'; '1,429' ~ "
          "'...1429...')")

    print("\nact plan (editor calls stream via subprocess, chained "
          "--init-checkpoint):")
    reusable = []
    prev = "(base)"
    for k, it in enumerate(man["items"][:4], start=1):
        ck = os.path.join(CKPTS, "demo_act%d.pt" % k)
        mark = " [ckpt EXISTS -> reused unless --fresh]" \
            if os.path.exists(ck) else ""
        if os.path.exists(ck):
            reusable.append("act%d" % k)
        print("  act%d  teach %s  init=%s%s" % (k, it["id"], prev, mark))
        prev = ck
    print("  act5  ITERATIVE demo-grade repair of live casualties "
          "(re-measure each round, union prior wounds, max %d rounds; "
          "always trains live)  init=%s" % (REPAIR_MAX_ROUNDS, prev))
    print("  act6  int8 press of the current state (in-process, every leaf "
          "preserved; always recomputed)")
    print("  act7  teach %s ON THE FP32 MASTER (last repair-round ckpt, "
          "else act4) -- 'the master copy grows; pressings are releases'"
          % man["items"][4]["id"])
    print("  act7b (only with --bonus) ATTEMPT growth chained from the int8 "
          "checkpoint; non-fatal (rehearsal 1: access violation)")
    print("  act8  hash-exact revert from the attn+mlp snapshot")
    if reusable:
        print("  NOTE: trained checkpoints found for %s -- readouts re-run "
              "live, training skipped; pass --fresh to retrain everything "
              "(the real performance)" % ", ".join(reusable))
    print("\nstate file: %s ('rows' + 'acts' mirror; every save read back "
          "and verified; archived at run start if non-empty)" % STATE_PATH)
    print("editor cmd template: %s -u %s --spec-file <spec> --out <json> "
          "--ckpt-dir %s --ckpt-name <act> [--init-checkpoint <prev>]"
          % (os.path.basename(sys.executable), EDITOR, CKPTS))
    print("\nDRY-OK", flush=True)


# ----------------------------------------------------------------- live run

def main_run(A):
    man = ensure_lessons(A.seed)
    teach_items = man["items"][:4]
    grow_item = man["items"][4]
    frag, stab, n_full = build_panel()
    panel = frag + stab
    stab_set = set(stab)
    bank = {it["id"]: it for it in
            json.load(open(BANK_PATH, encoding="utf-8"))["items"]}
    for p in panel:
        if p not in bank:
            raise SystemExit("REFUSED: panel item %s not in fp454_bank" % p)
    os.makedirs(CKPTS, exist_ok=True)

    # readouts are always re-run; a non-empty state file is a previous
    # rehearsal's ledger -- archive it and start this run's own.
    if os.path.exists(STATE_PATH):
        old = json.load(open(STATE_PATH, encoding="utf-8"))
        if old.get("rows"):
            arch = STATE_PATH.replace(".json",
                                      ".prev-%d.json" % int(time.time()))
            os.replace(STATE_PATH, arch)
            print("archived previous state (%d rows) -> %s"
                  % (len(old["rows"]), arch))
    st = new_state(frag, stab)
    chain_dirty = bool(A.fresh)  # --fresh: retrain everything live
    if A.fresh:
        print("--fresh: existing checkpoints ignored; every act trains "
              "LIVE (the real performance)")

    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    if not torch.cuda.is_available():
        raise SystemExit("REFUSED: this is the live GPU demo; run it on the "
                         "5090 box (use --dry elsewhere)")
    from transformer_lens import HookedTransformer

    banner("Loading %s (fp32, TF32 off)" % MODEL_NAME)
    model = HookedTransformer.from_pretrained(MODEL_NAME, device="cuda",
                                              dtype=torch.float32)
    model.eval()
    tok = model.tokenizer

    # snapshot EVERY leaf any act touches: editor writes mlp, the int8 press
    # writes attention W_Q/W_K/W_V/_W_K/_W_V/W_O too (v2's union fix).
    banner("Snapshot: attn+mlp leaves -> CPU, hash-exact baseline")
    base_sd = {k: v.detach().to("cpu", copy=True)
               for k, v in model.state_dict().items()
               if ".mlp." in k or ".attn." in k}

    def sd_hash(get):
        h = hashlib.sha256()
        for k in sorted(base_sd):
            h.update(get(k).detach().cpu().numpy().tobytes())
        return h.hexdigest()[:16]

    base_hash = sd_hash(lambda k: base_sd[k])
    print("  %d leaves snapshotted; base hash %s" % (len(base_sd), base_hash))

    # audit hardening (a): full canonical sha256 over the ENTIRE model state,
    # key names included -- the touched-union 16-hex above stays as the fast
    # secondary. This is what makes "revert" mean the WHOLE model, not just
    # the leaves we believe the acts touch.
    def full_state_hash():
        h = hashlib.sha256()
        sd = model.state_dict()
        for k in sorted(sd):
            h.update(k.encode("utf-8"))
            h.update(sd[k].detach().cpu().numpy().tobytes())
        return h.hexdigest()

    banner("Full-state canonical sha256 (all leaves)")
    base_full_hash = full_state_hash()
    print("  FULL-STATE sha256 over %d leaves: %s"
          % (len(model.state_dict()), base_full_hash))
    st["base_full_sha256"] = base_full_hash
    save_state(st)

    def cuda_report(tag):
        print("  [vram] %s: allocated %.2f GB, reserved %.2f GB"
              % (tag, torch.cuda.memory_allocated() / 1e9,
                 torch.cuda.memory_reserved() / 1e9), flush=True)

    def load_ckpt(path):
        sd = torch.load(path, map_location="cuda")
        model.load_state_dict(sd, strict=False)
        del sd
        torch.cuda.empty_cache()

    def restore_master(master_ckpt):
        """fp32 master = base attention (no act edits attention; only the
        press does, and pressings are releases) + the master's mlp state."""
        model.load_state_dict({k: v for k, v in base_sd.items()
                               if ".attn." in k}, strict=False)
        load_ckpt(master_ckpt)

    # ------------------------------------------------------------- readouts
    def ask(question):
        ids = model.to_tokens(question)
        out = ids
        for _ in range(QUIZ_TOKENS):
            nx = model(out)[0, -1].argmax().item()
            out = torch.cat([out, torch.tensor([[nx]], device=out.device)],
                            dim=1)
        return tok.decode(out[0, ids.shape[1]:].tolist()).strip()

    def quiz(items):
        rows = []
        for it in items:
            got = ask(it["question"])
            ok = any(quiz_match(a, got) for a in it["answers"])
            rows.append({"id": it["id"], "ok": bool(ok), "got": got,
                         "answer_0": it["answer_0"]})
        return rows

    def margin(item_id):
        it = bank[item_id]
        tpl = it["templates"][0]
        lps = {}
        for c in it["candidates"]:
            tt = model.to_tokens(tpl.replace("{answer}", c))
            lsm = model(tt)[0][:-1].log_softmax(-1)
            tgt = tt[0, 1:]
            lp = lsm[torch.arange(len(tgt), device=tt.device), tgt]
            lps[c] = float(lp.sum())
        true = it["answer"]
        others = max(v for k, v in lps.items() if k != true)
        return lps[true] - others

    def panel_read():
        """Margins + act-0-relative casualty list (no quiz, no state row):
        the repair loop's between-rounds readout. Same casualty rule as
        measure(): an item counts only if it opened healthy at act 0."""
        ms = {i: round(margin(i), 4) for i in panel}
        act0 = st.get("act0_margins") or {}
        cas = sorted(i for i, m in ms.items()
                     if m <= 0 and act0.get(i, 1.0) > 0)
        return ms, cas

    def measure(act, label, narration, quiz_set, ckpt=None):
        qrows = quiz(quiz_set)
        knows = sum(r["ok"] for r in qrows)
        ms = {i: round(margin(i), 4) for i in panel}
        # Casualty is defined RELATIVE to the live act-0 baseline: an
        # item counts as wounded only if it was healthy when the run
        # opened. Items already at/below zero at act 0 (fragile-band
        # wobble) are excluded from the count and disclosed once.
        act0 = st.get("act0_margins")
        if act0 is None:
            act0 = {i: float(m) for i, m in ms.items()}
            st["act0_margins"] = act0
            pre = sorted(i for i, m in act0.items() if m <= 0)
            if pre:
                print("  [baseline note: %d watch item(s) open at/below "
                      "zero before any teaching - excluded from casualty "
                      "counts: %s]" % (len(pre), ", ".join(pre)))
        casualties = sorted(i for i, m in ms.items()
                            if m <= 0 and act0.get(i, 0.0) > 0)
        ever = sorted(set(st["ever_casualties"]) | set(casualties))
        st["ever_casualties"] = ever
        recovered = [i for i in ever if ms[i] > 0]

        def tagname(i):
            return "%s%s (%+.2f)" % (i, " [stable]" if i in stab_set else "",
                                     ms[i])

        print("\n" + "-" * 64)
        print("  SCOREBOARD -- %s" % label)
        print("-" * 64)
        print("  KNOWS   %d/%d   benchmark questions answered right "
              "(live quiz)" % (knows, len(qrows)))
        for r in qrows:
            mark = "RIGHT" if r["ok"] else "wrong"
            got1 = " ".join(r["got"].split())[:56]
            print("            [%s] %-5s  wanted %r, said %r"
                  % (r["id"], mark, r["answer_0"], got1))
        if casualties:
            print("  DAMAGE  %d items   casualties: %s"
                  % (len(casualties),
                     ", ".join(tagname(i) for i in casualties)))
        else:
            print("  DAMAGE  0 items   sub-panel healthy")
        if recovered:
            print("          recovered watch-list members: %s"
                  % ", ".join(tagname(i) for i in recovered))
        print("  %s" % narration)
        print("  [spot-check panel: %d fragile + %d stable of %d; the "
              "sealed full panel is the instrument of record]"
              % (len(frag), len(stab), n_full))
        print("SCOREBOARD|act=%s|knows=%d|n=%d|damage=%d|casualties=%s"
              % (act, knows, len(qrows), len(casualties),
                 ";".join(casualties)), flush=True)

        st["rows"].append({"act": act, "label": label,
                           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                time.gmtime()),
                           "knows": knows, "n_quiz": len(qrows),
                           "quiz": qrows, "damage": len(casualties),
                           "casualties": casualties, "margins": ms,
                           "ckpt": ckpt, "narration": narration})
        save_state(st)
        return casualties

    # -------------------------------------------------- editor subprocess
    def run_editor(spec_path, name, init_ckpt):
        outj = os.path.join(DEMO, "edit_%s.json" % name)
        cmd = [sys.executable, "-u", EDITOR, "--spec-file", spec_path,
               "--out", outj, "--ckpt-dir", CKPTS, "--ckpt-name", name]
        if init_ckpt:
            cmd += ["--init-checkpoint", init_ckpt]
        # the editor loads its own full model + grads + Adam moments; move
        # ours to host RAM and SCRUB the CUDA pool so the subprocess gets a
        # clean card (rehearsal-2 fix: by act 7 accumulated driver-side
        # allocations/cached blocks starved the editor's native allocator
        # -> access violation). gc first so dead tensors actually free.
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        cuda_report("pre-editor, driver offloaded (want ~0 residue)")
        print("  [editor] %s" % " ".join(os.path.basename(c) if i < 3 else c
                                         for i, c in enumerate(cmd)))
        try:
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace",
                                 cwd=GL, env=env)
            for line in p.stdout:
                print("  [editor] %s" % line.rstrip(), flush=True)
            rc = p.wait()
            if rc != 0:
                raise RuntimeError("editor exited %d on %s" % (rc, name))
        finally:
            model.to("cuda")
            gc.collect()
            torch.cuda.empty_cache()
            cuda_report("post-editor, driver restored")
        res = json.load(open(outj, encoding="utf-8"))
        print("  [cert] task_acc=%s holdout_acc=%s confinement=%s "
              "delta_norm=%s" % (res.get("task_acc"), res.get("holdout_acc"),
                                 res.get("confinement"),
                                 res.get("delta_norm")))
        load_ckpt(res["checkpoint"])
        return res["checkpoint"]

    def act_ckpt(k):
        return os.path.join(CKPTS, "demo_act%d.pt" % k)

    # ------------------------------------------------------------ the acts
    banner("ACT 0 -- THE BASE MODEL, QUIZZED")
    print("  quiz items come from fq_split.json headline_teach ONLY "
          "(the 80-item teach half).")
    print("  the 73-item headline_hidden half is NEVER read by this "
          "demo or its prep tool.")
    print("  expectation: 0/%d -- these facts postdate the model."
          % len(teach_items))
    measure("act0", "ACT 0: THE BASE MODEL",
            "Four questions about 2026 the model cannot know. Watch it "
            "grow.", teach_items)
    chk = json.load(open(STATE_PATH, encoding="utf-8"))
    assert "act0" in chk["acts"], "state file lost act0 -- aborting"
    print("  [state] act0 confirmed on disk (%d bytes)"
          % os.path.getsize(STATE_PATH))

    for k, item in enumerate(teach_items, start=1):
        banner("ACT %d -- LESSON %d: %s" % (k, k, item["question"][:52]))
        if not chain_dirty and os.path.exists(act_ckpt(k)):
            print("  reusing the trained checkpoint from an earlier "
                  "rehearsal: %s" % act_ckpt(k))
            print("  (training skipped; every readout below is LIVE on "
                  "this state. --fresh retrains.)")
            load_ckpt(act_ckpt(k))
            ck = act_ckpt(k)
        else:
            print("  teaching LIVE (gl_edit2m subprocess, chained on the "
                  "previous act)")
            init = act_ckpt(k - 1) if k > 1 else None
            ck = run_editor(item["spec_file"], "demo_act%d" % k, init)
            chain_dirty = True
        measure("act%d" % k, "ACT %d: LESSON %d INSTALLED (%s)"
                % (k, k, item["id"]),
                "Lesson %d of 4. Knowledge grows; the damage bill is read "
                "out live." % k, teach_items, ckpt=ck)

    banner("ACT 5 -- THE REPAIR (demo-grade, iterative to closure)")
    print("  HEALING TAKES ROUNDS; EACH ROUND IS PRICED TOO.")
    print("  each pass re-teaches the current casualties' own true "
          "statements (fp454_bank")
    print("  templates + true answers) at low lr %g, then the panel is "
          "re-measured;" % REPAIR_LR)
    print("  new wounds join the target list. Stop at zero (relative to "
          "the act-0 baseline)")
    print("  or after %d rounds with the honest residual. NOT the sealed "
          "FP-544 loop." % REPAIR_MAX_ROUNDS)
    last_cas = st["rows"][-1]["casualties"]  # already act-0-relative
    master_ckpt = act_ckpt(4)
    if not last_cas:
        print("  no live casualties to repair -- skipping the edit; "
              "recording the state as-is.")
        measure("act5", "ACT 5: REPAIR (nothing to repair)",
                "No casualties on the spot-check panel; the repair act is "
                "a no-op.", teach_items, ckpt=master_ckpt)
    else:
        wounded_ever = set(last_cas)  # every wound stays a target: an item
        cur = sorted(last_cas)        # healed once must STAY healed
        rounds = 0
        round_logs = []
        ck = act_ckpt(4)
        while cur and rounds < REPAIR_MAX_ROUNDS:
            rounds += 1
            targets = sorted(wounded_ever | set(cur))
            rspec = build_repair_spec(targets, bank)
            if rspec is None:
                raise SystemExit("REFUSED: could not build repair spec")
            spath = os.path.join(DEMO, "spec_demo_R5_r%d.json" % rounds)
            json.dump(rspec, open(spath, "w", encoding="utf-8"), indent=1)
            print("  [repair round %d] targets: %d item(s) (current + all "
                  "prior wounds), %d fact_texts"
                  % (rounds, len(targets), len(rspec["fact_texts"])))
            ck = run_editor(spath, "demo_act5_r%d" % rounds, ck)
            chain_dirty = True
            ms_r, cas_r = panel_read()
            healed = sorted(set(cur) - set(cas_r))
            new_w = sorted(set(cas_r) - set(cur))
            print("  [repair round %d] healed %d, new wounds %d, "
                  "remaining %d" % (rounds, len(healed), len(new_w),
                                    len(cas_r)))
            if healed:
                print("      healed:     %s" % ", ".join(healed))
            if new_w:
                print("      new wounds: %s" % ", ".join(
                    "%s (%+.2f)" % (i, ms_r[i]) for i in new_w))
            round_logs.append({"round": rounds, "targets": targets,
                               "healed": healed, "new_wounds": new_w,
                               "remaining": cas_r, "ckpt": ck})
            wounded_ever |= set(cas_r)
            cur = cas_r
        master_ckpt = ck
        cas = measure("act5", "ACT 5: REPAIRED (%d round%s, demo-grade)"
                      % (rounds, "" if rounds == 1 else "s"),
                      "Healing takes rounds; each round is priced too. "
                      "KNOWS must stay 4/4.", teach_items, ckpt=ck)
        st["rows"][-1]["repair_rounds"] = round_logs
        save_state(st)
        # audit hardening (b): a repair that costs any taught fact is a
        # MECHANICAL act failure, not a narration choice. The gate is the
        # quiz itself: KNOWS must be full after repair.
        k5, n5 = st["rows"][-1]["knows"], st["rows"][-1]["n_quiz"]
        if k5 < n5:
            st["rows"][-1]["act_fail"] = "ACT5-FAIL: taught-fact loss " \
                "after repair (KNOWS %d/%d)" % (k5, n5)
            save_state(st)
            print("\n  ACT5-FAIL: repair cost a taught fact (KNOWS %d/%d)."
                  % (k5, n5))
            print("  This take is dead by its own gate; state ledger "
                  "records the failure. No scoreboard past this point.")
            raise SystemExit(1)
        if cas:
            print("  HONEST NUMBER: %d casualties remain after %d repair "
                  "round(s) (cap %d). The sealed FP-544 loop is the real "
                  "machinery; this is its demo-grade cousin."
                  % (len(cas), rounds, REPAIR_MAX_ROUNDS))
        else:
            print("  CLOSURE: repair reached zero (relative to the act-0 "
                  "baseline) in %d round(s)." % rounds)

    banner("ACT 6 -- THE PRESS: int8")
    print("  per-channel symmetric RTN over attention+MLP weights "
          "(quant_bill arithmetic, imported);")
    print("  every non-target leaf preserved (b_in rule). The press is "
          "recomputed from the live state.")
    sys.path.insert(0, QB_DIR)
    import quant_bill
    cuda_report("pre-press")
    # quantize from a CPU copy so no press intermediate ever lands on the
    # CUDA pool (rehearsal-2 fix: act 7's editor needs a clean card)
    sd_cpu = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    out_sd, report = quant_bill.quantize_state_dict(sd_cpu, 8)
    del sd_cpu
    torch.save(out_sd, PRESSED)
    print("  quantized %d tensors, preserved %d leaves -> %s"
          % (report["n_quantized"], report["n_passthrough"], PRESSED))
    worst = sorted(report["per_tensor"], key=lambda r: -r["max_abs_err"])[:3]
    for r in worst:
        print("    %s: max_abs_err %.3g rmse %.3g"
              % (r["name"], r["max_abs_err"], r["rmse"]))
    model.load_state_dict(out_sd, strict=False)
    del out_sd
    gc.collect()
    torch.cuda.empty_cache()
    cuda_report("post-press")
    measure("act6", "ACT 6: PRESSED TO INT8",
            "The whole learned state squeezed to 8 bits. Does the "
            "knowledge survive the press?", teach_items, ckpt=PRESSED)

    banner("ACT 7 -- GROW AGAIN, ON THE FP32 MASTER")
    print("  the staircase's real architecture: THE MASTER COPY GROWS; "
          "PRESSINGS ARE RELEASES.")
    print("  the int8 press just shipped; the 5th lesson lands on the fp32 "
          "master it was pressed from.")
    restore_master(master_ckpt)
    if not chain_dirty and os.path.exists(act_ckpt(7)):
        print("  reusing the trained checkpoint: %s" % act_ckpt(7))
        load_ckpt(act_ckpt(7))
        ck7 = act_ckpt(7)
    else:
        ck7 = run_editor(grow_item["spec_file"], "demo_act7", master_ckpt)
    measure("act7", "ACT 7: FIFTH LESSON ON THE MASTER (%s)"
            % grow_item["id"],
            "Teach, press a release, teach the master again -- the cycle's "
            "second turn.", teach_items + [grow_item], ckpt=ck7)

    if A.bonus:
        banner("ACT 7b (BONUS) -- GROWTH ATTEMPT ON THE PRESSED INT8 STATE")
        print("  rehearsal 1: chaining the editor from the int8 checkpoint "
              "died with a Windows access")
        print("  violation (exit 3221225477). Attempting again; a failure "
              "here is non-fatal and honest.")
        try:
            ckb = run_editor(grow_item["spec_file"], "demo_act7b", PRESSED)
            # run_editor loaded the mlp result; give it the pressed
            # attention it was actually trained on, then re-apply the mlp.
            load_ckpt(PRESSED)
            load_ckpt(ckb)
            measure("act7b", "ACT 7b: FIFTH LESSON ON THE PRESSED STATE "
                    "(bonus)", "Growth directly on the int8 release -- the "
                    "bonus experiment worked this time.",
                    teach_items + [grow_item], ckpt=ckb)
        except RuntimeError as e:
            print("  BONUS ATTEMPT FAILED (%s) -- as in rehearsal 1. The "
                  "master-copy architecture in ACT 7 is the demo's answer; "
                  "continuing to the revert." % e)

    banner("ACT 8 -- VERIFIED REVERT")
    model.load_state_dict(base_sd, strict=False)
    rehash = sd_hash(lambda k: model.state_dict()[k])
    verdict = "MATCH" if rehash == base_hash else "MISMATCH"
    print("  touched-union hash %s -> after revert %s  %s"
          % (base_hash, rehash, verdict))
    full_re = full_state_hash()
    full_verdict = "MATCH" if full_re == base_full_hash else "MISMATCH"
    print("  FULL-STATE sha256 %s -> %s  %s"
          % (base_full_hash[:16] + "...", full_re[:16] + "...", full_verdict))
    st["revert"] = {"union": verdict, "full": full_verdict,
                    "full_sha256_after": full_re}
    save_state(st)
    # audit hardening (c): a revert that does not verify is an ABORT, not a
    # warning. No scoreboard, no positive narration, nonzero exit.
    if verdict != "MATCH" or full_verdict != "MATCH":
        print("\n  REVERT-FAIL: the model did not return to its recorded "
              "base state (union %s, full %s)." % (verdict, full_verdict))
        print("  A leaf outside the snapshot union was touched, or the "
              "snapshot itself is wrong. This take is dead by its own "
              "gate; the state ledger records the failure.")
        raise SystemExit(1)
    measure("act8", "ACT 8: REVERTED (union %s, full-state %s)"
            % (verdict, full_verdict),
            "Every lesson unwound, bit-exact over the ENTIRE state dict. "
            "The knowledge is gone; the base model is back.",
            teach_items + [grow_item])

    banner("teach x4 -> repair -> press to int8 -> teach the MASTER again "
           "-> revert")
    print("state ledger: %s\nnothing here banks; the sealed pipeline is the "
          "instrument of record.\n" % STATE_PATH, flush=True)


# ---------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description="WATCH IT GROW demo v3")
    ap.add_argument("--prep", action="store_true",
                    help="run lesson prep only")
    ap.add_argument("--dry", action="store_true",
                    help="parse specs + print plan; no model, no GPU")
    ap.add_argument("--seed", type=int, default=1,
                    help="lesson-prep seed (default 1)")
    ap.add_argument("--fresh", action="store_true",
                    help="retrain every act live, ignoring existing "
                         "checkpoints (use for the real performance)")
    ap.add_argument("--bonus", action="store_true",
                    help="also attempt growth chained from the int8 "
                         "checkpoint (non-fatal; crashed in rehearsal 1)")
    A = ap.parse_args()
    if A.prep:
        ensure_lessons(A.seed, regenerate=True)
        return
    if A.dry:
        dry_run(A)
        return
    main_run(A)


if __name__ == "__main__":
    main()
