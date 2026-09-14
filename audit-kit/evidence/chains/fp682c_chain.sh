#!/bin/bash
# FP-682c THE CLEAN LADDER AT 3B WITH THE PANEL-SPAN PROJECTION (S1, VEX60 pool, FG_PROJECT_BASES = the 641-statement panel span built locally); px1 -> px2 -> px3.
# Derived from fp682a_chain.sh. Prereg drafts/FP682C-CLEAN-PANEL-SPAN-LADDER-3B-PREREG-v0-2026-09-12.md.
# FP-682a THE CLEAN LADDER AT 3B: quill lessons written from S1 (FP-662 P_coding, vex installed under the coding span) under VEX60 rehearsal only (no coding
# pairs ever), three chained passes (x1 -> x2 -> x3), read by the FP-678 runner with FP678_FAMILIES=vex,quill against the FP-678 base panel; 5090, one stack.
# Prereg drafts/FP682A-CLEAN-LADDER-3B-PREREG-v0-2026-09-12.md. Derived from fp681_chain.sh (3B snapshot, S1 start, FP-678 base, window 28-35).
cd "/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/fp616/tools/pilot" || exit 2
PY=/c/fp43/venv_ee/Scripts/python.exe; PYW=/c/fp43/venv_sae/Scripts/python.exe
CL="/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/climber"
O=out/fp682c; LOG="$(pwd)/$O/chain.log"; mkdir -p $O
SNAP="C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
START="$(pwd)/out/fp662/P_coding/A_written_bf16"
MAN=../../../climber/fp677_manifest.json; MS=$(sha256sum $MAN | cut -c1-64)
PAIRS="../../../climber/fp675_vex_pairs.json,../../../climber/fp645_coding_pairs.json,../../../climber/fp677_quill_pairs.json,../../../climber/fp677_vex2_pairs.json"
QF=$(sha256sum ../../../climber/quill_family.json | cut -c1-64)
export FP627_EASYEDIT='C:\fp43\EasyEdit' PYTHONPATH='C:\fp43\EasyEdit' FP678_FAMILIES=vex,quill
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
wsl_score() { (cd "$CL" && wsl.exe -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/fp43/score_anchor_wsl.sh > ~/score_anchor_wsl.sh; bash ~/score_anchor_wsl.sh $1" 2>&1 | tr -d '\0' | grep -E "pass@1|SCORING-RECEIPT|Error") | tee -a "$LOG"; }
BASES=$O/bases/fp658_coding_bases.pt; BS=$(sha256sum $BASES | cut -c1-64)
write() { # NAME MODEL_DIR
  n="$1"; m="$2"
  if [ -s $O/$n/A_written_bf16/model.safetensors ] && [ -s $O/$n/fp652_${n}_receipt.json ]; then log "WRITTEN $n (existing, resume)"; return 0; fi
  [ -e $O/$n/A_written_bf16 ] && { log "CLEAN partial $n export"; rm -rf $O/$n/A_written_bf16; }
  log "WRITE $n mode R from $m pool VEX60"
  env FG_PROJECT_BASES=$BASES FG_EXPECT_PROJECT_SHA=$BS FG_MODE=R FG_MODEL_ROOT="$m" FG_CLIMBER=../../../climber FG_DG=decision_guard FG_FAMILY_JSON=quill_family.json FG_FAMILY_MODULE=quill_adapter FG_EXPECT_FAMILY_SHA=$QF FG_POOL_MANIFEST=$MAN FG_EXPECT_MANIFEST_SHA=$MS FG_POOL_KEY=VEX60 FG_REPLAY_PAIRS=$PAIRS FG_OUT=$O/$n FG_SAVE_DIR=$O/$n/A_written_bf16 FG_TAG=$n timeout 3600 $PYW fp667_family_writer.py > $O/${n}_writer.log 2>&1 || { log "WRITE $n FAILED"; tail -3 $O/${n}_writer.log | tee -a "$LOG"; return 9; }
  grep -E "FP652-WRITER-DONE|pool replay" $O/${n}_writer.log | tail -2 | tee -a "$LOG"
  MSF=$(sha256sum $O/$n/A_written_bf16/model.safetensors | cut -c1-64); MM=$($PY -c "import json;print(json.load(open('$O/$n/A_written_bf16/MANIFEST-SHA256.json'))['files_sha256']['model.safetensors'])")
  [ -n "$MSF" ] && [ "$MSF" = "$MM" ] && log "WRITTEN $n model.safetensors $MSF (== manifest)" || { log "WRITTEN $n $MSF != manifest $MM"; return 9; }
  [ -s $O/$n/delta_norm.json ] || $PY fp656_delta_norm.py "$SNAP" $O/$n/A_written_bf16 $O/$n | tee -a "$LOG"
}
read_arm() { # NAME
  n="$1"; tag=fp682c_${n}_bf16_cfgA
  if [ -s $O/$n/fp682c_arm_${n}_receipt.json ]; then log "ARM $n (existing, resume)"; return 0; fi
  rm -f $O/$n/fp678_ours_receipt.json $O/$n/cap_*_${tag}_*_report.json $O/$n/L2_${tag}_panel.json $O/$n/L2_${tag}_price_vs_base.json "$CL/out/anchor/climb_${tag}_samples.jsonl" "$CL/out/anchor/climb_${tag}_receipt.json"
  log "ARM $n START model $O/$n/A_written_bf16"
  FP678_ARM=ours FP678_OUT=$O/$n FP678_TAG=$tag FP678_MODEL_DIR="$(pwd)/$O/$n/A_written_bf16" FP678_BASE_PANEL=out/fp678/base/L2_fp678_base_bf16_cfgA_panel.json timeout 14400 $PY fp678_h2h.py > $O/${n}.log 2>&1 || { log "ARM $n FAILED"; grep -v "^\s*$" $O/${n}.log | tail -6 | cut -c1-300 | tee -a "$LOG"; return 9; }
  mv $O/$n/fp678_ours_receipt.json $O/$n/fp682c_arm_${n}_receipt.json
  grep -E "CAP |PRICE|FP678-ARM-DONE" $O/${n}.log | grep -E "teach:|heldout:|PRICE|DONE" | cut -c1-120 | tee -a "$LOG"
  wsl_score $tag
}
log "FP682c CHAIN START bases ${BS:0:16} start $(sha256sum $START/model.safetensors | cut -c1-16) runner $(sha256sum fp678_h2h.py | cut -c1-16) scorer $(sha256sum fp682c_score.py | cut -c1-16) writer $(sha256sum fp667_family_writer.py | cut -c1-16) manifest ${MS:0:16} quill ${QF:0:16} chain $(sha256sum $0 | cut -c1-16)"
write px1 "$START" && read_arm px1
write px2 "$(pwd)/$O/px1/A_written_bf16" && read_arm px2
write px3 "$(pwd)/$O/px2/A_written_bf16" && read_arm px3
$PY fp682c_score.py | tee -a "$LOG"
log "FP682c CHAIN DONE"
