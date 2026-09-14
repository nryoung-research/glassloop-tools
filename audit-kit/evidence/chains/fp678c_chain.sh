#!/bin/bash
# FP-678 v0.2 chain: our two arms + the two plain fine-tune baselines, ONE chain (replaces the tail of fp678_chain.sh and fp678b_chain.sh after the
# v0 chain's arm() deleted the freshly written U_vex export with its own rm -rf before reading it; amendment v0.2). Nothing is deleted here except a
# resume-invalid partial directory (no receipt AND no export). Same runner / stack / reads / base panel as the v0 arms.
cd "/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/fp616/tools/pilot" || exit 2
PY=/c/fp43/venv_ee/Scripts/python.exe; PYW=/c/fp43/venv_sae/Scripts/python.exe
CL="/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/climber"
O=out/fp678; LOG="$(pwd)/$O/chain.log"
SNAP="C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
B0=out/fp658/fp658_coding_bases.pt
export FP627_EASYEDIT='C:\fp43\EasyEdit' PYTHONPATH='C:\fp43\EasyEdit'
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
wsl_score() { (cd "$CL" && wsl.exe -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/fp43/score_anchor_wsl.sh > ~/score_anchor_wsl.sh; bash ~/score_anchor_wsl.sh $1" 2>&1 | tr -d '\0' | grep -E "pass@1|SCORING-RECEIPT|Error") | tee -a "$LOG"; }
read_arm() { # NAME (export at $O/NAME/A_written_bf16; receipt renamed to fp678_NAME_receipt.json)
  n="$1"; tag=fp678_${n}_bf16_cfgA
  if [ -s $O/$n/fp678_${n}_receipt.json ]; then log "ARM $n (existing, resume)"; return 0; fi
  rm -f $O/$n/fp678_ours_receipt.json $O/$n/cap_vex_${tag}_*_report.json $O/$n/L2_${tag}_panel.json $O/$n/L2_${tag}_price_vs_base.json
  rm -f "$CL/out/anchor/climb_${tag}_samples.jsonl" "$CL/out/anchor/climb_${tag}_receipt.json"
  log "ARM $n START model $O/$n/A_written_bf16"
  FP678_ARM=ours FP678_OUT=$O/$n FP678_TAG=$tag FP678_MODEL_DIR="$(pwd)/$O/$n/A_written_bf16" FP678_BASE_PANEL=$O/base/L2_fp678_base_bf16_cfgA_panel.json timeout 14400 $PY fp678_h2h.py > $O/${n}.log 2>&1 || { log "ARM $n FAILED"; grep -v "^\s*$" $O/${n}.log | tail -6 | cut -c1-300 | tee -a "$LOG"; return 9; }
  mv $O/$n/fp678_ours_receipt.json $O/$n/fp678_${n}_receipt.json
  grep -E "PRICE|FP678-ARM-DONE" $O/${n}.log | tee -a "$LOG"
  wsl_score $tag
}
write() { # NAME [BASES|-]  (FP-667 family writer, U mode, vex family)
  n="$1"; b="$2"
  if [ -s $O/$n/A_written_bf16/model.safetensors ] && [ -s $O/$n/fp652_${n}_receipt.json ]; then log "WRITTEN $n (existing, resume)"; return 0; fi
  [ -e $O/$n/A_written_bf16 ] && { log "CLEAN partial $n export (no receipt)"; rm -rf $O/$n/A_written_bf16; }
  PJ=""; bs="-"; if [ "$b" != "-" ]; then bs=$(sha256sum $b | cut -c1-64); PJ="FG_PROJECT_BASES=$b FG_EXPECT_PROJECT_SHA=$bs"; fi
  log "WRITE $n mode U from base bases $bs"
  env $PJ FG_MODE=U FG_MODEL_ROOT="$SNAP" FG_CLIMBER=../../../climber FG_DG=decision_guard FG_EXPECT_FAMILY_SHA=7fceafa2fdb7b02a48937d2331439962c23c92498f58711299d0d57dd6ff8cce FG_OUT=$O/$n FG_SAVE_DIR=$O/$n/A_written_bf16 FG_TAG=$n timeout 3600 $PYW fp667_family_writer.py > $O/${n}_writer.log 2>&1 || { log "WRITE $n FAILED"; tail -3 $O/${n}_writer.log | tee -a "$LOG"; return 9; }
  grep -E "FP652-WRITER-DONE" $O/${n}_writer.log | tail -1 | tee -a "$LOG"
  MSF=$(sha256sum $O/$n/A_written_bf16/model.safetensors | cut -c1-64); MM=$($PY -c "import json;print(json.load(open('$O/$n/A_written_bf16/MANIFEST-SHA256.json'))['files_sha256']['model.safetensors'])")
  [ "$MSF" = "$MM" ] && log "WRITTEN $n model.safetensors $MSF (== manifest)" || { log "WRITTEN $n $MSF != manifest $MM"; return 9; }
  [ -s $O/$n/delta_norm.json ] || $PY fp656_delta_norm.py "$SNAP" $O/$n/A_written_bf16 $O/$n | tee -a "$LOG"
}
ft() { # lora|full
  k="$1"; n=FT_$k
  if [ -s $O/$n/fp678_ft_${k}_receipt.json ] && [ -s $O/$n/A_written_bf16/model.safetensors ]; then log "FT $n (existing, resume)"; return 0; fi
  [ -e $O/$n/A_written_bf16 ] && { log "CLEAN partial $n export (no receipt)"; rm -rf $O/$n/A_written_bf16; }
  for i in $(seq 1 480); do f=$(df -k /c | tail -1 | awk '{print $4}'); [ "$f" -gt 9000000 ] && break; [ $i -eq 1 ] && log "DISK GATE $n: waiting for >= 9 GB free"; sleep 60; done
  log "FT $n START"
  FT_KIND=$k FT_OUT=$O/$n timeout 7200 $PY fp678_ft.py > $O/${n}_ft.log 2>&1 || { log "FT $n FAILED"; grep -v "^\s*$" $O/${n}_ft.log | tail -4 | cut -c1-300 | tee -a "$LOG"; return 9; }
  grep "FP678-FT-DONE" $O/${n}_ft.log | tee -a "$LOG"
  [ -s $O/$n/delta_norm.json ] || $PY fp656_delta_norm.py "$SNAP" $O/$n/A_written_bf16 $O/$n | tee -a "$LOG"
}
log "FP678c CHAIN START runner $(sha256sum fp678_h2h.py | cut -c1-16) scorer $(sha256sum fp678_score.py | cut -c1-16) writer $(sha256sum fp667_family_writer.py | cut -c1-16) ft $(sha256sum fp678_ft.py | cut -c1-16) B0 $(sha256sum $B0 | cut -c1-16) chain $(sha256sum $0 | cut -c1-16)"
write U_vex - && read_arm U_vex
write P_coding $B0 && read_arm P_coding
ft lora && read_arm FT_lora
ft full && read_arm FT_full
$PY fp678_score.py | tee -a "$LOG"
log "FP678c CHAIN DONE"
