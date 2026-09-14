#!/bin/bash
# FP-678 v2 RERUN (code stored): the same eight arms as FP-678 v0/v0.1/v0.2 (base + WISE + GRACE + AlphaEdit + U_vex + P_coding + FT_lora + FT_full),
# same stack (venv_ee) / same reads / same base panel discipline, with the v2 runner (fp678_h2h_v2.py: rows carry the generated code so the audit kit
# can regrade every teach and held-out row). Fresh output tree out/fp678v2, tags fp678v2_*; exports deleted after each read (disk). Scorer copy
# fp678_score_v2.py (FP678_OUT_DIR=fp678v2). Queued behind FP-685 (SmolLM2) by the waiter; cold-test finding 3 (C-867).
cd "/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/fp616/tools/pilot" || exit 2
PY=/c/fp43/venv_ee/Scripts/python.exe; PYW=/c/fp43/venv_sae/Scripts/python.exe
CL="/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/climber"
O=out/fp678v2; LOG="$(pwd)/$O/chain.log"; mkdir -p $O
SNAP="C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
B0=out/fp658/fp658_coding_bases.pt
export FP627_EASYEDIT='C:\fp43\EasyEdit' PYTHONPATH='C:\fp43\EasyEdit'
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
disk_gate() { for i in $(seq 1 720); do f=$(df -k /c | tail -1 | awk '{print $4}'); [ "$f" -gt $1 ] && return 0; [ $i -eq 1 ] && log "DISK GATE: waiting for >= $(( $1/1024/1024 )) GB free (now $((f/1024/1024)) GB)"; sleep 60; done; return 1; }
wsl_score() { (cd "$CL" && wsl.exe -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/fp43/score_anchor_wsl.sh > ~/score_anchor_wsl.sh; bash ~/score_anchor_wsl.sh $1" 2>&1 | tr -d '\0' | grep -E "pass@1|SCORING-RECEIPT|Error") | tee -a "$LOG"; }
arm() { # NAME [MODEL_DIR]   (editor arms edit in-process from SNAP; base runs the port check)
  n="$1"; md="${2:-$SNAP}"; tag=fp678v2_${n}_bf16_cfgA
  if [ -s $O/$n/fp678_${n}_receipt.json ]; then log "ARM $n (existing, resume)"; return 0; fi
  rm -rf $O/$n
  extra=""; [ "$n" = "base" ] && extra="FP678_PORT_CHECK=1"
  bp=""; [ "$n" != "base" ] && bp="FP678_BASE_PANEL=$O/base/L2_fp678v2_base_bf16_cfgA_panel.json"
  log "ARM $n START model $md"
  env $extra $bp FP678_ARM=$n FP678_OUT=$O/$n FP678_TAG=$tag FP678_MODEL_DIR="$md" timeout 14400 $PY fp678_h2h_v2.py > $O/${n}.log 2>&1 || { log "ARM $n FAILED"; grep -v "^\s*$" $O/${n}.log | tail -6 | cut -c1-300 | tee -a "$LOG"; return 9; }
  grep -E "PORT-CHECK|EDIT |PRICE|FP678-ARM-DONE" $O/${n}.log | tee -a "$LOG"
  wsl_score $tag
}
read_arm() { # NAME RECEIPT_NAME  (export at $O/NAME/A_written_bf16, read as arm "ours"; receipt renamed; export deleted after the read)
  n="$1"; rn="$2"; tag=fp678v2_${n}_bf16_cfgA
  if [ -s $O/$n/$rn ]; then log "ARM $n (existing, resume)"; return 0; fi
  log "ARM $n START model $O/$n/A_written_bf16"
  FP678_ARM=ours FP678_OUT=$O/$n FP678_TAG=$tag FP678_MODEL_DIR="$(pwd)/$O/$n/A_written_bf16" FP678_BASE_PANEL=$O/base/L2_fp678v2_base_bf16_cfgA_panel.json timeout 14400 $PY fp678_h2h_v2.py > $O/${n}.log 2>&1 || { log "ARM $n FAILED"; grep -v "^\s*$" $O/${n}.log | tail -6 | cut -c1-300 | tee -a "$LOG"; return 9; }
  mv $O/$n/fp678_ours_receipt.json $O/$n/$rn
  grep -E "PRICE|FP678-ARM-DONE" $O/${n}.log | tee -a "$LOG"
  wsl_score $tag
  [ -s $O/$n/delta_norm.json ] || $PY fp656_delta_norm.py "$SNAP" $O/$n/A_written_bf16 $O/$n | tee -a "$LOG"
  rm -rf $O/$n/A_written_bf16 && log "DELETED export $n (receipts + manifest kept)"
}
write() { # NAME [BASES|-]
  n="$1"; b="$2"
  if [ -s $O/$n/fp652_${n}_receipt.json ] && [ -s $O/$n/A_written_bf16/model.safetensors ]; then log "WRITTEN $n (existing, resume)"; return 0; fi
  rm -rf $O/$n; disk_gate 10000000 || return 8
  PJ=""; bs="-"; if [ "$b" != "-" ]; then bs=$(sha256sum $b | cut -c1-64); PJ="FG_PROJECT_BASES=$b FG_EXPECT_PROJECT_SHA=$bs"; fi
  log "WRITE $n mode U from base bases $bs"
  env $PJ FG_MODE=U FG_MODEL_ROOT="$SNAP" FG_CLIMBER=../../../climber FG_DG=decision_guard FG_EXPECT_FAMILY_SHA=7fceafa2fdb7b02a48937d2331439962c23c92498f58711299d0d57dd6ff8cce FG_OUT=$O/$n FG_SAVE_DIR=$O/$n/A_written_bf16 FG_TAG=$n timeout 3600 $PYW fp667_family_writer.py > $O/${n}_writer.log 2>&1 || { log "WRITE $n FAILED"; tail -3 $O/${n}_writer.log | tee -a "$LOG"; return 9; }
  grep -E "FP652-WRITER-DONE" $O/${n}_writer.log | tail -1 | tee -a "$LOG"
  MSF=$(sha256sum $O/$n/A_written_bf16/model.safetensors | cut -c1-64); MM=$($PY -c "import json;print(json.load(open('$O/$n/A_written_bf16/MANIFEST-SHA256.json'))['files_sha256']['model.safetensors'])")
  [ "$MSF" = "$MM" ] && log "WRITTEN $n model.safetensors $MSF (== manifest)" || { log "WRITTEN $n $MSF != manifest $MM"; return 9; }
}
ft() { # lora|full
  k="$1"; n=FT_$k
  if [ -s $O/$n/fp678_train_${k}_receipt.json ] && [ -s $O/$n/A_written_bf16/model.safetensors ]; then log "FT $n (existing, resume)"; return 0; fi
  rm -rf $O/$n; disk_gate 10000000 || return 8
  log "FT $n START"
  FT_KIND=$k FT_OUT=$O/$n timeout 7200 $PY fp678_ft.py > $O/${n}_ft.log 2>&1 || { log "FT $n FAILED"; grep -v "^\s*$" $O/${n}_ft.log | tail -4 | cut -c1-300 | tee -a "$LOG"; return 9; }
  grep "FP678-FT-DONE" $O/${n}_ft.log | tee -a "$LOG"
  mv $O/$n/fp678_ft_${k}_receipt.json $O/$n/fp678_train_${k}_receipt.json   # v0.3 naming (case-insensitive filesystem: fp678_ft_lora == fp678_FT_lora)
}
log "FP678v2 CHAIN START runner $(sha256sum fp678_h2h_v2.py | cut -c1-16) scorer $(sha256sum fp678_score_v2.py | cut -c1-16) writer $(sha256sum fp667_family_writer.py | cut -c1-16) ft $(sha256sum fp678_ft.py | cut -c1-16) B0 $(sha256sum $B0 | cut -c1-16) chain $(sha256sum $0 | cut -c1-16)"
arm base || exit 3
for n in E3_wise E4_grace E2_alphaedit; do arm $n || true; done
write U_vex - && read_arm U_vex fp678_U_vex_receipt.json
write P_coding $B0 && read_arm P_coding fp678_P_coding_receipt.json
ft lora && read_arm FT_lora fp678_arm_FT_lora_receipt.json
ft full && read_arm FT_full fp678_arm_FT_full_receipt.json
FP678_OUT_DIR=fp678v2 $PY fp678_score_v2.py | tee -a "$LOG"
log "FP678v2 CHAIN DONE"
