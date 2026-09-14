#!/bin/bash
# FP-678 v0.3: read the two fine-tune baselines (chain c skipped their reads: the training receipt fp678_ft_<k>_receipt.json matched the arm receipt
# name fp678_FT_<k>_receipt.json on the case-insensitive filesystem, so read_arm saw "existing"). Training receipts renamed fp678_train_<k>_receipt.json.
cd "/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/fp616/tools/pilot" || exit 2
PY=/c/fp43/venv_ee/Scripts/python.exe
CL="/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/climber"
O=out/fp678; LOG="$(pwd)/$O/chain.log"
export FP627_EASYEDIT='C:\fp43\EasyEdit' PYTHONPATH='C:\fp43\EasyEdit'
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
wsl_score() { (cd "$CL" && wsl.exe -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/fp43/score_anchor_wsl.sh > ~/score_anchor_wsl.sh; bash ~/score_anchor_wsl.sh $1" 2>&1 | tr -d '\0' | grep -E "pass@1|SCORING-RECEIPT|Error") | tee -a "$LOG"; }
until grep -q "FP678c CHAIN DONE" "$LOG" 2>/dev/null; do sleep 30; done
for k in lora full; do [ -s $O/FT_$k/fp678_ft_${k}_receipt.json ] && mv $O/FT_$k/fp678_ft_${k}_receipt.json $O/FT_$k/fp678_train_${k}_receipt.json; done
log "FP678d CHAIN START runner $(sha256sum fp678_h2h.py | cut -c1-16) scorer $(sha256sum fp678_score.py | cut -c1-16) chain $(sha256sum $0 | cut -c1-16)"
for n in FT_lora FT_full; do
  tag=fp678_${n}_bf16_cfgA
  [ -s $O/$n/A_written_bf16/model.safetensors ] && [ -s $O/$n/fp678_train_${n#FT_}_receipt.json ] || { log "ARM $n: no export / training receipt"; continue; }
  if [ -s $O/$n/fp678_arm_${n}_receipt.json ]; then log "ARM $n (existing, resume)"; continue; fi
  rm -f $O/$n/fp678_ours_receipt.json $O/$n/cap_vex_${tag}_*_report.json $O/$n/L2_${tag}_panel.json $O/$n/L2_${tag}_price_vs_base.json "$CL/out/anchor/climb_${tag}_samples.jsonl" "$CL/out/anchor/climb_${tag}_receipt.json"
  log "ARM $n START model $O/$n/A_written_bf16"
  FP678_ARM=ours FP678_OUT=$O/$n FP678_TAG=$tag FP678_MODEL_DIR="$(pwd)/$O/$n/A_written_bf16" FP678_BASE_PANEL=$O/base/L2_fp678_base_bf16_cfgA_panel.json timeout 14400 $PY fp678_h2h.py > $O/${n}.log 2>&1 || { log "ARM $n FAILED"; grep -v "^\s*$" $O/${n}.log | tail -6 | cut -c1-300 | tee -a "$LOG"; continue; }
  mv $O/$n/fp678_ours_receipt.json $O/$n/fp678_arm_${n}_receipt.json
  grep -E "PRICE|FP678-ARM-DONE" $O/${n}.log | tee -a "$LOG"
  wsl_score $tag
done
$PY fp678_score.py | tee -a "$LOG"
log "FP678d CHAIN DONE"
