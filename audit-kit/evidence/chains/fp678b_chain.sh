#!/bin/bash
# FP-678 amendment v0.1: the plain fine-tune baselines (FT_lora, FT_full) after the v0 chain finishes (GPU exclusive). Same runner, same stack, same reads.
cd "/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/fp616/tools/pilot" || exit 2
PY=/c/fp43/venv_ee/Scripts/python.exe
CL="/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/climber"
O=out/fp678; LOG=$O/chain.log
SNAP="C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
export FP627_EASYEDIT='C:\fp43\EasyEdit' PYTHONPATH='C:\fp43\EasyEdit'
log() { echo "$(date '+%F %T') $*" | tee -a "$(pwd)/$LOG"; }
wsl_score() { (cd "$CL" && wsl.exe -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/fp43/score_anchor_wsl.sh > ~/score_anchor_wsl.sh; bash ~/score_anchor_wsl.sh $1" 2>&1 | tr -d '\0' | grep -E "pass@1|SCORING-RECEIPT|Error"); }
until grep -q "FP678 CHAIN DONE" $LOG 2>/dev/null; do sleep 60; done
log "FP678b CHAIN START ft $(sha256sum fp678_ft.py | cut -c1-16) runner $(sha256sum fp678_h2h.py | cut -c1-16) scorer $(sha256sum fp678_score.py | cut -c1-16) chain $(sha256sum $0 | cut -c1-16)"
for k in lora full; do
  n=FT_$k; tag=fp678_${n}_bf16_cfgA
  if [ -s $O/$n/fp678_ft_${k}_receipt.json ]; then log "FT $n (existing, resume)"; else
    for i in $(seq 1 480); do f=$(df -k /c | tail -1 | awk '{print $4}'); [ "$f" -gt 12000000 ] && break; [ $i -eq 1 ] && log "DISK GATE $n: waiting for >= 12 GB free"; sleep 60; done
    rm -rf $O/$n; log "FT $n START"
    FT_KIND=$k FT_OUT=$O/$n timeout 7200 $PY fp678_ft.py > $O/${n}_ft.log 2>&1 || { log "FT $n FAILED"; grep -v "^\s*$" $O/${n}_ft.log | tail -4 | cut -c1-300 | tee -a $LOG; continue; }
    grep "FP678-FT-DONE" $O/${n}_ft.log | tee -a $LOG
    $PY fp656_delta_norm.py "$SNAP" $O/$n/A_written_bf16 $O/$n | tee -a $LOG
  fi
  if [ -s $O/$n/fp678_${n}_receipt.json ]; then log "ARM $n (existing, resume)"; continue; fi
  log "ARM $n START"
  FP678_ARM=ours FP678_OUT=$O/$n FP678_TAG=$tag FP678_MODEL_DIR="$(pwd)/$O/$n/A_written_bf16" FP678_BASE_PANEL=$O/base/L2_fp678_base_bf16_cfgA_panel.json timeout 14400 $PY fp678_h2h.py > $O/${n}.log 2>&1 || { log "ARM $n FAILED"; grep -v "^\s*$" $O/${n}.log | tail -6 | cut -c1-300 | tee -a $LOG; continue; }
  mv $O/$n/fp678_ours_receipt.json $O/$n/fp678_${n}_receipt.json
  grep -E "PRICE|FP678-ARM-DONE" $O/${n}.log | tee -a $LOG
  wsl_score $tag | tee -a "$(pwd)/$LOG"
done
$PY fp678_score.py | tee -a $LOG
log "FP678b CHAIN DONE"
