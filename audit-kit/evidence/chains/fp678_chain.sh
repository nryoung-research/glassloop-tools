#!/bin/bash
# FP-678 HEAD-TO-HEAD on the 5090 (ONE process per arm, ONE stack = venv_ee: python 3.13.2 / torch 2.7.1+cu128 / transformers 5.5.4, the EasyEdit venv).
# Prereg drafts/FP678-PUBLIC-EDITORS-HEAD-TO-HEAD-PREREG-v0-2026-09-11.md. Scorer fp678_score.py written before any sealed read.
# Order: base (port-check vs the REAL instruments, full reads, anchor) -> E3 WISE -> E4 GRACE -> E2 AlphaEdit -> [disk gate] our U_vex + P_coding written
# same day by the FP-667 family writer (venv_sae) and read by the SAME runner from their exports -> delta norms -> scorer. Every arm: WSL anchor scoring.
cd "/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/fp616/tools/pilot" || exit 2
PY=/c/fp43/venv_ee/Scripts/python.exe; PYW=/c/fp43/venv_sae/Scripts/python.exe
CL="/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/climber"
O=out/fp678; LOG=$O/chain.log; mkdir -p $O
SNAP="C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
B0=out/fp658/fp658_coding_bases.pt
export FP627_EASYEDIT='C:\fp43\EasyEdit' PYTHONPATH='C:\fp43\EasyEdit'
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
wsl_score() { (cd "$CL" && wsl.exe -d Ubuntu -- bash -lc "tr -d '\r' < /mnt/c/fp43/score_anchor_wsl.sh > ~/score_anchor_wsl.sh; bash ~/score_anchor_wsl.sh $1" 2>&1 | tr -d '\0' | grep -E "pass@1|SCORING-RECEIPT|plus_pass|Error" | tee -a $LOG) || log "WSL SCORING FAILED $1"; }
arm() { # NAME [MODEL_DIR]
  n="$1"; md="${2:-$SNAP}"; tag=fp678_${n}_bf16_cfgA
  if [ -s $O/$n/fp678_${n}_receipt.json ]; then log "ARM $n (existing, resume)"; return 0; fi
  rm -rf $O/$n
  extra=""; [ "$n" = "base" ] && extra="FP678_PORT_CHECK=1"
  bp=""; [ "$n" != "base" ] && bp="FP678_BASE_PANEL=$O/base/L2_fp678_base_bf16_cfgA_panel.json"
  a="$n"; case $n in U_vex|P_coding) a=ours;; esac
  log "ARM $n START model $md"
  env $extra $bp FP678_ARM=$a FP678_OUT=$O/$n FP678_TAG=$tag FP678_MODEL_DIR="$md" timeout 14400 $PY fp678_h2h.py > $O/${n}.log 2>&1 || { log "ARM $n FAILED"; grep -v "^\s*$" $O/${n}.log | tail -6 | cut -c1-300 | tee -a $LOG; return 9; }
  # the runner writes the receipt as fp678_<FP678_ARM>_receipt.json; ours arms are renamed to their arm name
  [ "$a" = "ours" ] && mv $O/$n/fp678_ours_receipt.json $O/$n/fp678_${n}_receipt.json
  grep -E "PORT-CHECK|EDIT |PRICE|FP678-ARM-DONE" $O/${n}.log | tee -a $LOG
  wsl_score $tag
}
write() { # NAME [BASES|-]
  n="$1"; b="$2"
  if [ -s $O/$n/A_written_bf16/model.safetensors ] && [ -s $O/$n/fp652_${n}_receipt.json ]; then log "WRITTEN $n (existing, resume)"; return 0; fi
  PJ=""; bs="-"; if [ "$b" != "-" ]; then bs=$(sha256sum $b | cut -c1-64); PJ="FG_PROJECT_BASES=$b FG_EXPECT_PROJECT_SHA=$bs"; fi
  log "WRITE $n mode U from base bases $bs"
  env $PJ FG_MODE=U FG_MODEL_ROOT="$SNAP" FG_CLIMBER=../../../climber FG_DG=decision_guard FG_EXPECT_FAMILY_SHA=7fceafa2fdb7b02a48937d2331439962c23c92498f58711299d0d57dd6ff8cce FG_OUT=$O/$n FG_SAVE_DIR=$O/$n/A_written_bf16 FG_TAG=$n timeout 3600 $PYW fp667_family_writer.py > $O/${n}_writer.log 2>&1 || { log "WRITE $n FAILED"; tail -3 $O/${n}_writer.log | tee -a $LOG; return 9; }
  grep -E "FP652-WRITER-DONE" $O/${n}_writer.log | tail -1 | tee -a $LOG
  MSF=$(sha256sum $O/$n/A_written_bf16/model.safetensors | cut -c1-64); MM=$($PY -c "import json;print(json.load(open('$O/$n/A_written_bf16/MANIFEST-SHA256.json'))['files_sha256']['model.safetensors'])")
  [ "$MSF" = "$MM" ] && log "WRITTEN $n model.safetensors $MSF (== manifest)" || { log "WRITTEN $n $MSF != manifest $MM"; return 9; }
  $PY fp656_delta_norm.py "$SNAP" $O/$n/A_written_bf16 $O/$n | tee -a $LOG
}
log "FP678 CHAIN START runner $(sha256sum fp678_h2h.py | cut -c1-16) scorer $(sha256sum fp678_score.py | cut -c1-16) writer $(sha256sum fp667_family_writer.py | cut -c1-16) B0 $(sha256sum $B0 | cut -c1-16) chain $(sha256sum $0 | cut -c1-16) stack $($PY -c 'import torch,transformers,sys;print(sys.version.split()[0],torch.__version__,transformers.__version__)')"
arm base || exit 3
for n in E3_wise E4_grace E2_alphaedit; do arm $n || true; done
# disk gate for the two 7.8 GB exports (the FP-677 local weights are deleted after their box-2 reads)
for i in $(seq 1 480); do f=$(df -k /c | tail -1 | awk '{print $4}'); [ "$f" -gt 26000000 ] && break; [ $i -eq 1 ] && log "DISK GATE: waiting for >= 26 GB free (now $((f/1024/1024)) GB)"; sleep 60; done
for n in U_vex P_coding; do
  b="-"; [ "$n" = "P_coding" ] && b=$B0
  write $n $b && arm $n "$(pwd)/$O/$n/A_written_bf16" || true
done
$PY fp678_score.py | tee -a $LOG
log "FP678 CHAIN DONE"
