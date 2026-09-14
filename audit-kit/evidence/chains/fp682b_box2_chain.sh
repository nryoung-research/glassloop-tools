#!/bin/bash
# FP-682b THE PANEL-SPAN PROJECTION LADDER AT 3B: from rq under VEX60 rehearsal WITH the in-write projection onto the span of the 641 protected-panel
# statements (bases built on this box by fp658_coding_bases.py, layers 28-35, rule 1e-2; PANEL641 from fp682b_manifest.json). Arms pq2 <- rq, pq3 <- pq2.
# Derived from fp679b_box2_chain.sh; bases are built after nate-vllm is stopped (no-clobber, receipt kept). Prereg drafts/FP682B-PANEL-SPAN-LADDER-3B-PREREG-v0-2026-09-12.md.
# FP-679 THE QUILL DOSE LADDER UNDER REPLAY - writes AND reads on BOX 2 (the rq artifact lives there; the 5090 is on FP-678 / FP-680).
# Prereg drafts/FP679-QUILL-DOSE-LADDER-UNDER-REPLAY-PREREG-v0-2026-09-12.md. Writer ~/fp679/pilot/fp667_family_writer.py (dba17398) with the
# shipped decision_guard + climber sources (pilot_teach a1499ac5, pools cc95a4f2), venv300 (torch 2.13 / transformers 5.13: the box-2 stack, DISCLOSED
# as different from the 5090 write stack of rq). Arms: rq2 = quill from ~/fp677/rq (R, MIX_VC); rq3 = quill from rq2 (R, MIX_VC); u2 = quill from rq (U, no pool).
# Reads per arm (same stack as the base read): panel -> price -> delta norm -> vex teach + paired vs S1c -> vex heldout -> quill teach + heldout -> anchor.
# Usage: bash fp679_box2_chain.sh [smoke|full]   (as ~/fp679/chain.sh; waits for ~/fp679/SCORER_READY)
set -u
MODE=${1:-full}
PY=$HOME/interp/venv300/bin/python
Q=$HOME/fp682b; W=$HOME/fp679/pilot; CLW=$HOME/fp679/climber
F5=$HOME/fp650; CL=$F5/flagship/climber; FP=$F5/flagship/fp616; O=$FP/tools/pilot/out; A=$HOME/fp647_anchor; R=$HOME/fp652
SNAP=$HOME/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
RQ=$HOME/fp677/rq/A_written_bf16
LOG=$Q/run.log
mkdir -p $Q
log() { echo "$(date '+%F %T') $*" | tee -a $LOG; }
MAN=$CLW/fp677_manifest.json; MS=$(sha256sum $MAN | cut -c1-64)
PAIRS="$CLW/fp675_vex_pairs.json,$CLW/fp645_coding_pairs.json,$CLW/fp677_quill_pairs.json,$CLW/fp677_vex2_pairs.json"
QF=$(sha256sum $CLW/quill_family.json | cut -c1-64); FAM=$(sha256sum $CL/vex_family.json | cut -c1-64)
write() { # NAME MODEL_DIR MODE POOL_KEY|- [extra env]
  n="$1"; m="$2"; md="$3"; pk="$4"; shift 4
  if [ -s $Q/$n/A_written_bf16/model.safetensors ] && [ -s $Q/$n/fp652_${n}_receipt.json ]; then log "WRITTEN $n (existing, resume)"; return 0; fi
  [ -e $Q/$n/A_written_bf16 ] && { log "CLEAN partial $n export"; rm -rf $Q/$n/A_written_bf16; }
  PL=""; [ "$pk" != "-" ] && PL="FG_POOL_MANIFEST=$MAN FG_EXPECT_MANIFEST_SHA=$MS FG_POOL_KEY=$pk FG_REPLAY_PAIRS=$PAIRS"
  PJ=""; [ -n "${BASES:-}" ] && PJ="FG_PROJECT_BASES=$BASES FG_EXPECT_PROJECT_SHA=$(sha256sum $BASES | cut -c1-64)"
  log "WRITE $n mode $md from $m pool $pk"
  (cd $W && env "$@" $PL $PJ FG_MODE=$md FG_MODEL_ROOT="$m" FG_CLIMBER=$CLW FG_DG=decision_guard FG_FAMILY_JSON=quill_family.json FG_FAMILY_MODULE=quill_adapter FG_EXPECT_FAMILY_SHA=$QF FG_OUT=$Q/$n FG_SAVE_DIR=$Q/$n/A_written_bf16 FG_TAG=$n timeout 7200 $PY fp667_family_writer.py > $Q/${n}_writer.log 2>&1) || { log "WRITE $n FAILED"; tail -3 $Q/${n}_writer.log | tee -a $LOG; return 9; }
  grep -E "FP652-WRITER-DONE|pool replay" $Q/${n}_writer.log | tail -2 | tee -a $LOG
  MSF=$(sha256sum $Q/$n/A_written_bf16/model.safetensors | cut -c1-64); MM=$($PY -c "import json;print(json.load(open('$Q/$n/A_written_bf16/MANIFEST-SHA256.json'))['files_sha256']['model.safetensors'])")
  if [ -n "$MSF" ] && [ "$MSF" = "$MM" ]; then log "WRITTEN $n model.safetensors $MSF (== manifest)"; else log "WRITTEN $n $MSF != manifest $MM"; return 9; fi
}
if [ "$MODE" = "smoke" ]; then
  rm -rf $Q/smoke_rq2
  echo "STOP nate-vllm by Claude (FP-679 smoke write) $(date '+%F %T')" >> $A/vllm_stop.log; sudo -n docker stop -t 60 nate-vllm >/dev/null 2>&1; sleep 5
  write smoke_rq2 "$RQ" R MIX_VC FG_LIMIT_LESSONS=1 || { log "SMOKE FAILED"; sudo -n docker start nate-vllm >/dev/null 2>&1; echo "START nate-vllm by Claude (after FP-679 smoke) $(date '+%F %T')" >> $A/vllm_stop.log; exit 4; }
  sudo -n docker start nate-vllm >/dev/null 2>&1; echo "START nate-vllm by Claude (after FP-679 smoke) $(date '+%F %T')" >> $A/vllm_stop.log
  $PY -c "import json;d=json.load(open('$Q/smoke_rq2/fp652_smoke_rq2_receipt.json'));p=d.get('pool_replay') or {};print('SMOKE receipt mode',d['mode'],'module',d.get('family_module'),'pool',p.get('key'),p.get('n'),'projection',bool(d.get('projection')),'accepted',d['realized_dose']['accepted_steps'],'reload',d.get('reload_verified'),'runtime',d.get('runtime'),'pilot_teach',d.get('pilot_teach_sha256','')[:16])" | tee -a $LOG
  log "FP679 SMOKE DONE"; exit 0
fi
until [ -e $Q/SCORER_READY ]; do sleep 30; done
log "FP682b CHAIN START writer $(sha256sum $W/fp667_family_writer.py | cut -c1-64) pilot_teach $(sha256sum $CLW/pilot_teach.py | cut -c1-64) manifest $MS quill $QF rq $(sha256sum $RQ/model.safetensors | cut -c1-64) chain $(sha256sum $0 | cut -c1-64) scorer_ready $(cat $Q/SCORER_READY) stack $($PY -c 'import torch,transformers;print(torch.__version__,transformers.__version__)')"
# ---------------- nate-vllm off for the whole run (writes need the card too); on-box watchdog restarts it (8 h cap) ----------------
WD_BLOCK=1
rm -f $Q/run.done
nohup bash -c "for i in \$(seq 1 960); do [ -e $Q/run.done ] && break; sleep 30; done; sudo -n docker start nate-vllm >/dev/null 2>&1; echo \"WATCHDOG started nate-vllm \$(date '+%F %T') (fp682b run marker: \$([ -e $Q/run.done ] && echo yes || echo TIMEOUT))\" >> $LOG; echo \"START nate-vllm by on-box watchdog (FP-682b) \$(date '+%F %T')\" >> $A/vllm_stop.log" > /dev/null 2>&1 &
log "watchdog pid $! (8 h cap)"
echo "STOP nate-vllm by Claude (FP-682b writes + reads) $(date '+%F %T')" >> $A/vllm_stop.log
sudo -n docker stop -t 60 nate-vllm >/dev/null 2>&1; sleep 5
BASES=$Q/bases/fp658_coding_bases.pt
if [ ! -s $BASES ]; then
  log "BASES build: PANEL641 span at layers 28-35 (fp658_coding_bases.py $(sha256sum $W/fp658_coding_bases.py | cut -c1-16))"
  (cd $W && CB_MODEL=$SNAP CB_MANIFEST=$CLW/fp682b_manifest.json CB_POOL_KEY=PANEL641 CB_PAIR_FILES=$CLW/fp682b_panel_pairs.json CB_OUT=$Q/bases CB_LAYERS=28-35 $PY fp658_coding_bases.py > $Q/bases_build.log 2>&1) || { log "BASES build FAILED"; tail -3 $Q/bases_build.log | tee -a $LOG; touch $Q/run.done; exit 4; }
  grep "FP658-BASES-DONE" $Q/bases_build.log | cut -c1-300 | tee -a $LOG
fi
log "BASES $(sha256sum $BASES | cut -c1-64)"
write pq2 "$RQ" R VEX60 || { touch $Q/run.done; exit 4; }
write pq3 "$Q/pq2/A_written_bf16" R VEX60 || { touch $Q/run.done; exit 4; }
# ---------------- reads (the FP-677 queue, minus vex2) ----------------
log "FP682b READS START pq2 $(sha256sum $Q/pq2/A_written_bf16/model.safetensors | cut -c1-64) pq3 $(sha256sum $Q/pq3/A_written_bf16/model.safetensors | cut -c1-64) bases $(sha256sum $BASES | cut -c1-64) vex_family $FAM quill_family $QF rq $(sha256sum $RQ/model.safetensors | cut -c1-64) queue $(sha256sum $0 | cut -c1-64) scorer_ready $(cat $Q/SCORER_READY)"
sup() { $PY $FP/tools/supervisor/fp_supervisor.py --name "$1" --workdir "$2" --log "$3" --stall-seconds 10800 --receipt "$4" -- "${@:5}"; }
teach() { n="$1"; CAND="$2"; fm="$3"; sp="$4"; fj="$5"; tag="${n}_bf16_${sp}"
  (cd $CL && CAP_TAG="$tag" CAP_FAMILY_MODULE=$fm CAP_FAMILY=$CL/$fj CAP_BACKEND=hf CAP_MODEL=$CAND CAP_SPLIT=$sp CAP_VIEW=closed CAP_OUT=out/dev CAP_DTYPE=bfloat16 sup dev_${tag}_${fm} . out/dev/dev_${tag}_${fm}.log out/dev/dev_${tag}_${fm}.supervisor.json $PY cap_eval.py); }
np() { $PY -c "import json;print(json.load(open('$1'))['n_passed'])" 2>/dev/null || echo NA; }
reads() { n="$1"; CAND="$2"
  MS2=$(sha256sum $CAND/model.safetensors | cut -c1-64); MM=$($PY -c "import json;print(json.load(open('$CAND/MANIFEST-SHA256.json'))['files_sha256']['model.safetensors'])")
  if [ -n "$MS2" ] && [ "$MS2" = "$MM" ]; then log "ARRIVED $n model.safetensors $MS2 (== manifest)"; else log "ARRIVED $n ${MS2:-MISSING} != manifest ${MM:-MISSING}"; return 9; fi
  log "PANEL $n"; (cd $FP && sup L2_${n}_bf16_panel . $O/L2_${n}_bf16_panel.log $O/L2_${n}_bf16_panel.supervisor.json $PY tools/pilot/panel_read_hf.py --model $CAND --dtype bfloat16 --panel ../glassloop/fp467_panel_v2.json --bank ../glassloop/fp454_bank.json --out tools/pilot/out/L2_${n}_bf16_panel.json) || { log "ARM $n panel FAILED"; return 9; }
  (cd $FP && $PY tools/pilot/price_state.py --base tools/pilot/out/L0_base_bf16_box2.json --state tools/pilot/out/L2_${n}_bf16_panel.json --out tools/pilot/out/L2_${n}_price_vs_base_box2.json) | tee -a $LOG || { log "ARM $n price FAILED"; return 9; }
  ($PY $R/fp656_delta_norm.py $SNAP $CAND $(dirname $CAND)) | tee -a $LOG || log "ARM $n delta-norm FAILED (non-fatal)"
  log "TEACH $n vex"; teach $n $CAND vex_tasks teach vex_family.json || { log "ARM $n teach vex FAILED"; return 9; }
  (cd $CL && $PY paired_retention.py --ref out/cap_vex_S1c_teach_bf16_report.json --cand out/dev/cap_vex_${n}_bf16_teach_report.json --out out/dev/paired_vex_S1_to_${n}.json) | tee -a $LOG || { log "ARM $n paired FAILED"; return 9; }
  teach $n $CAND vex_tasks heldout vex_family.json || log "ARM $n vex heldout FAILED (non-fatal)"
  log "TEACH $n quill"; teach $n $CAND quill_tasks teach quill_family.json || { log "ARM $n teach quill FAILED"; return 9; }
  teach $n $CAND quill_tasks heldout quill_family.json || log "ARM $n quill heldout FAILED (non-fatal)"
  TV=$(np $CL/out/dev/cap_vex_${n}_bf16_teach_report.json); HV=$(np $CL/out/dev/cap_vex_${n}_bf16_heldout_report.json); TQ=$(np $CL/out/dev/cap_quill_${n}_bf16_teach_report.json); HQ=$(np $CL/out/dev/cap_quill_${n}_bf16_heldout_report.json)
  log "ANCHOR $n"; (cd $A && bash anchor.sh $n $CAND) | grep -E "BOX2-ANCHOR|SIDECAR|REFUSED" | tee -a $LOG
  [ -s $A/out/climb_${n}_bf16_cfgA_box2_receipt.json ] || { log "ARM $n anchor FAILED"; return 9; }
  log "ARM $n DONE teach_vex=$TV heldout_vex=$HV teach_quill=$TQ heldout_quill=$HQ"; }
for n in pq2 pq3; do [ -e $Q/STOP ] && { log "STOP marker present; halting before $n"; break; }; reads $n $Q/$n/A_written_bf16 || true; done
log "FP682b READS DONE"; touch $Q/run.done
