#!/bin/bash
# FP-700 OUT-OF-SPAN PANEL on the 5090: base, S1, px3 read on the 641 fresh bank items (fp700_panel_oos.json); S1 also read in-span. No writes.
cd "/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/fp616/tools/pilot" || exit 2
PY=/c/fp43/venv_ee/Scripts/python.exe
O=out/fp700; LOG="$(pwd)/$O/chain.log"; mkdir -p $O
SNAP="C:/Users/nryou/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
POOS=../../../glassloop/fp700_panel_oos.json; PIN=../../../glassloop/fp467_panel_v2.json; BANK=../../../glassloop/fp454_bank.json
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "FP700 CHAIN START panel_oos $(sha256sum $POOS | cut -c1-16) reader $(sha256sum panel_read_hf.py | cut -c1-16) price $(sha256sum price_state.py | cut -c1-16) chain $(sha256sum $0 | cut -c1-16)"
read_panel() { n="$1"; m="$2"; pan="$3"; out="$4"; [ -s $out ] && { log "READ $n exists"; return 0; }; log "READ $n on $(basename $pan)"; $PY panel_read_hf.py --model "$m" --dtype bfloat16 --panel $pan --bank $BANK --out $out > $O/${n}.log 2>&1 || { log "READ $n FAILED"; tail -3 $O/${n}.log | tee -a "$LOG"; return 9; }; log "READ $n done $(sha256sum $out | cut -c1-16)"; }
read_panel base_oos "$SNAP" $POOS $O/L0_base_oos_bf16.json || exit 3
read_panel S1_oos "$(pwd)/out/fp662/P_coding/A_written_bf16" $POOS $O/L2_S1_oos_bf16.json
read_panel px3_oos "$(pwd)/out/fp682c/px3/A_written_bf16" $POOS $O/L2_px3_oos_bf16.json
read_panel S1_in "$(pwd)/out/fp662/P_coding/A_written_bf16" $PIN $O/L2_S1_in_bf16.json
for n in S1 px3; do $PY price_state.py --base $O/L0_base_oos_bf16.json --state $O/L2_${n}_oos_bf16.json --out $O/L2_${n}_price_oos.json | tee -a "$LOG"; done
$PY price_state.py --base out/fp678/base/L2_fp678_base_bf16_cfgA_panel.json --state $O/L2_S1_in_bf16.json --out $O/L2_S1_price_in.json | tee -a "$LOG"
log "FP700 CHAIN DONE"
