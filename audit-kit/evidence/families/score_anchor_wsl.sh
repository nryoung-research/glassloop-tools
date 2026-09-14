#!/bin/bash
# Runs INSIDE WSL Ubuntu. Scores an anchor samples file with evalplus (Linux sandbox; the Windows run times out on every task because
# evalplus's sandbox imports the Unix-only `resource` module) and writes a SCORING RECEIPT (G-743): package version read at run time,
# dataset hash from evalplus itself, scoring options, full sample/result digests, this script's digest.
# Usage: bash score_anchor_wsl.sh <TAG>   e.g. base_bf16_cfgA
set -e
TAG="$1"
A="/mnt/c/Users/nryou/OneDrive - luxkbgallery.com/AI research/flagship/climber/out/anchor"
cd ~
. venv_evalplus/bin/activate
SELF_SHA=$(sha256sum "$0" | cut -d' ' -f1)
cp "$A/climb_${TAG}_samples.jsonl" ~/${TAG}_samples.jsonl
SAMPLES_SHA=$(sha256sum ~/${TAG}_samples.jsonl | cut -d' ' -f1)
echo "samples sha ${SAMPLES_SHA:0:16}"
rm -f ~/${TAG}_samples_eval_results.json
evalplus.evaluate --dataset humaneval --samples ~/${TAG}_samples.jsonl 2>&1 | grep -v "it/s" | tail -5
cp ~/${TAG}_samples_eval_results.json "$A/climb_${TAG}_samples_eval_results_WSL.json"
RES_SHA=$(sha256sum "$A/climb_${TAG}_samples_eval_results_WSL.json" | cut -d' ' -f1)
python - "$TAG" "$A" "$SAMPLES_SHA" "$RES_SHA" "$SELF_SHA" <<'EOF'
import sys, json, hashlib, platform, evalplus, time
from evalplus.data import get_human_eval_plus_hash
tag, A, ssha, rsha, self_sha = sys.argv[1:6]
res = json.load(open(f"{A}/climb_{tag}_samples_eval_results_WSL.json"))
ev = res["eval"]
base = sum(1 for v in ev.values() if v[0]["base_status"] == "pass")
plus = sum(1 for v in ev.values() if v[0]["base_status"] == "pass" and v[0]["plus_status"] == "pass")
rec = {"kind": "anchor_scoring_receipt", "tag": tag, "scorer": "evalplus.evaluate", "evalplus_version": evalplus.__version__,
       "dataset": "humaneval", "dataset_hash_evalplus": get_human_eval_plus_hash(), "dataset_hash_in_results": res.get("hash"),
       "options": {"dataset": "humaneval", "samples": f"~/{tag}_samples.jsonl", "defaults": True, "parallel": None, "mini": False},
       "host": {"system": platform.system(), "release": platform.release(), "python": platform.python_version(), "wsl": True},
       "samples_sha256": ssha, "results_sha256": rsha, "script_sha256": self_sha, "n_tasks": len(ev),
       "base_pass": base, "plus_pass": plus, "pass_at_1_base": round(base / len(ev), 6), "pass_at_1_plus": round(plus / len(ev), 6),
       "stamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
p = f"{A}/climb_{tag}_scoring_receipt_WSL.json"
b = json.dumps(rec, sort_keys=True, indent=1).encode()
open(p, "wb").write(b)
print("SCORING-RECEIPT", p.split("/")[-1], hashlib.sha256(b).hexdigest()[:16], "evalplus", rec["evalplus_version"], "dataset", rec["dataset_hash_evalplus"][:16], "base", base, "plus", plus)
EOF
echo "results sha ${RES_SHA:0:16}"
