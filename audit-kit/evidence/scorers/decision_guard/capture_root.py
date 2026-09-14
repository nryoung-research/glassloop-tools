"""Preflight or explicitly capture root traces on the authored coding bank.

No training, benchmark reads, code execution, downloads or service management.
Default mode reads small metadata only. --execute explicitly loads the local
pretrained checkpoint on the selected device and writes a new capture directory.
Captured completions are UNGRADED and cannot yet qualify as correct protected code.
"""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def model_files(model_root):
    weight_files = sorted(p for p in model_root.glob('*.safetensors') if p.is_file())
    if not weight_files:
        raise ValueError('This capture supports local safetensors bundles only')
    index = model_root / 'model.safetensors.index.json'
    if index.exists():
        weight_map = json.loads(index.read_bytes()).get('weight_map')
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError('Invalid or empty weight index')
        for name in weight_map.values():
            if (not isinstance(name, str) or '/' in name or '\\' in name
                    or not name.endswith('.safetensors') or Path(name).name != name
                    or not (model_root / name).is_file()):
                raise ValueError('Invalid or missing indexed weight shard')
    metadata = [p for p in model_root.iterdir() if p.is_file()
                and p.suffix in ('.json', '.txt', '.model', '.jinja')]
    templates = [p for p in (model_root / 'additional_chat_templates').glob('*.jinja') if p.is_file()]
    return sorted(set(weight_files + metadata + templates))


def prepare(model_root, bank_path, expected_bank_sha, output, device, token_cap):
    model_root, bank_path, output = model_root.resolve(), bank_path.resolve(), output.resolve()
    if not model_root.is_dir() or not (model_root / 'config.json').is_file():
        raise ValueError('A local HF model directory containing config.json is required')
    if not 1 <= token_cap <= 512:
        raise ValueError('Root capture cap must be in 1..512')
    if device not in ('cpu', 'cuda:0'):
        raise ValueError('This pilot supports an explicit cpu or cuda:0 device only')
    if output.exists():
        raise ValueError('Capture output must not already exist')
    bank_bytes = bank_path.read_bytes()
    actual = hashlib.sha256(bank_bytes).hexdigest()
    if len(expected_bank_sha) != 64 or actual != expected_bank_sha:
        raise ValueError('Bank bytes differ from the exact supplied SHA256')
    bank = json.loads(bank_bytes)
    if bank.get('kind') != 'decision_guard_coding_bank_v1' or bank.get('version') != 1:
        raise ValueError('Unsupported coding bank schema')
    items = bank['items']
    if bank.get('role') != 'train' or len(items) != bank.get('n_tasks') or not items:
        raise ValueError('Require a complete declared training bank')
    if len({item['task_id'] for item in items}) != len(items):
        raise ValueError('Duplicate task IDs')
    for item in items:
        if item.get('role') != 'train' or hashlib.sha256(item['prompt'].encode()).hexdigest() != item['prompt_sha256']:
            raise ValueError('Bank prompt identity/role mismatch')
        cases = item.get('cases')
        if not isinstance(cases, list) or not cases:
            raise ValueError('Every task requires a nonempty fixture list')
        fixture_bytes = json.dumps(cases, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=True, allow_nan=False).encode('ascii')
        if hashlib.sha256(fixture_bytes).hexdigest() != item.get('fixture_sha256'):
            raise ValueError('Bank fixture identity mismatch')
    if sum(len(item['cases']) for item in items) != bank.get('n_cases'):
        raise ValueError('Bank fixture count mismatch')
    bound_files = model_files(model_root)
    plan = {
        'kind': 'decision_guard_root_capture_preflight',
        'model_root': str(model_root), 'bank': str(bank_path), 'bank_sha256': actual,
        'tasks': len(items), 'task_ids': [item['task_id'] for item in items],
        'device': device, 'max_new_tokens': token_cap, 'output': str(output),
        'bound_model_files': [p.relative_to(model_root).as_posix() for p in bound_files],
        'model_file_bytes': sum(p.stat().st_size for p in bound_files),
        'execution_requested': False,
        'generation_rule': 'Declared greedy copy of inherited HF config; preserve repetition penalty and EOS list; pad=tokenizer EOS.',
        'correctness': 'NOT_MEASURED; no completion is executed by this tool',
        'scope': 'Declared mechanism-bank trace capture only; no VEX training, model selection or benchmark read',
    }
    return plan, items, bound_files


def execute(plan, items, bound_files):
    output = Path(plan['output'])
    if output.exists():
        raise ValueError('Capture output must not already exist')
    if sha(Path(plan['bank'])) != plan['bank_sha256']:
        raise ValueError('Bank changed after preflight; execution refused')
    model_root = Path(plan['model_root'])
    if model_files(model_root) != bound_files:
        raise ValueError('Model file roster changed after preflight; execution refused')
    # Delayed imports: the default preflight has no torch/transformers/model work.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import transformers
    from decoder import freeze_root_trace, DecoderGuardError

    output.mkdir(parents=True, exist_ok=False)
    manifest_path = output / 'capture_manifest.json'
    hashes = {p.relative_to(model_root).as_posix(): sha(p) for p in bound_files}
    identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    manifest = dict(plan, execution_requested=True, status='INCOMPLETE',
                    source_sha256=sha(Path(__file__)), model_files_sha256=hashes,
                    model_manifest_sha256=identity, python=platform.python_version(),
                    torch=torch.__version__, transformers=transformers.__version__,
                    started=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    model = AutoModelForCausalLM.from_pretrained(plan['model_root'], local_files_only=True,
                                               dtype=torch.bfloat16).to(plan['device']).eval()
    tokenizer = AutoTokenizer.from_pretrained(plan['model_root'], local_files_only=True)
    config = copy.deepcopy(model.generation_config)
    config.do_sample = False
    config.pad_token_id = tokenizer.eos_token_id
    manifest['attention_implementation'] = getattr(model.config, '_attn_implementation', None)
    manifest['model_parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
    complete, failed = 0, 0
    with (output / 'root_traces.jsonl').open('x', encoding='utf-8') as stream:
        for index, item in enumerate(items):
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': item['prompt']}],
                tokenize=True, add_generation_prompt=True, return_tensors='pt')
            ids = rendered.input_ids if hasattr(rendered, 'input_ids') else rendered
            if isinstance(ids, dict):
                ids = ids['input_ids']
            ids = ids.to(plan['device'])
            row = {'task_id': item['task_id'], 'prompt_sha256': item['prompt_sha256'],
                   'fixture_sha256': item['fixture_sha256'], 'correctness': 'NOT_MEASURED'}
            try:
                trace = freeze_root_trace(model, ids, torch.ones_like(ids),
                    external_model_identity=identity, max_new_tokens=plan['max_new_tokens'],
                    generation_config=config)
                if not all(math.isfinite(m) for m in trace.margins):
                    raise DecoderGuardError('Nonfinite margins cannot be serialized as this numeric trace schema')
                row.update(status='CAPTURED_UNGRADED', trace=asdict(trace),
                           completion_text=tokenizer.decode(trace.completion_ids, skip_special_tokens=True))
                complete += 1
            except DecoderGuardError as exc:
                row.update(status='INELIGIBLE_OR_INCOMPLETE', reason=str(exc))
                failed += 1
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            print(json.dumps({'completed_tasks': index + 1, 'captured_ungraded': complete, 'ineligible': failed}), flush=True)
    unchanged = model_files(model_root) == bound_files and all(
        sha(path) == hashes[path.relative_to(model_root).as_posix()] for path in bound_files)
    bank_unchanged = sha(Path(plan['bank'])) == plan['bank_sha256']
    manifest.update(status=('CAPTURED_UNGRADED' if failed == 0 else 'PARTIAL_UNGRADED')
                    if unchanged and bank_unchanged else 'INVALID_INPUT_DRIFT',
                    captured_ungraded=complete, ineligible=failed,
                    model_files_unchanged=unchanged, bank_unchanged=bank_unchanged,
                    root_traces_sha256=sha(output / 'root_traces.jsonl'),
                    finished=time.strftime('%Y-%m-%dT%H:%M:%S%z'))
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-root', type=Path, required=True)
    parser.add_argument('--bank', type=Path, required=True)
    parser.add_argument('--expected-bank-sha', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda:0'], required=True)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    plan, items, files = prepare(args.model_root, args.bank, args.expected_bank_sha,
                                args.output, args.device, args.max_new_tokens)
    print(json.dumps(plan, indent=2))
    if args.execute:
        print(json.dumps(execute(plan, items, files), indent=2))


if __name__ == '__main__':
    main()
