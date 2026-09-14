"""Run local component tests and bind results to source hashes; never load checkpoints."""
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import unittest


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('This validation requires CUDA_VISIBLE_DEVICES to be explicitly empty')
    import torch
    import transformers
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA was initialized before CPU validation')
    directory = Path(__file__).resolve().parent
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in sorted(directory.glob('*.py'))}
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(str(directory), pattern='test_*.py')
    with (directory / 'cpu_validation.log').open('w', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    still_same = all(hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
                     for name, digest in sources.items())
    initialized = torch.cuda.is_initialized()
    receipt = {
        'kind': 'decision_guard_cpu_component_validation',
        'status': 'PASS' if result.wasSuccessful() and not result.skipped and still_same and not initialized else 'FAIL',
        'tests_run': result.testsRun, 'failures': len(result.failures),
        'errors': len(result.errors), 'skipped': len(result.skipped),
        'elapsed_seconds': round(time.perf_counter() - started, 6),
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__,
        'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'cuda_initialized_after': initialized,
        'pretrained_checkpoints_loaded_by_tests': False,
        'fixture_scope': 'Tiny synthetic CPU tensors, a config-initialized GPT2 fixture, and a config-initialized tiny Qwen2 BF16 pipeline. No 3B efficacy, benchmark or GPU result.',
        'source_hashes_unchanged': still_same, 'source_sha256': sources,
    }
    (directory / 'cpu_validation.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'source_sha256'}, indent=2))
    return 0 if receipt['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
