"""Metadata-only capture preflight tests on tiny fake checkpoint files.

No torch/transformers import, pretrained checkpoint, model, GPU, code submission,
or valid capture execution is permitted. Calls to execute exercise refusal paths
only, under an import guard that fails before any model runtime can be imported.
"""
import builtins
from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import capture_root


HERE = Path(__file__).resolve().parent


@contextmanager
def no_model_imports():
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('torch', 'transformers', 'decoder'):
            raise AssertionError('A metadata/refusal path attempted model runtime import: ' + name)
        return original(name, *args, **kwargs)
    with patch('builtins.__import__', side_effect=guarded):
        yield


class CapturePreflightTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='decision_guard_capture_test_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = self.root / 'fake_model'
        self.model.mkdir()
        (self.model / 'config.json').write_text('{}', encoding='utf-8')
        # Deliberately NOT valid weights: only the existence/metadata is tested.
        (self.model / 'model.safetensors').write_bytes(b'FAKE: NEVER LOAD')
        self.bank = self.root / 'bank.json'
        self.payload = json.loads((HERE / 'coding_bank.json').read_text(encoding='utf-8'))
        self.write_bank()
        self.output = self.root / 'capture_output'

    def write_bank(self):
        self.bank.write_text(json.dumps(self.payload, sort_keys=True), encoding='utf-8')
        self.bank_sha = hashlib.sha256(self.bank.read_bytes()).hexdigest()

    def prepare(self, **changes):
        args = dict(model_root=self.model, bank_path=self.bank, expected_bank_sha=self.bank_sha,
                    output=self.output, device='cpu', token_cap=64)
        args.update(changes)
        with no_model_imports():
            return capture_root.prepare(**args)

    def test_module_import_is_metadata_only(self):
        spec = importlib.util.spec_from_file_location('_capture_root_import_test', HERE / 'capture_root.py')
        module = importlib.util.module_from_spec(spec)
        with no_model_imports():
            spec.loader.exec_module(module)
        self.assertTrue(callable(module.prepare))

    def test_default_cli_never_reads_fake_weight_bytes_or_imports_runtime(self):
        original_open = Path.open
        def metadata_open(path, *args, **kwargs):
            if path.suffix == '.safetensors':
                raise AssertionError('Preflight read weight bytes')
            return original_open(path, *args, **kwargs)
        argv = ['capture_root.py', '--model-root', str(self.model), '--bank', str(self.bank),
                '--expected-bank-sha', self.bank_sha, '--output', str(self.output),
                '--device', 'cpu', '--max-new-tokens', '64']
        stream = io.StringIO()
        with patch.object(Path, 'open', new=metadata_open), patch.object(sys, 'argv', argv), no_model_imports(), redirect_stdout(stream):
            capture_root.main()
        plan = json.loads(stream.getvalue())
        self.assertFalse(plan['execution_requested'])
        self.assertEqual(plan['tasks'], 16)
        self.assertEqual(plan['bank_sha256'], self.bank_sha)
        self.assertIn('NOT_MEASURED', plan['correctness'])
        self.assertFalse(self.output.exists())

    def test_hash_and_parse_use_one_bank_read(self):
        original = Path.read_bytes
        reads = []
        def counted(path):
            if path.resolve() == self.bank.resolve():
                reads.append(path)
            return original(path)
        with patch.object(Path, 'read_bytes', new=counted):
            plan, items, files = self.prepare()
        self.assertEqual(len(reads), 1)
        self.assertEqual(len(items), 16)

    def test_expected_bank_hash_must_match_exact_bytes(self):
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.prepare(expected_bank_sha='0' * 64)

    def test_prompt_drift_rejected_even_with_new_outer_hash(self):
        self.payload['items'][0]['prompt'] += '\nChanged prompt.'
        self.write_bank()
        with self.assertRaisesRegex(ValueError, 'prompt identity'):
            self.prepare()

    def test_fixture_drift_rejected_even_with_new_outer_hash(self):
        self.payload['items'][0]['cases'][0]['expected'] = ['not the original expected value']
        self.write_bank()
        with self.assertRaisesRegex(ValueError, 'fixture identity'):
            self.prepare()

    def test_declared_fixture_count_is_validated(self):
        self.payload['n_cases'] += 1
        self.write_bank()
        with self.assertRaisesRegex(ValueError, 'fixture count'):
            self.prepare()

    def test_empty_fixture_list_is_refused(self):
        self.payload['items'][0]['cases'] = []
        self.payload['items'][0]['fixture_sha256'] = hashlib.sha256(b'[]').hexdigest()
        self.write_bank()
        with self.assertRaisesRegex(ValueError, 'nonempty fixture'):
            self.prepare()

    def test_duplicate_ids_and_bad_roles_or_schema_are_refused(self):
        original = copy.deepcopy(self.payload)
        edits = [lambda p: p['items'][1].update(task_id=p['items'][0]['task_id']),
                 lambda p: p.update(role='exam'),
                 lambda p: p['items'][0].update(role='monitor'),
                 lambda p: p.update(kind='other_bank'),
                 lambda p: p.update(version=99),
                 lambda p: p.update(n_tasks=17)]
        for edit in edits:
            self.payload = copy.deepcopy(original)
            edit(self.payload)
            self.write_bank()
            with self.subTest(payload_kind=self.payload['kind'], role=self.payload['role']):
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_existing_output_refused_without_changing_it(self):
        self.output.mkdir()
        sentinel = self.output / 'keep.txt'
        sentinel.write_text('unchanged', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'already exist'):
            self.prepare()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'unchanged')

    def test_root_and_named_chat_templates_are_bound_by_relative_name(self):
        (self.model / 'chat_template.jinja').write_text('root template', encoding='utf-8')
        nested = self.model / 'additional_chat_templates'
        nested.mkdir()
        (nested / 'chat_template.jinja').write_text('named template', encoding='utf-8')
        plan, _, _ = self.prepare()
        self.assertIn('chat_template.jinja', plan['bound_model_files'])
        self.assertIn('additional_chat_templates/chat_template.jinja', plan['bound_model_files'])
        self.assertEqual(len(plan['bound_model_files']), len(set(plan['bound_model_files'])))

    def test_indexed_shards_are_in_the_bound_roster(self):
        (self.model / 'model-00001.safetensors').write_bytes(b'ALSO FAKE')
        (self.model / 'model.safetensors.index.json').write_text(
            json.dumps({'weight_map': {'a': 'model.safetensors', 'b': 'model-00001.safetensors'}}), encoding='utf-8')
        plan, _, _ = self.prepare()
        self.assertIn('model-00001.safetensors', plan['bound_model_files'])
        self.assertIn('model.safetensors.index.json', plan['bound_model_files'])

    def test_invalid_index_names_maps_and_missing_shards_are_refused(self):
        index = self.model / 'model.safetensors.index.json'
        for weight_map in ({}, [], {'a': '../outside.safetensors'}, {'a': 'nested\\weight.safetensors'},
                           {'a': 'missing.safetensors'}, {'a': 'weights.bin'}, {'a': 1}):
            index.write_text(json.dumps({'weight_map': weight_map}), encoding='utf-8')
            with self.subTest(weight_map=weight_map):
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_directory_named_like_weights_is_not_a_bundle(self):
        weight = self.model / 'model.safetensors'
        weight.unlink()
        weight.mkdir()
        with self.assertRaisesRegex(ValueError, 'safetensors'):
            self.prepare()

    def test_changed_bank_refuses_before_import_or_output_creation(self):
        plan, items, files = self.prepare()
        self.bank.write_bytes(self.bank.read_bytes() + b'\n')
        with no_model_imports(), self.assertRaisesRegex(ValueError, 'Bank changed'):
            capture_root.execute(plan, items, files)
        self.assertFalse(self.output.exists())

    def test_added_template_refuses_before_import_or_output_creation(self):
        plan, items, files = self.prepare()
        (self.model / 'chat_template.jinja').write_text('new loader input', encoding='utf-8')
        with no_model_imports(), self.assertRaisesRegex(ValueError, 'roster changed'):
            capture_root.execute(plan, items, files)
        self.assertFalse(self.output.exists())

    def test_late_output_creation_refuses_before_runtime_import(self):
        plan, items, files = self.prepare()
        self.output.mkdir()
        with no_model_imports(), self.assertRaisesRegex(ValueError, 'already exist'):
            capture_root.execute(plan, items, files)


if __name__ == '__main__':
    unittest.main(verbosity=2)
