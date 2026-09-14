"""CPU-only connected tensor tests of the installed HF processor chain."""

import os
import unittest

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch

from decoder import GreedyTrace, _configuration
from search import SearchMarginError, decision_prefix, make_margin_callback


class SearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as exc:
            raise unittest.SkipTest(f"transformers tiny model unavailable: {exc}")
        cls.model = GPT2LMHeadModel(GPT2Config(
            vocab_size=13, n_positions=16, n_ctx=16, n_embd=8, n_layer=1,
            n_head=1, bos_token_id=1, eos_token_id=12, pad_token_id=0,
        )).cpu().eval()

    def setUp(self):
        from transformers import GenerationConfig
        self.model.generation_config = GenerationConfig(
            do_sample=False, num_beams=1, num_return_sequences=1,
            eos_token_id=[12, 11], pad_token_id=0, repetition_penalty=1.05,
        )

    def root(self, **config):
        for key, value in config.items():
            setattr(self.model.generation_config, key, value)
        _, signature, eos, _ = _configuration(self.model, None, 4)
        return GreedyTrace((1,), (1,), (2, 12), (.1, .1), (.01, .01), eos,
                           "eos:12", signature, 4, 0, "synthetic-search-fixture")

    def callback(self, root, vector, position=0, **kwargs):
        def forward(prefix, mask):
            self.assertEqual(prefix.dtype, torch.long)
            self.assertEqual(mask.shape, prefix.shape)
            return vector.unsqueeze(0)
        return make_margin_callback(self.model, root, position, forward, **kwargs)

    def test_repetition_105_changes_winner_and_gradient_is_connected(self):
        root = self.root()
        values = torch.zeros(13)
        values[1], values[2] = 1.04, 1.0
        masters = torch.nn.Parameter(values)
        margin = self.callback(root, masters)()
        self.assertEqual(int(masters.argmax()), 1)
        self.assertAlmostEqual(float(margin.detach()), 1.0 - 1.04 / 1.05, places=6)
        self.assertGreater(float(margin.detach()), 0.)
        gradient = torch.autograd.grad(margin, masters)[0]
        self.assertAlmostEqual(float(gradient[2]), 1.0, places=6)
        self.assertAlmostEqual(float(gradient[1]), -1.0 / 1.05, places=6)

    def test_current_full_vocabulary_rival_is_used(self):
        root = self.root(repetition_penalty=1.)
        masters = torch.nn.Parameter(torch.zeros(13))
        with torch.no_grad():
            masters[2], masters[9] = 1., 1.5
        margin = self.callback(root, masters)()
        self.assertAlmostEqual(float(margin.detach()), -.5)
        self.assertEqual(float(torch.autograd.grad(margin, masters)[0][9]), -1.)

    def test_masked_rivals_use_actual_hf_suppression(self):
        root = self.root(suppress_tokens=[1], repetition_penalty=1.)
        masters = torch.nn.Parameter(torch.zeros(13))
        with torch.no_grad():
            masters[1], masters[2], masters[5] = 100., 2., .5
        margin = self.callback(root, masters)()
        self.assertAlmostEqual(float(margin.detach()), 1.5)
        self.assertEqual(float(torch.autograd.grad(margin, masters)[0][1]), 0.)

    def test_prefix_contains_root_tokens_and_original_prompt_boundary(self):
        root = self.root(min_new_tokens=2)
        decision = decision_prefix(root, 1)
        self.assertEqual(decision.ids, (1, 2))
        self.assertEqual(decision.attention_mask, (1, 1))
        self.assertEqual(decision.expected_token_id, 12)
        self.assertEqual(decision.original_prompt_length, 1)
        masters = torch.nn.Parameter(torch.zeros(13))
        with self.assertRaisesRegex(SearchMarginError, "expected root token is masked"):
            self.callback(root, masters, position=1)()

    def test_observed_native_value_keeps_surrogate_derivative(self):
        root = self.root(repetition_penalty=1.)
        masters = torch.nn.Parameter(torch.arange(13, dtype=torch.float32))
        with torch.no_grad():
            masters[12] = 100000000.
        margin = self.callback(root, masters, observed_margin=.125)()
        self.assertAlmostEqual(float(margin.detach()), .125)
        self.assertEqual(float(torch.autograd.grad(margin, masters)[0][2]), 1.)

    def test_config_drift_and_sampling_fail(self):
        root = self.root()
        masters = torch.nn.Parameter(torch.zeros(13))
        callback = self.callback(root, masters)
        self.model.generation_config.repetition_penalty = 1.
        with self.assertRaisesRegex(SearchMarginError, "differs"):
            callback()
        self.model.generation_config.do_sample = True
        with self.assertRaisesRegex(SearchMarginError, "sampling"):
            callback()

    def test_detached_forward_fails(self):
        root = self.root()
        with self.assertRaisesRegex(SearchMarginError, "detached"):
            self.callback(root, torch.zeros(13))()

    def test_guidance_fails_closed(self):
        root = self.root(guidance_scale=2.)
        with self.assertRaisesRegex(SearchMarginError, "guidance"):
            self.callback(root, torch.nn.Parameter(torch.zeros(13)))()

    def test_eos_masks_use_both_configured_ids(self):
        root = self.root(min_new_tokens=1, repetition_penalty=1.)
        masters = torch.nn.Parameter(torch.zeros(13))
        with torch.no_grad():
            masters[2], masters[11], masters[12] = 1., 10., 20.
        margin = self.callback(root, masters)()
        self.assertAlmostEqual(float(margin.detach()), 1.)


if __name__ == "__main__":
    unittest.main()
