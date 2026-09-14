"""CPU tensor fixtures and an optional random tiny HF model; no checkpoint I/O."""

import copy
import os
import unittest
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import torch

from decoder import DecoderGuardError, freeze_root_trace, verify_root_trace


class FakeConfig:
    def __init__(self, **kwargs):
        self.do_sample = False
        self.num_beams = 1
        self.num_return_sequences = 1
        self.eos_token_id = [3, 4]
        self.repetition_penalty = 1.05
        self.__dict__.update(kwargs)

    def to_dict(self):
        return copy.deepcopy(self.__dict__)


class FakeModel:
    training = False
    dtype = torch.float32

    def __init__(self, completion=(2, 4), scores=None, **config):
        self.config = SimpleNamespace(vocab_size=5, is_encoder_decoder=False)
        self.generation_config = FakeConfig(**config)
        self.completion = completion
        self.scores = scores if scores is not None else (
            torch.tensor([[0.0, 0.1, 2.0, 0.2, 0.3]]),
            torch.tensor([[0.0, 0.1, 0.2, 0.3, 2.0]]),
        )
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            sequences=torch.cat((kwargs["input_ids"], torch.tensor([self.completion])), dim=1),
            scores=self.scores,
        )


class DecoderTests(unittest.TestCase):
    def setUp(self):
        self.prompt = torch.tensor([[0, 1]])

    def freeze(self, model, **kwargs):
        return freeze_root_trace(model, self.prompt, external_model_identity="fixture:root", max_new_tokens=4, **kwargs)

    def test_complete_scores_and_second_eos_are_preserved(self):
        model = FakeModel()
        root = self.freeze(model)
        self.assertEqual(root.completion_ids, (2, 4))
        self.assertEqual(root.eos_token_ids, (3, 4))
        self.assertEqual(root.termination_reason, "eos:4")
        self.assertEqual(root.floors, (0.01, 0.01))
        self.assertEqual(root.retained_score_bytes, 40)
        call = model.calls[0]
        self.assertTrue(call["generation_config"].output_scores)
        self.assertTrue(call["generation_config"].return_dict_in_generate)
        self.assertEqual(call["generation_config"].repetition_penalty, 1.05)
        self.assertIsNot(call["generation_config"], model.generation_config)
        check = verify_root_trace(model, root, self.prompt, external_model_identity="fixture:candidate")
        self.assertGreater(check.minimum_margin_surplus, 0)
        self.assertEqual(check.root_external_model_identity, "fixture:root")

    def test_first_eos_supported(self):
        root = self.freeze(FakeModel(completion=(3,), scores=(torch.tensor([[0., 0., 0., 2., 1.]]),)))
        self.assertEqual(root.termination_reason, "eos:3")

    def test_sampling_beams_and_missing_score_vocabulary_fail_preflight(self):
        for settings in ({"do_sample": True}, {"num_beams": 2}, {"num_return_sequences": 2}):
            model = FakeModel(**settings)
            with self.assertRaises(DecoderGuardError):
                self.freeze(model)
            self.assertFalse(model.calls)
        model = FakeModel()
        model.config.vocab_size = None
        with self.assertRaises(DecoderGuardError):
            self.freeze(model)

    def test_missing_scores_and_incomplete_trace_fail(self):
        for model in (FakeModel(scores=()), FakeModel(scores=(torch.zeros(1, 5),)), FakeModel(completion=(2, 1))):
            with self.assertRaises(DecoderGuardError):
                self.freeze(model)

    def test_nonwinner_tie_nan_and_truncated_vocabulary_fail(self):
        for bad in (
            torch.tensor([[0., 3., 2., 0., 0.]]),
            torch.tensor([[0., 2., 2., 0., 0.]]),
            torch.tensor([[float("nan"), 0., 2., 0., 0.]]),
            torch.tensor([[float("inf"), 0., 2., 0., 0.]]),
            torch.tensor([[0., 0., 2., 0.]]),
        ):
            with self.subTest(score=bad):
                with self.assertRaises(DecoderGuardError):
                    self.freeze(FakeModel(scores=(bad, torch.tensor([[0., 0., 0., 0., 2.]]))))

    def test_negative_infinity_masks_are_valid(self):
        model = FakeModel(completion=(4,), scores=(torch.tensor([[float("-inf")] * 4 + [0.]]),))
        root = self.freeze(model)
        self.assertEqual(root.floors, (0.01,))
        self.assertEqual(root.margins, (float("inf"),))

    def test_byte_budget_fails_before_generate(self):
        model = FakeModel()
        with self.assertRaises(DecoderGuardError):
            self.freeze(model, max_score_bytes=79)
        self.assertFalse(model.calls)

    def test_frozen_floor_and_full_vocabulary_rival(self):
        root = self.freeze(FakeModel())
        model = FakeModel(scores=(torch.tensor([[0., 0.1, 2., 0.2, 1.995]]), torch.tensor([[0., 0., 0., 0., 2.]])))
        with self.assertRaisesRegex(DecoderGuardError, "floor"):
            verify_root_trace(model, root, self.prompt, external_model_identity="fixture:candidate")

    def test_config_and_raw_prompt_drift_fail(self):
        root = self.freeze(FakeModel())
        with self.assertRaisesRegex(DecoderGuardError, "GenerationConfig"):
            verify_root_trace(FakeModel(repetition_penalty=1.0), root, self.prompt, external_model_identity="fixture:candidate")
        with self.assertRaisesRegex(DecoderGuardError, "raw prompt"):
            verify_root_trace(FakeModel(), root, torch.tensor([[1, 0]]), external_model_identity="fixture:candidate")

    def test_exact_completion_mismatch_fails(self):
        root = self.freeze(FakeModel())
        model = FakeModel(completion=(1, 4), scores=(torch.tensor([[0., 2., 1., 0., 0.]]), torch.tensor([[0., 0., 0., 0., 2.]])))
        with self.assertRaisesRegex(DecoderGuardError, "completion IDs"):
            verify_root_trace(model, root, self.prompt, external_model_identity="fixture:candidate")

    def test_tokens_after_eos_and_unsupported_stops_fail(self):
        for model in (FakeModel(completion=(3, 4)), FakeModel(stop_strings=["STOP"]), FakeModel(max_time=1.0)):
            with self.assertRaises(DecoderGuardError):
                self.freeze(model)

    def test_root_floors_are_capped_fraction_not_confidence_copy(self):
        model = FakeModel(completion=(4,), scores=(torch.tensor([[0., 0., 0., 0.999, 1.]]),))
        root = self.freeze(model)
        self.assertAlmostEqual(root.floors[0], 0.0001, places=7)


class TinyTransformersTests(unittest.TestCase):
    def test_real_generate_processed_scores_include_repetition_penalty(self):
        try:
            from transformers import GPT2Config, GPT2LMHeadModel
        except ImportError as exc:
            self.skipTest(f"transformers tiny causal-LM import unavailable: {exc}")
        # Random CPU model, constructed from config. No from_pretrained/download.
        torch.manual_seed(20260908)
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=13, n_positions=16, n_ctx=16, n_embd=8, n_layer=1,
            n_head=1, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
            bos_token_id=1, eos_token_id=12, pad_token_id=0,
        )).cpu().eval()
        # An output bias constructs a stable, explicit winner-order reversal:
        # repeated token 1 wins raw (4), unseen token 2 wins after penalty 2 (3).
        model.lm_head.bias = torch.nn.Parameter(torch.zeros(13))
        with torch.no_grad():
            model.lm_head.weight.zero_()
            model.lm_head.bias[1] = 4.
            model.lm_head.bias[2] = 3.
            model.lm_head.bias[12] = 2.5
        model.generation_config.do_sample = False
        model.generation_config.num_beams = 1
        model.generation_config.eos_token_id = [12, 11]
        model.generation_config.pad_token_id = 0
        model.generation_config.repetition_penalty = 2.
        prompt = torch.tensor([[1]])
        with torch.inference_mode():
            raw = model(input_ids=prompt).logits[0, -1]
            generated = model.generate(input_ids=prompt, max_new_tokens=4, output_scores=True, return_dict_in_generate=True)
        self.assertEqual(int(raw.argmax()), 1)
        self.assertEqual(int(generated.scores[0].argmax()), 2)
        self.assertEqual(generated.sequences[0].tolist(), [1, 2, 12])
        root = freeze_root_trace(model, prompt, external_model_identity="random-cpu-fixture", max_new_tokens=4)
        self.assertEqual(root.completion_ids, (2, 12))
        self.assertAlmostEqual(root.margins[0], 0.5, places=6)
        verify_root_trace(model, root, prompt, external_model_identity="same-random-cpu-fixture")


if __name__ == "__main__":
    unittest.main()
