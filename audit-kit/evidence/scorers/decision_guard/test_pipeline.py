"""Actual tiny Qwen2/HF CPU integration; no pretrained checkpoint or benchmark.

This checks the engineering chain, not efficacy on the research Qwen2.5-3B.
"""
import unittest

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM, GenerationConfig

from controller import constrained_step, MarginConstraint, FiniteCheck
from decoder import freeze_root_trace, verify_root_trace, DecoderGuardError
from search import make_margin_callback
from shadow import NativeBF16Shadow


class TinyPipelineTests(unittest.TestCase):
    def make_pipeline(self):
        torch.manual_seed(774)
        config = Qwen2Config(vocab_size=32, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=2, num_attention_heads=2,
                            num_key_value_heads=2, max_position_embeddings=32,
                            bos_token_id=1, eos_token_id=31, pad_token_id=31,
                            attention_dropout=0.)
        native = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16, device='cpu').eval()
        native.generation_config = GenerationConfig(do_sample=False, num_beams=1,
            eos_token_id=[30, 31], pad_token_id=31, repetition_penalty=1.05,
            forced_eos_token_id=31, max_new_tokens=4, use_cache=True)
        parameters = dict(native.named_parameters())
        masters = {}
        for layer in range(2):
            for tl, hf in (('W_in', 'up_proj'), ('W_gate', 'gate_proj'), ('W_out', 'down_proj')):
                source = parameters[f'model.layers.{layer}.mlp.{hf}.weight']
                masters[f'blocks.{layer}.mlp.{tl}'] = torch.nn.Parameter(source.detach().T.float().contiguous().clone())
        shadow = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        prompt = torch.tensor([[1, 3, 5]], dtype=torch.long)
        mask = torch.ones_like(prompt)
        root = freeze_root_trace(shadow, prompt, mask, external_model_identity='tiny-random-qwen2:root', max_new_tokens=4)
        return native, masters, shadow, prompt, mask, root

    def test_search_native_shadow_and_actual_generation_survive_transaction(self):
        native, masters, shadow, prompt, mask, root = self.make_pipeline()
        active = tuple(masters)
        def functional(ids, attention_mask):
            return shadow.functional_forward(active, input_ids=ids, attention_mask=attention_mask,
                                             use_cache=False)
        teach_ids = torch.tensor([[1, 7, 9]], dtype=torch.long)
        teach_mask = torch.ones_like(teach_ids)
        target = torch.tensor([11], dtype=torch.long)
        def teaching():
            output = functional(teach_ids, teach_mask)
            return F.cross_entropy(output.logits[:, -1, :].float(), target)
        def finite_loss():
            output = shadow.forward(input_ids=teach_ids, attention_mask=teach_mask, use_cache=False)
            return float(F.cross_entropy(output.logits[:, -1, :].float(), target))
        margins = []
        for pos in range(min(3, len(root.completion_ids) - 1)):
            callback = make_margin_callback(shadow, root, pos, functional, device='cpu')
            margins.append(MarginConstraint(f'root:{pos}', callback, root.floors[pos]))
        def check():
            try:
                verify_root_trace(shadow, root, prompt, mask,
                                  external_model_identity='tiny-random-qwen2:candidate')
                return FiniteCheck(True, bool(shadow.changed_from_accepted()), 'tiny actual HF generation preserved')
            except DecoderGuardError as exc:
                return FiniteCheck(False, bool(shadow.changed_from_accepted()), str(exc))
        optimizer = torch.optim.AdamW(list(masters.values()), lr=.001, weight_decay=0.)
        before = finite_loss()
        result = shadow.run_step(constrained_step, masters, optimizer, teaching,
                                 finite_teaching_loss=finite_loss,
                                 constraints=margins, finite_check=check)
        # This fixture must exercise an accepted nonzero native update, not just
        # demonstrate that rejecting everything preserves a random model.
        self.assertTrue(result.accepted, result)
        self.assertLess(finite_loss(), before - 1e-6)
        verify_root_trace(shadow, root, prompt, mask,
                          external_model_identity='tiny-random-qwen2:settled')
        self.assertEqual(shadow.changed_from_accepted(), ())


if __name__ == '__main__':
    unittest.main(verbosity=2)
