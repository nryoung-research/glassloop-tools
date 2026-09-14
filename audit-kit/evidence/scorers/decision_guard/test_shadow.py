"""Real CPU tensors in a tiny HF-shaped module; no model/checkpoint loads."""
from types import SimpleNamespace
import unittest

import torch

from shadow import NativeBF16Shadow, ShadowScopeError, hf_name
from transaction import TensorTransaction


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
        self.gate_proj = torch.nn.Linear(2, 3, bias=False, dtype=torch.bfloat16)
        self.down_proj = torch.nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class TinyLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()


class TinyNative(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([TinyLayer(), TinyLayer()])
        self.register_buffer("tiny_buffer", torch.tensor(0.125, dtype=torch.bfloat16))
        self.config = SimpleNamespace(vocab_size=2)
        self.generate_calls = 0

    def forward(self, x, use_cache=False):
        for layer in self.model.layers:
            x = x + layer.mlp(x)
        return x + self.tiny_buffer

    def generate(self, x, **kwargs):
        self.generate_calls += 1
        return self.forward(x, **kwargs)


def fixture():
    native = TinyNative()
    masters = {}
    for layer in range(2):
        for index, name in enumerate(("W_in", "W_gate", "W_out")):
            shape = (2, 3) if name != "W_out" else (3, 2)
            values = torch.arange(1, 7, dtype=torch.float32).reshape(shape) / 16
            masters["blocks.%d.mlp.%s" % (layer, name)] = torch.nn.Parameter(values + layer / 32 + index / 64)
    return native, masters


class ShadowTests(unittest.TestCase):
    def test_transposes_and_syncs_every_declared_inherited_matrix(self):
        native, masters = fixture()
        bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        hf = dict(native.named_parameters())
        for name, master in masters.items():
            self.assertTrue(torch.equal(hf[hf_name(name)], master.T.to(torch.bfloat16)))
        inherited = "blocks.1.mlp.W_out"
        with torch.no_grad():
            masters[inherited].add_(0.25)
        changed = bridge.sync_from_masters()
        self.assertEqual(changed, (inherited,))
        self.assertFalse(native.training)
        self.assertTrue(all(not p.requires_grad for p in native.parameters()))

    def test_bf16_noop_uses_previous_accepted_native_values(self):
        native, masters = fixture()
        name = "blocks.0.mlp.W_in"
        with torch.no_grad():
            masters[name].fill_(1.0)
        bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        with torch.no_grad():
            masters[name].add_(1e-4)
        self.assertEqual(bridge.changed_from_accepted(), ())
        with torch.no_grad():
            masters[name].add_(0.02)
        self.assertEqual(bridge.changed_from_accepted(), (name,))
        bridge.mark_accepted()
        self.assertEqual(bridge.changed_from_accepted(), ())

    def test_functional_search_connects_only_active_masters(self):
        native, masters = fixture()
        bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        active = "blocks.0.mlp.W_in"
        output = bridge.functional_forward((active,), torch.tensor([[0.5, -0.25]], dtype=torch.bfloat16))
        self.assertTrue(output.requires_grad)
        gradients = torch.autograd.grad(output.float().sum(), tuple(masters.values()), allow_unused=True)
        for name, gradient in zip(masters, gradients):
            if name == active:
                self.assertIsNotNone(gradient)
                self.assertEqual(gradient.dtype, torch.float32)
                self.assertGreater(float(gradient.abs().sum()), 0.0)
            else:
                self.assertIsNone(gradient)
        self.assertTrue(all(parameter.grad is None and not parameter.requires_grad for parameter in native.parameters()))
        self.assertFalse(bridge(torch.tensor([[0.5, -0.25]], dtype=torch.bfloat16)).requires_grad)

    def test_ordinary_forward_and_generate_resync_native_weights(self):
        native, masters = fixture()
        bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        x = torch.tensor([[0.5, 0.25]], dtype=torch.bfloat16)
        old = bridge.forward(x)
        with torch.no_grad():
            masters["blocks.1.mlp.W_out"].add_(0.5)
        candidate = bridge.generate(x)
        self.assertEqual(native.generate_calls, 1)
        self.assertFalse(torch.equal(candidate, old))
        self.assertTrue(torch.equal(candidate, native(x)))
        self.assertFalse(candidate.requires_grad)

    def test_unchanged_finite_baseline_does_not_invalidate_search_graph(self):
        native, masters = fixture()
        bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        active = "blocks.0.mlp.W_in"
        x = torch.tensor([[0.5, -0.25]], dtype=torch.bfloat16)
        search = bridge.functional_forward((active,), x)
        bridge.forward(x)  # Controller measures native baseline before autograd.
        gradient, = torch.autograd.grad(search.float().sum(), (masters[active],))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_missing_extra_and_unsupported_master_names_refuse(self):
        for issue in ("missing", "extra", "bias"):
            with self.subTest(issue=issue):
                native, masters = fixture()
                declaration = tuple(masters)
                if issue == "missing":
                    del masters[declaration[-1]]
                elif issue == "extra":
                    declaration = declaration[:-1]
                else:
                    masters["blocks.0.mlp.b_in"] = torch.zeros(3)
                    declaration = tuple(masters)
                with self.assertRaises(ShadowScopeError):
                    NativeBF16Shadow(native, masters, declared_master_names=declaration, torch_module=torch)
        native, masters = fixture()
        duplicate = list(masters.items()) + [next(iter(masters.items()))]
        with self.assertRaises(ShadowScopeError):
            NativeBF16Shadow(native, duplicate, declared_master_names=tuple(masters), torch_module=torch)

    def test_missing_native_matrix_or_wrong_dtype_shape_refuse(self):
        for issue in ("native_missing", "native_dtype", "master_dtype", "shape"):
            with self.subTest(issue=issue):
                native, masters = fixture()
                key = "blocks.0.mlp.W_in"
                if issue == "native_missing":
                    del native.model.layers[0].mlp.up_proj
                elif issue == "native_dtype":
                    native.float()
                elif issue == "master_dtype":
                    masters[key] = torch.nn.Parameter(masters[key].to(torch.bfloat16))
                else:
                    masters[key] = torch.nn.Parameter(torch.zeros(4, 3))
                with self.assertRaises(ShadowScopeError):
                    NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)

    def test_validation_precedes_any_partial_sync(self):
        native, masters = fixture()
        names = tuple(masters)
        bridge = NativeBF16Shadow(native, masters, declared_master_names=names, torch_module=torch)
        first = dict(native.named_parameters())[hf_name(names[0])]
        before = first.clone()
        with torch.no_grad():
            masters[names[0]].add_(1)
        del masters[names[-1]]
        with self.assertRaises(ShadowScopeError):
            bridge.sync_from_masters()
        self.assertTrue(torch.equal(first, before))

    def test_rejection_and_exception_resync_after_master_transaction(self):
        for explode in (False, True):
            with self.subTest(explode=explode):
                native, masters = fixture()
                caches = {"old": object()}
                bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch, external_cache_refs=caches)
                name = "blocks.0.mlp.W_in"
                parameter = masters[name]
                parameter.grad = torch.ones_like(parameter)
                optimizer = torch.optim.SGD([parameter], lr=0.5)
                native_before = dict(native.named_parameters())[hf_name(name)].clone()

                def controller():
                    with TensorTransaction([(name, parameter)], optimizer, torch_module=torch):
                        optimizer.step()
                        bridge.sync_from_masters()
                        self.assertFalse(torch.equal(dict(native.named_parameters())[hf_name(name)], native_before))
                        caches["candidate"] = object()
                        if explode:
                            raise RuntimeError("candidate evaluator failed")
                        return SimpleNamespace(accepted=False)

                if explode:
                    with self.assertRaisesRegex(RuntimeError, "candidate evaluator failed"):
                        bridge.run_step(controller)
                else:
                    self.assertFalse(bridge.run_step(controller).accepted)
                self.assertTrue(torch.equal(dict(native.named_parameters())[hf_name(name)], native_before))
                self.assertEqual(bridge.changed_from_accepted(), ())
                self.assertFalse(caches)

    def test_accepted_controller_advances_only_previous_accepted_snapshot(self):
        native, masters = fixture()
        bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        name = "blocks.0.mlp.W_in"
        p = masters[name]
        p.grad = torch.ones_like(p)
        optimizer = torch.optim.SGD([p], lr=0.1)
        def controller():
            with TensorTransaction([(name, p)], optimizer, torch_module=torch) as tx:
                optimizer.step()
                self.assertTrue(bridge.changed_from_accepted())
                tx.commit()
                return SimpleNamespace(accepted=True)
        self.assertTrue(bridge.run_step(controller).accepted)
        self.assertEqual(bridge.changed_from_accepted(), ())

    def test_external_cache_and_disconnected_active_scope_refuse(self):
        native, masters = fixture()
        bridge = NativeBF16Shadow(native, masters, declared_master_names=tuple(masters), torch_module=torch)
        x = torch.ones(1, 2, dtype=torch.bfloat16)
        with self.assertRaises(ShadowScopeError):
            bridge.generate(x, past_key_values=object())
        with self.assertRaises(ShadowScopeError):
            bridge.functional_forward(tuple(masters)[:1], x, use_cache=True)
        with self.assertRaises(ShadowScopeError):
            bridge.functional_forward(("blocks.3.mlp.W_in",), x)
        name = tuple(masters)[0]
        masters[name].requires_grad_(False)
        with self.assertRaises(ShadowScopeError):
            bridge.functional_forward((name,), x)


if __name__ == "__main__":
    unittest.main(verbosity=2)
