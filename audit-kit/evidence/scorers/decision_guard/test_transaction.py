"""Tiny real CPU tensor/optimizer tests. No model, GPU or model training."""
import copy
import importlib.util
import random
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

from transaction import ScopeViolation, TensorTransaction


def assert_tree_equal(test, actual, expected):
    if isinstance(expected, torch.Tensor):
        test.assertIsInstance(actual, torch.Tensor)
        test.assertEqual((actual.shape, actual.dtype, actual.device), (expected.shape, expected.dtype, expected.device))
        test.assertTrue(torch.equal(actual, expected))
    elif isinstance(expected, dict):
        test.assertEqual(set(actual), set(expected))
        for key in expected:
            assert_tree_equal(test, actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        test.assertEqual(type(actual), type(expected))
        test.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            assert_tree_equal(test, a, e)
    else:
        test.assertEqual(actual, expected)


def make_optimizer(kind="adamw", initialize=True):
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    q = torch.nn.Parameter(torch.tensor([[0.5, 0.25]], dtype=torch.float64))
    if kind == "adamw":
        optimizer = torch.optim.AdamW([p, q], lr=0.01, amsgrad=True, weight_decay=0.0)
    else:
        optimizer = torch.optim.SGD([p, q], lr=0.01, momentum=0.8)
    p.grad = torch.tensor([0.3, -0.1], dtype=p.dtype)
    q.grad = torch.tensor([[0.2, -0.4]], dtype=q.dtype)
    if initialize:
        optimizer.step()  # One hand-assigned tiny tensor step, no backward/model.
    return [p, q], optimizer


class TransactionTests(unittest.TestCase):
    def test_module_import_does_not_import_torch(self):
        code = "import sys; import transaction; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parent,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_adamw_rejection_restores_weights_gradients_moments_defaults_rng(self):
        parameters, optimizer = make_optimizer()
        before_weights = [p.detach().clone() for p in parameters]
        before_grads = [p.grad.clone() for p in parameters]
        before_state = copy.deepcopy(optimizer.state_dict())
        before_defaults = copy.deepcopy(optimizer.defaults)
        py_rng, cpu_rng = random.getstate(), torch.get_rng_state().clone()
        with patch.object(torch.cuda, "is_initialized", side_effect=AssertionError("CPU path touched CUDA")), \
             patch.object(torch.cuda, "get_rng_state", side_effect=AssertionError("CPU path touched CUDA")):
            with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch):
                optimizer.step()
                parameters[0].grad.add_(9)
                parameters[1].grad = None
                optimizer.param_groups[0]["lr"] = 2.0
                optimizer.defaults["lr"] = 3.0
                optimizer.transient_flag = "must disappear"
                random.random()
                torch.rand(4)
        for p, weight, grad in zip(parameters, before_weights, before_grads):
            self.assertTrue(torch.equal(p, weight))
            self.assertTrue(torch.equal(p.grad, grad))
        assert_tree_equal(self, optimizer.state_dict(), before_state)
        assert_tree_equal(self, optimizer.defaults, before_defaults)
        self.assertFalse(hasattr(optimizer, "transient_flag"))
        self.assertEqual(random.getstate(), py_rng)
        self.assertTrue(torch.equal(torch.get_rng_state(), cpu_rng))

    def test_first_step_rejection_removes_new_adam_state(self):
        parameters, optimizer = make_optimizer(initialize=False)
        self.assertFalse(optimizer.state)
        parameters[1].grad = None
        with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch):
            optimizer.step()
            parameters[1].grad = torch.ones_like(parameters[1])
            self.assertTrue(optimizer.state)
        self.assertFalse(optimizer.state)
        self.assertIsNone(parameters[1].grad)

    def test_exception_restores_sgd_state(self):
        parameters, optimizer = make_optimizer("sgd")
        before = copy.deepcopy(optimizer.state_dict())
        values = [p.clone() for p in parameters]
        with self.assertRaisesRegex(RuntimeError, "evaluator exploded"):
            with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch):
                optimizer.step()
                raise RuntimeError("evaluator exploded")
        assert_tree_equal(self, optimizer.state_dict(), before)
        for p, value in zip(parameters, values):
            self.assertTrue(torch.equal(p, value))

    def test_restore_weights_only_keeps_candidate_moments_then_commit(self):
        parameters, optimizer = make_optimizer()
        old = [p.clone() for p in parameters]
        with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch) as tx:
            optimizer.step()
            candidate_state = copy.deepcopy(optimizer.state_dict())
            deltas = [p.detach() - tx.original_weights[name] for name, p in zip(("p", "q"), parameters)]
            tx.restore_weights_only()
            assert_tree_equal(self, optimizer.state_dict(), candidate_state)
            with torch.no_grad():
                for p, delta in zip(parameters, deltas):
                    p.add_(delta, alpha=0.25)
            tx.commit()
        assert_tree_equal(self, optimizer.state_dict(), candidate_state)
        for p, value, delta in zip(parameters, old, deltas):
            self.assertTrue(torch.equal(p, value + 0.25 * delta))

    def test_shape_dtype_and_requires_grad_changes_refuse_commit_and_restore(self):
        for mutation in ("shape", "dtype", "requires_grad", "storage"):
            with self.subTest(mutation=mutation):
                parameters, optimizer = make_optimizer()
                original = parameters[0].clone()
                storage = parameters[0].untyped_storage().data_ptr()
                with self.assertRaises(ScopeViolation):
                    with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch) as tx:
                        if mutation == "shape":
                            parameters[0].data = torch.tensor([9.0], dtype=torch.float64)
                        elif mutation == "dtype":
                            parameters[0].data = parameters[0].float()
                        elif mutation == "requires_grad":
                            parameters[0].requires_grad_(False)
                        else:
                            parameters[0].data = parameters[0].detach().clone()
                        tx.commit()
                self.assertTrue(torch.equal(parameters[0], original))
                self.assertEqual(parameters[0].dtype, original.dtype)
                self.assertEqual(parameters[0].untyped_storage().data_ptr(), storage)
                self.assertTrue(parameters[0].requires_grad)

    def test_optimizer_membership_mutation_refuses_commit_and_restores(self):
        parameters, optimizer = make_optimizer()
        before = copy.deepcopy(optimizer.state_dict())
        group = optimizer.param_groups[0]
        with self.assertRaises(ScopeViolation):
            with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch) as tx:
                optimizer.param_groups[0]["params"] = [parameters[1], parameters[0]]
                tx.commit()
        self.assertIs(optimizer.param_groups[0], group)
        self.assertIs(optimizer.param_groups[0]["params"][0], parameters[0])
        assert_tree_equal(self, optimizer.state_dict(), before)

    def test_optimizer_moment_metadata_changes_refuse_commit_and_restore(self):
        for mutation in ("shape", "dtype"):
            with self.subTest(mutation=mutation):
                parameters, optimizer = make_optimizer()
                before = copy.deepcopy(optimizer.state_dict())
                with self.assertRaises(ScopeViolation):
                    with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch) as tx:
                        moment = optimizer.state[parameters[0]]["exp_avg"]
                        optimizer.state[parameters[0]]["exp_avg"] = moment[:1] if mutation == "shape" else moment.float()
                        tx.commit()
                assert_tree_equal(self, optimizer.state_dict(), before)

    def test_gradient_flag_and_state_factory_restored(self):
        parameters, optimizer = make_optimizer()
        self.assertFalse(parameters[0].grad.requires_grad)
        with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch):
            parameters[0].grad.requires_grad_(True)
            optimizer.state.default_factory = list
        self.assertFalse(parameters[0].grad.requires_grad)
        self.assertIs(optimizer.state.default_factory, dict)

    def test_out_of_scope_or_aliased_parameters_refuse_entry(self):
        parameters, optimizer = make_optimizer()
        with self.assertRaises(ScopeViolation):
            with TensorTransaction([("p", parameters[0])], optimizer, torch_module=torch):
                self.fail("entered incomplete scope")
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        alias = torch.nn.Parameter(p.detach())
        aliased_optimizer = torch.optim.SGD([p, alias], lr=0.1)
        with self.assertRaises(ScopeViolation):
            with TensorTransaction([("p", p), ("alias", alias)], aliased_optimizer, torch_module=torch):
                self.fail("entered aliased scope")

    def test_hooks_and_unsupported_optimizer_refuse_entry(self):
        parameters, optimizer = make_optimizer()
        handle = optimizer.register_step_post_hook(lambda *args: None)
        with self.assertRaises(ScopeViolation):
            with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch):
                self.fail("entered hook scope")
        handle.remove()
        other = torch.optim.Adam(parameters, lr=0.01)
        with self.assertRaises(ScopeViolation):
            with TensorTransaction(zip(("p", "q"), parameters), other, torch_module=torch):
                self.fail("entered unsupported optimizer")

    def test_error_after_commit_still_rolls_back(self):
        parameters, optimizer = make_optimizer()
        before = [p.clone() for p in parameters]
        with self.assertRaises(ValueError):
            with TensorTransaction(zip(("p", "q"), parameters), optimizer, torch_module=torch) as tx:
                optimizer.step()
                tx.commit()
                raise ValueError("late evaluator failure")
        for p, value in zip(parameters, before):
            self.assertTrue(torch.equal(p, value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
