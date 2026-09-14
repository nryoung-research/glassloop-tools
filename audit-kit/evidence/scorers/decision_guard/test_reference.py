"""Adversarial tests of the mathematical reference, not of a language model."""
import math
import unittest

from reference import (
    backtrack_transaction, bfloat16_scalar, compensated_point, dot,
    gated_gradient, gated_mlp, greedy_trace_guard, minnorm_direction,
)


class DirectionTests(unittest.TestCase):
    def test_known_projection_and_trust_radius(self):
        result = minnorm_direction([[1, 0], [0, 1]], [1, 2], 3)
        self.assertEqual(result.delta, (1.0, 2.0))
        self.assertAlmostEqual(result.length, math.sqrt(5))
        self.assertIsNone(minnorm_direction([[1, 0], [0, 1]], [1, 2], 2))

    def test_redundant_and_infeasible_constraints(self):
        result = minnorm_direction([[1, 0], [2, 0], [0, 1]], [1, 2, -3], 3)
        self.assertEqual(result.delta, (1.0, 0.0))
        self.assertIsNone(minnorm_direction([[1], [-1]], [1, 1], 100))

    def test_compensation_escapes_independent_projection(self):
        root = (1.0, 1.0)
        old_grad, new_grad = gated_gradient(root, 1), gated_gradient(root, -1)
        result = minnorm_direction([old_grad, [-x for x in old_grad], new_grad], [0, 0, 0.02], 1)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(dot(old_grad, result.delta), 0.0, places=12)
        self.assertGreaterEqual(dot(new_grad, result.delta), 0.02 - 1e-12)
        # Every scalar old-input key is nonzero: its exact nullspace is {0}.
        independent = (0.0, 0.0)
        self.assertEqual(dot(new_grad, independent), 0.0)
        point = compensated_point(root, 0.5)
        self.assertAlmostEqual(gated_mlp(point, 1), gated_mlp(root, 1), places=12)
        self.assertAlmostEqual(gated_mlp(point, -1), 0.5, places=12)

    def test_nonlinear_postcheck_overrules_linear_certificate(self):
        root = (1.0, 1.0)
        old = gated_mlp(root, 1)
        go, gn = gated_gradient(root, 1), gated_gradient(root, -1)
        result = minnorm_direction([go, [-x for x in go], gn], [0, 0, 0.02], 1)
        state = list(root)
        rejected = []

        def check(candidate):
            drift = abs(gated_mlp(candidate, 1) - old)
            if drift > 1e-4:
                rejected.append(drift)
            return drift <= 1e-4, "finite root check"

        trial = backtrack_transaction(state, result.delta, check, 1)
        self.assertTrue(rejected)
        self.assertTrue(trial.accepted)
        self.assertLess(trial.scale, 1)
        self.assertLessEqual(abs(gated_mlp(state, 1) - old), 1e-4)

    def test_zero_kl_gradient_cannot_certify_no_damage(self):
        # Bernoulli teacher p=1/2, candidate sigmoid(theta):
        # KL has zero gradient at theta=0 but positive curvature and finite drift.
        def kl(theta):
            q = 1 / (1 + math.exp(-theta))
            return 0.5 * math.log(0.5 / q) + 0.5 * math.log(0.5 / (1 - q))
        eps = 1e-6
        gradient = (kl(eps) - kl(-eps)) / (2 * eps)
        self.assertAlmostEqual(gradient, 0.0, places=8)
        self.assertGreater(kl(1.0), 0.1)


class GuardTests(unittest.TestCase):
    def test_all_rivals_not_just_cached_runner_up(self):
        def logits(_):
            return {"win": 2, "cached-runner-up": 1, "previously-low-rival": 3, "<eos>": -1}
        result = greedy_trace_guard(logits, (), ("win", "<eos>"))
        self.assertFalse(result.passed)

    def test_eos_is_a_decision(self):
        def logits(prefix):
            return {"win": 3, "more": 1, "<eos>": 0} if not prefix else {"win": 0, "more": 2, "<eos>": 1}
        result = greedy_trace_guard(logits, (), ("win", "<eos>"))
        self.assertFalse(result.passed)
        self.assertEqual(result.checked_positions, 2)
        with self.assertRaises(ValueError):
            greedy_trace_guard(logits, (), ("win",))

    def test_positive_float_margin_can_collapse_after_scalar_bf16_rounding(self):
        def logits(prefix):
            return {"win": 1.0005, "rival": 1.0, "<eos>": -1.0} if not prefix else {"win": 0.0, "rival": 1.0, "<eos>": 2.0}
        def rounded(prefix):
            return {token: bfloat16_scalar(value) for token, value in logits(prefix).items()}
        self.assertTrue(greedy_trace_guard(logits, (), ("win", "<eos>")).passed)
        self.assertFalse(greedy_trace_guard(rounded, (), ("win", "<eos>")).passed)
        # The same failure must roll back a proposed state, not just print FAIL.
        state = [0.0]
        result = backtrack_transaction(state, [1.0], lambda _: (greedy_trace_guard(rounded, (), ("win", "<eos>")).passed, "rounded decision"), 1)
        self.assertFalse(result.accepted)
        self.assertEqual(state, [0.0])

    def test_nonfinite_logits_fail(self):
        result = greedy_trace_guard(lambda _: {"win": 2, "bad": float("nan"), "<eos>": 0}, (), ("win", "<eos>"))
        self.assertFalse(result.passed)


class TransactionTests(unittest.TestCase):
    def test_failure_and_exception_restore_state(self):
        state = [1.0, 2.0]
        result = backtrack_transaction(state, [0.1, 0.2], lambda _: (False, "reject"), 1)
        self.assertFalse(result.accepted)
        self.assertEqual(state, [1.0, 2.0])
        def crash(_):
            raise RuntimeError("check failed")
        with self.assertRaises(RuntimeError):
            backtrack_transaction(state, [0.1, 0.2], crash, 1)
        self.assertEqual(state, [1.0, 2.0])

    def test_cumulative_root_budget_catches_incremental_drift(self):
        root = 0.0
        state = [root]
        for _ in range(3):
            previous = state[0]
            result = backtrack_transaction(state, [0.04], lambda p: (abs(p[0] - previous) <= 0.05, "increment only"), 0.05, 0)
            self.assertTrue(result.accepted)
        self.assertGreater(abs(state[0] - root), 0.1)
        state = [root]
        def root_check(p):
            return abs(p[0] - root) <= 0.1, "cumulative root budget"
        for _ in range(2):
            self.assertTrue(backtrack_transaction(state, [0.04], root_check, 0.05, 0).accepted)
        before = state[:]
        self.assertFalse(backtrack_transaction(state, [0.04], root_check, 0.05, 0).accepted)
        self.assertEqual(state, before)

    def test_trust_radius_and_invalid_inputs(self):
        state = [0.0]
        result = backtrack_transaction(state, [4.0], lambda _: (True, "pass"), 1.0)
        self.assertTrue(result.accepted)
        self.assertEqual(result.scale, 0.25)
        self.assertEqual(state, [1.0])
        with self.assertRaises(ValueError):
            backtrack_transaction(state, [float("nan")], lambda _: (True, "pass"), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
