import unittest

import torch

from controller import constrained_step, FiniteCheck, MarginConstraint


class ControllerTests(unittest.TestCase):
    def setup_case(self):
        weights = torch.nn.Parameter(torch.tensor([0., 0.], dtype=torch.float64))
        opt = torch.optim.AdamW([weights], lr=.1, weight_decay=0.)
        return weights, opt

    def test_joint_correction_acquires_while_preserving_old_linear_output(self):
        p, opt = self.setup_case()
        def teaching():
            return (p[0] - 1.) ** 2
        constraints = [MarginConstraint('old lower', lambda: p.sum(), 0.),
                       MarginConstraint('old upper', lambda: -p.sum(), 0.)]
        result = constrained_step([('w', p)], opt, teaching, constraints=constraints,
                                  finite_teaching_loss=lambda: float(teaching().detach()),
                                  finite_check=lambda: FiniteCheck(abs(float(p.detach().sum())) < 1e-9,
                                                                 bool(torch.any(p != 0)), 'linear finite guard'))
        self.assertTrue(result.accepted)
        self.assertGreater(float(p.detach()[0]), 0.)
        self.assertAlmostEqual(float(p.detach().sum()), 0., places=10)
        self.assertLess(result.teaching_after, result.teaching_before)
        self.assertEqual(int(opt.state[p]['step']), 1)

    def test_failed_finite_guard_restores_weights_and_adam_state(self):
        p, opt = self.setup_case()
        result = constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                                  finite_teaching_loss=lambda: float(((p[0] - 1.) ** 2).detach()),
                                  constraints=[], finite_check=lambda: FiniteCheck(False, True, 'reject'))
        self.assertFalse(result.accepted)
        torch.testing.assert_close(p, torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(len(opt.state), 0)

    def test_materialized_noop_is_not_acquisition(self):
        p, opt = self.setup_case()
        result = constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                                  finite_teaching_loss=lambda: float(((p[0] - 1.) ** 2).detach()),
                                  constraints=[], finite_check=lambda: FiniteCheck(True, False, 'native no-op'))
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, 'native no-op')
        self.assertEqual(len(opt.state), 0)

    def test_finite_nonlinear_guard_forces_backtracking(self):
        p, opt = self.setup_case()
        result = constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                                  finite_teaching_loss=lambda: float(((p[0] - 1.) ** 2).detach()),
                                  constraints=[], finite_check=lambda: FiniteCheck(float(p.detach().square().sum()) < .003,
                                                                                 bool(torch.any(p != 0)), 'quadratic guard'))
        self.assertTrue(result.accepted)
        self.assertAlmostEqual(result.step_scale, .5)

    def test_contradictory_acquisition_constraint_refuses(self):
        p, opt = self.setup_case()
        result = constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                                  finite_teaching_loss=lambda: float(((p[0] - 1.) ** 2).detach()),
                                  constraints=[MarginConstraint('must not increase', lambda: -p[0], 0.)],
                                  finite_check=lambda: FiniteCheck(True, True, 'unused'))
        self.assertFalse(result.accepted)
        self.assertEqual(len(opt.state), 0)

    def test_exception_restores_the_whole_step(self):
        p, opt = self.setup_case()
        def broken():
            raise RuntimeError('decoder failed')
        with self.assertRaisesRegex(RuntimeError, 'decoder failed'):
            constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                             finite_teaching_loss=lambda: float(((p[0] - 1.) ** 2).detach()),
                             constraints=[], finite_check=broken)
        torch.testing.assert_close(p, torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)
        self.assertEqual(len(opt.state), 0)

    def test_proxy_progress_cannot_replace_materialized_progress(self):
        p, opt = self.setup_case()
        result = constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                                  finite_teaching_loss=lambda: float(((p[0] + 1.) ** 2).detach()),
                                  constraints=[], finite_check=lambda: FiniteCheck(True, True, 'unused'))
        self.assertFalse(result.accepted)
        self.assertEqual(len(opt.state), 0)

    def test_backtracking_preserves_fixed_predicted_descent_requirement(self):
        p, opt = self.setup_case()
        result = constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                                  finite_teaching_loss=lambda: float(((p[0] - 1.) ** 2).detach()),
                                  descent_fraction=1., constraints=[],
                                  finite_check=lambda: FiniteCheck(float(p.detach().square().sum()) < .003,
                                                                 True, 'needs a smaller step'))
        self.assertFalse(result.accepted)
        self.assertIn('fixed acquisition-descent', result.reason)
        self.assertEqual(len(opt.state), 0)

    def test_disconnected_shadow_margin_is_refused(self):
        p, opt = self.setup_case()
        shadow = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, 'disconnected'):
            constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                             finite_teaching_loss=lambda: 1.,
                             constraints=[MarginConstraint('shadow', lambda: shadow.square(), .1)],
                             finite_check=lambda: FiniteCheck(True, True, 'unused'))

    def test_sequence_wise_backward_matches_single_combined_loss(self):
        p, opt = self.setup_case()
        q, opt_q = self.setup_case()
        def backward():
            total = 0.
            for index, target in ((0, 1.), (1, -1.)):
                loss = (p[index] - target) ** 2 / 2
                total += float(loss.detach())
                loss.backward()
            return total
        common = dict(constraints=[], finite_check=lambda: FiniteCheck(True, True, 'pass'))
        first = constrained_step([('w', p)], opt, lambda: (p[0] - 1.) ** 2,
                                 finite_teaching_loss=lambda: float(((p[0] - 1.) ** 2).detach()),
                                 optimization_backward=backward, **common)
        second = constrained_step([('w', q)], opt_q, lambda: (q[0] - 1.) ** 2,
                                  finite_teaching_loss=lambda: float(((q[0] - 1.) ** 2).detach()),
                                  optimization_loss=lambda: ((q[0] - 1.) ** 2 + (q[1] + 1.) ** 2) / 2,
                                  **common)
        self.assertTrue(first.accepted and second.accepted)
        torch.testing.assert_close(p, q, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
