import unittest

import numpy as np

from gram_projector import from_dense, project_gram


class ProjectorTests(unittest.TestCase):
    def test_no_constraints_is_ball_projection(self):
        d, r = from_dense(np.empty((0, 2)), [3., 4.], [], 2.)
        np.testing.assert_allclose(d, [1.2, 1.6])
        self.assertTrue(r.ball_active)

    def test_feasible_proposal_is_unchanged(self):
        d, r = from_dense([[1., 0.], [0., 1.]], [1., 2.], [0., 0.], 3.)
        np.testing.assert_allclose(d, [1., 2.])
        self.assertAlmostEqual(r.distance_to_proposal, 0.)

    def test_halfspace_projection_without_ball(self):
        d, _ = from_dense([[1., 0.]], [-2., 3.], [1.], 5.)
        np.testing.assert_allclose(d, [1., 3.])

    def test_active_halfspace_and_ball(self):
        d, r = from_dense([[1., 0.]], [-2., 3.], [1.], 2.)
        np.testing.assert_allclose(d, [1., np.sqrt(3.)])
        self.assertTrue(r.ball_active)

    def test_naive_radial_scaling_would_violate_nonzero_floor(self):
        d, _ = from_dense([[1., 0.]], [-2., 3.], [1.], 2.)
        naive = np.array([1., 3.]) * 2 / np.sqrt(10)
        self.assertLess(naive[0], 1.)
        self.assertGreaterEqual(d[0], 1. - 1e-9)

    def test_equality_compensation_and_acquisition(self):
        a = np.array([[1., 1.], [-1., -1.], [1., -1.]])
        d, _ = from_dense(a, [1., 0.], [0., 0., 0.2], 2.)
        np.testing.assert_allclose(d, [.5, -.5])

    def test_impossible_constraints_or_too_small_radius(self):
        self.assertIsNone(from_dense([[1.], [-1.]], [0.], [1., 1.], 100.)[0])
        self.assertIsNone(from_dense([[1.]], [0.], [2.], 1.)[0])

    def test_zero_and_tiny_radii_do_not_get_absolute_slack(self):
        self.assertIsNone(from_dense([[1.]], [0.], [1e-6], 0.)[0])
        self.assertIsNone(from_dense([[1.]], [0.], [1e-6], 1e-8)[0])
        d, _ = from_dense([[1.]], [1.], [-1.], 0.)
        np.testing.assert_array_equal(d, [0.])

    def test_redundancy_and_positive_row_rescaling(self):
        rows = np.array([[1., 0.], [2., 0.], [0., 1.]])
        b = np.array([1., 2., -1.])
        d, _ = from_dense(rows, [-3., 3.], b, 2.)
        scales = np.array([1e-5, 1e5, 17.])
        rescaled, _ = from_dense(rows * scales[:, None], [-3., 3.], b * scales, 2.)
        np.testing.assert_allclose(d, rescaled, atol=1e-8)

    def test_zero_rows_and_inconsistent_gram(self):
        d, _ = from_dense([[0., 0.]], [1., 2.], [-1.], 3.)
        np.testing.assert_allclose(d, [1., 2.])
        self.assertIsNone(from_dense([[0., 0.]], [1., 2.], [1e-12], 3.)[0])
        with self.assertRaises(ValueError):
            project_gram([[1.]], [2.], [0.], 1., 2.)

    def test_random_projections_beat_all_sampled_feasible_points(self):
        rng = np.random.default_rng(1788)
        # Numerical stress oracle: dense random feasible points cannot beat a
        # true projection. Analytic tests above establish known optima.
        for _ in range(20):
            rows = rng.normal(size=(3, 2))
            b = -rng.uniform(.05, .6, size=3)  # origin always feasible
            u = rng.normal(size=2) * 2
            d, r = from_dense(rows, u, b, 1.)
            self.assertIsNotNone(r)
            self.assertTrue(np.all(rows @ d >= b - 1e-8))
            self.assertLessEqual(np.linalg.norm(d), 1 + 1e-8)
            points = rng.normal(size=(5000, 2))
            points /= np.maximum(1, np.linalg.norm(points, axis=1))[:, None]
            points = points[np.all(points @ rows.T >= b, axis=1)]
            if len(points):
                self.assertLessEqual(np.linalg.norm(d - u), np.min(np.linalg.norm(points - u, axis=1)) + 1e-8)


if __name__ == '__main__':
    unittest.main(verbosity=2)
