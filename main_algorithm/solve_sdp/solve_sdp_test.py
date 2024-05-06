import unittest

import cvxpy as cp
import numpy as np

from main_algorithm.solve_sdp.solve_sdp import solve_sdp


class TestSolveSDP(unittest.TestCase):
    def setUp(self):
        # Basic setup that can be reused across tests
        self.n = 5  # Dimension of vectors
        self.x_c = np.random.randn(self.n, 1)  # Central vector
        self.X_k = [np.random.randn(self.n, 1) for _ in range(10)]  # List of vectors
        self.rho = 1.5  # Constant multiplier

    def test_sdp_basic(self):
        """Test basic functionality with reasonable inputs."""
        optimal_value, M_opt, y_opt = solve_sdp(self.x_c, self.X_k, self.rho)
        self.assertIsNotNone(optimal_value)
        self.assertIsNotNone(M_opt)
        self.assertIsNotNone(y_opt)
        self.assertTrue(M_opt.shape == (self.n, self.n))
        self.assertTrue(len(y_opt) == len(self.X_k))
        self.assertTrue(cp.norm(y_opt, 1).value >= 0)

    def test_sdp_zero_rho(self):
        """Test with rho set to zero."""
        optimal_value, M_opt, y_opt = solve_sdp(self.x_c, self.X_k, 0)
        # Expecting different behavior, perhaps no constraints influenced by X_k
        self.assertIsNotNone(optimal_value)
        self.assertIsNotNone(M_opt)
        self.assertIsNotNone(y_opt)

    def test_sdp_negative_rho(self):
        """Test with a negative value of rho."""
        with self.assertRaises(ValueError, msg="Function should raise ValueError for negative rho"):
            solve_sdp(self.x_c, self.X_k, -1.5)

    def test_sdp_empty_X_k(self):
        X_k_empty = []
        optimal_value, M_opt, y_opt = solve_sdp(self.x_c, X_k_empty, self.rho)
        self.assertTrue(np.all(np.linalg.eigvals(M_opt) >= 0), "M should be positive semidefinite")
        self.assertEqual(len(y_opt), 0, "y should be empty because there are no constraints from X_k")
        self.assertTrue(np.array_equal(M_opt, M_opt.T), "M should be symmetric")

    def test_sdp_singular(self):
        """Test with all X_k identical to x_c."""
        X_k_singular = [self.x_c for _ in range(10)]
        optimal_value, M_opt, y_opt = solve_sdp(self.x_c, X_k_singular, self.rho)
        self.assertIsNotNone(optimal_value)
        self.assertIsNotNone(M_opt)
        self.assertIsNotNone(y_opt)

    def test_sdp_large_scale(self):
        """Test with a larger scale setup."""
        n_large = 50  # Larger dimension
        x_c_large = np.random.randn(n_large, 1)
        X_k_large = [np.random.randn(n_large, 1) for _ in range(100)]  # More vectors
        rho_large = 2.0
        optimal_value, M_opt, y_opt = solve_sdp(x_c_large, X_k_large, rho_large)
        self.assertIsNotNone(optimal_value)
        self.assertIsNotNone(M_opt)
        self.assertIsNotNone(y_opt)
        self.assertTrue(M_opt.shape == (n_large, n_large))
        self.assertTrue(len(y_opt) == len(X_k_large))


if __name__ == '__main__':
    unittest.main()
