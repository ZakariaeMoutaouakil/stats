import unittest

import numpy as np

from basics.compute_l2_norms.compute_l2_norms import compute_l2_norms


class TestComputeL2Norms(unittest.TestCase):
    def test_standard_vectors(self):
        """Test L2 norms with standard vectors."""
        vectors = [np.array([3, 4]), np.array([1, 1, 1, 1]), np.array([0, 0, 0])]
        expected = [5.0, 2.0, 0.0]
        result = compute_l2_norms(vectors)
        self.assertEqual(result, expected)

    def test_empty_vector(self):
        """Test with an empty vector."""
        vectors = [np.array([])]
        expected = [0.0]  # Norm of an empty vector is 0.0
        result = compute_l2_norms(vectors)
        self.assertEqual(result, expected)

    def test_negative_elements(self):
        """Test vectors with negative elements."""
        vectors = [np.array([-3, -4]), np.array([-1, 2, -3])]
        expected = [5.0, np.sqrt(1 + 4 + 9)]  # sqrt(9+16), sqrt(1+4+9)
        result = compute_l2_norms(vectors)
        self.assertEqual(result, expected)

    def test_single_element_vectors(self):
        """Test vectors with a single element."""
        vectors = [np.array([5]), np.array([-5])]
        expected = [5.0, 5.0]
        result = compute_l2_norms(vectors)
        self.assertEqual(result, expected)

    def test_large_numbers(self):
        """Test vectors with large numbers."""
        vectors = [np.array([1000000, 0]), np.array([0, 1000000])]
        expected = [1000000.0, 1000000.0]
        result = compute_l2_norms(vectors)
        self.assertEqual(result, expected)

    def test_mixed_dimensions(self):
        """Test a mix of vectors with different dimensions."""
        vectors = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6])]
        expected = [np.sqrt(1 + 4 + 9), np.sqrt(16 + 25), 6.0]
        result = compute_l2_norms(vectors)
        self.assertAlmostEqual(result[0], expected[0])
        self.assertAlmostEqual(result[1], expected[1])
        self.assertAlmostEqual(result[2], expected[2])


if __name__ == '__main__':
    unittest.main()
