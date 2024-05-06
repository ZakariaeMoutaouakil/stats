import unittest

import numpy as np

from basics.top_eigenvector.top_eigenvector import top_eigenvector


class TestTopEigenvector(unittest.TestCase):
    def test_2x2_matrix(self):
        A = np.array([[2, 0], [0, 1]])
        expected_vector = np.array([1, 0])  # Eigenvalues are 2 and 1, top eigenvector for eigenvalue 2
        result_vector = top_eigenvector(A)
        np.testing.assert_array_almost_equal(result_vector, expected_vector)

    def test_symmetric_matrix(self):
        A = np.array([[2, 1], [1, 2]])
        expected_vector = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])  # Normalized
        result_vector = top_eigenvector(A)
        np.testing.assert_array_almost_equal(np.abs(result_vector), np.abs(expected_vector))

    def test_larger_matrix(self):
        A = np.array([[1, 2, 3], [2, 3, 1], [3, 1, 2]])
        result_vector = top_eigenvector(A)
        self.assertEqual(len(result_vector), 3)  # Basic check for output dimensions

    def test_with_complex_numbers(self):
        A = np.array([[1, -1j], [1j, 1]])
        result_vector = top_eigenvector(A)
        self.assertEqual(len(result_vector), 2)  # Handling complex numbers


if __name__ == '__main__':
    unittest.main()
