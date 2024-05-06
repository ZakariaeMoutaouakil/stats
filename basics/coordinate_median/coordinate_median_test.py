import unittest

from basics.coordinate_median.coordinate_median import coordinate_median


class TestCoordinateMedian(unittest.TestCase):
    def test_basic(self):
        vectors = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        expected = [5.5, 6.5, 7.5]
        self.assertEqual(coordinate_median(vectors), expected)

    def test_odd_number_of_vectors(self):
        vectors = [[1, 2], [3, 4], [5, 6]]
        expected = [3, 4]
        self.assertEqual(coordinate_median(vectors), expected)

    def test_even_number_of_vectors(self):
        vectors = [[1, 2], [3, 4], [5, 6], [7, 8]]
        expected = [4, 5]
        self.assertEqual(coordinate_median(vectors), expected)

    def test_single_vector(self):
        vectors = [[1, 2, 3]]
        expected = [1, 2, 3]
        self.assertEqual(coordinate_median(vectors), expected)

    def test_empty_list(self):
        vectors = []
        expected = []
        self.assertEqual(coordinate_median(vectors), expected)

    def test_empty_vectors(self):
        vectors = [[], [], []]
        expected = []
        self.assertEqual(coordinate_median(vectors), expected)

    def test_vectors_of_varying_length(self):
        vectors = [[1, 2, 3], [4, 5]]
        with self.assertRaises(ValueError):
            coordinate_median(vectors)


if __name__ == '__main__':
    unittest.main()
