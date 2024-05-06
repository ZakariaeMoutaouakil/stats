import unittest

from basics.equipartition_vectors.equipartition_vectors import equipartition_vectors


class TestEquipartitionVectors(unittest.TestCase):

    def test_even_partition(self):
        vectors = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
        expected = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]]
        self.assertEqual(equipartition_vectors(vectors, 3), expected)

    def test_uneven_partition(self):
        vectors = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        expected = [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10]]]
        self.assertEqual(equipartition_vectors(vectors, 3), expected)

    def test_empty_vectors(self):
        vectors = []
        expected = [[], [], []]
        self.assertEqual(equipartition_vectors(vectors, 3), expected)

    def test_zero_partitions(self):
        vectors = [[1, 2], [3, 4], [5, 6]]
        with self.assertRaises(ZeroDivisionError):
            equipartition_vectors(vectors, 0)

    def test_single_element_partitions(self):
        vectors = [[1, 2], [3, 4], [5, 6], [7, 8]]
        expected = [[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]], []]
        self.assertEqual(equipartition_vectors(vectors, 5), expected)


if __name__ == "__main__":
    unittest.main()
