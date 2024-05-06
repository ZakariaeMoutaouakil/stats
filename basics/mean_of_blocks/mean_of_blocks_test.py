import unittest

from basics.mean_of_blocks.mean_of_blocks import mean_of_blocks


class TestMeanOfBlocks(unittest.TestCase):
    def test_single_block(self):
        blocks = [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
        expected = [[4, 5, 6]]  # Mean of each dimension
        self.assertEqual(mean_of_blocks(blocks), expected)

    def test_multiple_blocks(self):
        blocks = [[[1, 2], [3, 4]], [[5, 6]]]
        expected = [[2, 3], [5, 6]]  # Mean of each block
        self.assertEqual(mean_of_blocks(blocks), expected)

    def test_empty_block(self):
        blocks = [[], [[1, 2], [3, 4]]]  # First block is empty
        expected = [[], [2, 3]]  # Empty list for the empty block, mean for the second
        self.assertEqual(mean_of_blocks(blocks), expected)

    def test_all_empty_blocks(self):
        blocks = [[], []]
        expected = [[], []]  # Expect empty lists for both blocks
        self.assertEqual(mean_of_blocks(blocks), expected)

    def test_varying_dimensions(self):
        blocks = [[[1, 2, 3]], [[4, 5], [6, 7]]]
        expected = [[1, 2, 3], [5, 6]]  # Each block calculated separately
        self.assertEqual(mean_of_blocks(blocks), expected)


if __name__ == "__main__":
    unittest.main()
