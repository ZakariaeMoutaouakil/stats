from typing import List

import numpy as np


def coordinate_median(vectors: List[List[float]]) -> List[float]:
    if not vectors or not vectors[0]:
        return []  # Return an empty list if input is empty or vectors are not properly defined

    # Check if all vectors are of the same length
    first_len = len(vectors[0])
    if any(len(vec) != first_len for vec in vectors):
        raise ValueError("All vectors must be of the same length")

    transposed = list(zip(*vectors))
    medians = [np.median(dim) for dim in transposed]
    return medians


if __name__ == "__main__":
    # Example usage:
    vectors_ = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
    medians_ = coordinate_median(vectors_)

    print("Coordinate-wise median of the vectors:", medians_)
