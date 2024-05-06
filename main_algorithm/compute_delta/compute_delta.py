from typing import List

import numpy as np

from basics.compute_l2_norms.compute_l2_norms import compute_l2_norms
from basics.coordinate_median.coordinate_median import coordinate_median


def compute_delta(X_i: List[np.ndarray]):
    x_i: List[List[float]] = [list(array) for array in X_i]

    # Compute the coordinate-wise median of the vectors.
    mu_0 = coordinate_median(x_i)

    # Subtract the median from each vector to get a list of vectors.
    vectors = [mean - mu_0 for mean in X_i]

    # Compute the L2 norms of these difference vectors.
    norms = compute_l2_norms(vectors)

    # Return the median of these norms.
    return np.median(norms)


if __name__ == "__main__":
    # Example usage:
    vectors_ = [np.array([3, 4]), np.array([1, 1]), np.array([2, 2])]
    delta = compute_delta(vectors_)
    print("Computed delta:", delta)  # Output should show the median norm of the differences
