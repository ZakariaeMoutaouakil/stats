from typing import List

import numpy as np


def compute_l2_norms(vectors: List[np.ndarray]) -> List[float]:
    """
    Computes the L2 norm of each vector in a list of vectors.

    Parameters:
    - vectors (List[np.ndarray]): A list of numpy arrays representing vectors.

    Returns:
    - List[float]: A list of the L2 norms of each vector.
    """
    norms = [np.linalg.norm(vector) for vector in vectors]
    return norms


if __name__ == "__main__":
    # Example usage:
    vectors_ = [np.array([3, 4]), np.array([1, 1, 1, 1]), np.array([0, 0, 0])]
    norms_ = compute_l2_norms(vectors_)
    print(norms_)  # Output: [5.0, 2.0, 0.0]
