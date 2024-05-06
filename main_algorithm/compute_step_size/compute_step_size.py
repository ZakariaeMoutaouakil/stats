from typing import List

import numpy as np


def compute_step_size(X_k: List[np.ndarray], x_c: np.ndarray, v: np.ndarray) -> float:
    """
    Computes the theta_c as the median of the dot product of (X_k - x_c) with v1.

    Parameters:
    - X_k (List[np.ndarray]): List of vectors.
    - x_c (np.ndarray): Central vector.
    - v1 (np.ndarray): A specific vector to dot with.

    Returns:
    - float: The median of the dot products.
    """
    # Compute the difference between each X_k and x_c
    differences = [x_k - x_c for x_k in X_k]

    # Calculate the dot product of each difference with v1
    dot_products = [np.dot(diff, v) for diff in differences]

    # Compute the median of these dot products
    theta_c = np.median(dot_products)
    return theta_c


# Example usage:
if __name__ == "__main__":
    # Define some example vectors
    X_k = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
    x_c = np.array([2, 2])
    v1 = np.array([1, 0])

    # Compute theta_c
    step = compute_step_size(X_k, x_c, v1)
    print("Computed theta_c:", step)
