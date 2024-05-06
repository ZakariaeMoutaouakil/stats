from typing import List

import numpy as np

from basics.coordinate_median.coordinate_median import coordinate_median
from basics.equipartition_vectors.equipartition_vectors import equipartition_vectors
from basics.mean_of_blocks.mean_of_blocks import mean_of_blocks
from basics.top_eigenvector.top_eigenvector import top_eigenvector
from main_algorithm.almost_final_algorithm.final_algorithm import almost_final_algorithm
from main_algorithm.compute_step_size.compute_step_size import compute_step_size


def final_algorithm(X_i: List[List[float]], K: int):
    blocks = equipartition_vectors(X_i, K)
    means = mean_of_blocks(blocks)

    x_c = coordinate_median(means)  # Example computation for mu(0)
    d = len(x_c)
    bool = True
    T = np.log(8 * np.sqrt(d)) / np.log(1 / 0.81)
    i = 0

    while bool and i < T:
        means_np = [np.array(sublist) for sublist in means]
        bool, M = almost_final_algorithm(means_np, np.array(x_c))

        if bool:
            v = top_eigenvector(M)
            step = compute_step_size(means_np, np.array(x_c), v)
            x_c -= step * v
            i += 1
        else:
            x_c = M

        return x_c


# Example usage:
if __name__ == "__main__":
    X_i = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]  # Sample data
    K = 2  # Number of blocks
    result = final_algorithm(X_i, K)
    print("Resulting vector:", result)

    # Create a dataset with outliers
    np.random.seed(0)
    data = np.random.normal(loc=0, scale=1, size=(100, 2))  # 100 normal points
    outliers = np.random.normal(loc=0, scale=20, size=(10, 2))  # 10 extreme points
    data_with_outliers = np.vstack([data, outliers])

    # Apply the robust mean estimation algorithm
    robust_mean = final_algorithm(data_with_outliers.tolist(), K=5)
    print("Robust mean estimated:", robust_mean)

    # Apply a non-robust mean estimation for comparison
    non_robust_mean = np.mean(data_with_outliers, axis=0)
    print("Non-robust mean estimated:", non_robust_mean)
