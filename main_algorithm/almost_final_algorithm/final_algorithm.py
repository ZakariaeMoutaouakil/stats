from typing import List

import numpy as np

from basics.coordinate_median.coordinate_median import coordinate_median
from main_algorithm.binary_search.binary_search import binary_search
from main_algorithm.compute_delta.compute_delta import compute_delta
from main_algorithm.solve_sdp.solve_sdp import solve_sdp


def almost_final_algorithm(X_i: List[np.ndarray], x_c: np.ndarray):
    x_i: List[List[float]] = [list(array) for array in X_i]

    # Compute initial values
    mu_0 = coordinate_median(x_i)  # Example computation for mu(0)
    delta = compute_delta(X_i)  # Example value for delta
    d = len(x_c)

    T = np.ceil(np.log((10 ** 6) * (delta ** 4)))  # Example computation for T
    rho_0 = d / (delta ** 2)  # Example computation for rho_0

    # Run binary search to find an appropriate rho and M
    rho_mid, objective_value, M_opt, y_opt = binary_search(x_c, X_i, 0, rho_0, T)

    # Check the condition after binary search
    if 0.9981 <= np.trace(M_opt) + np.linalg.norm(M_opt, 1) <= 1:
        M_normed = M_opt / np.trace(M_opt)
        return True, M_normed
    else:
        # Placeholder for ALG, assume function is defined elsewhere
        if solve_sdp(x_c, X_i, rho_mid)[0] < 1:
            return False, mu_0
        else:
            return False, x_c


if __name__ == "__main__":
    # Example usage
    n = 5
    x_c = np.random.randn(n, 1)
    X_i = [np.random.randn(n, 1) for _ in range(10)]

    result, output = almost_final_algorithm(X_i, x_c)
    print("Result:", result)
    print("Output:", output)
