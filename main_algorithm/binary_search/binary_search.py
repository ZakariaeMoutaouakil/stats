from typing import List

import numpy as np

from main_algorithm.solve_sdp.solve_sdp import solve_sdp


def binary_search(x_c: np.ndarray, X_k: List[np.ndarray], rho_min: float, rho_max: float, T: int):
    tolerance = 0.9981
    lower_bound = rho_min
    upper_bound = rho_max
    i = 0

    while i < T:
        rho_mid = (lower_bound + upper_bound) / 2
        objective_value, M_opt, y_opt = solve_sdp(x_c, X_k, rho_mid)

        if tolerance <= objective_value <= 1:
            return rho_mid, objective_value, M_opt, y_opt
        elif objective_value < tolerance:
            upper_bound = rho_mid
        else:
            lower_bound = rho_mid

        i += 1

    # Final attempt with the midpoint
    rho_mid = (lower_bound + upper_bound) / 2
    objective_value, M_opt, y_opt = solve_sdp(x_c, X_k, rho_mid)
    return rho_mid, objective_value, M_opt, y_opt


if __name__ == "__main__":
    # Example usage
    n = 5
    x_c = np.random.randn(n, 1)
    X_k = [np.random.randn(n, 1) for _ in range(10)]
    rho_min = 0.1
    rho_max = 2.0
    T = 100

    rho_opt, objective_opt, M_opt, y_opt = binary_search(x_c, X_k, rho_min, rho_max, T)
    print("Optimal rho:", rho_opt)
    print("Optimal objective:", objective_opt)
    print("Optimal M:")
    print(M_opt)
    print("Optimal y:")
    print(y_opt)
