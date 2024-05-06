from typing import List, Tuple

import cvxpy as cp
import numpy as np


def solve_sdp(x_c: np.ndarray, X_k: List[np.ndarray], rho: float) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Solves a semidefinite programming problem with given parameters.

    Parameters:
    - x_c (np.ndarray): The central vector, shape (n, 1).
    - X_k (List[np.ndarray]): List of vectors, each of shape (n, 1).
    - rho (float): A constant multiplier applied to the difference (X_k - x_c).

    Returns:
    - float: The optimal value of the objective function.
    - np.ndarray: The optimal matrix M.
    - np.ndarray: The optimal vector y.
    """
    if rho < 0:
        raise ValueError("rho cannot be negative")

    n = x_c.shape[0]
    K = len(X_k)

    M = cp.Variable((n, n), symmetric=True)
    y = cp.Variable(K, nonneg=True) if K > 0 else None

    constraints = [M >> 0]

    if K > 0:
        for k in range(K):
            x_k = X_k[k]
            constraint_expr = rho * cp.quad_form(x_k - x_c, M) + 9 * K / 10 * y[k]
            constraints.append(constraint_expr >= 1)

    objective = cp.Minimize(cp.trace(M) + cp.norm(y, 1)) if y is not None else cp.Minimize(cp.trace(M))

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=True)  # Use the SCS solver, and enable verbose output for debugging

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise Exception(f"Problem not solved optimally. Status: {prob.status}")

    return prob.value, M.value, y.value if y is not None else np.array([])


if __name__ == "__main__":
    # Example usage
    n_ = 5  # Dimension of vectors
    x_c_ = np.random.randn(n_, 1)  # Central vector
    X_k_ = [np.random.randn(n_, 1) for _ in range(10)]  # List of vectors
    rho_ = 1.5  # Constant multiplier

    optimal_value_, M_opt_, y_opt_ = solve_sdp(x_c_, X_k_, rho_)
    print("The optimal value is:", optimal_value_)
    print("Optimal M is:")
    print(M_opt_)
    print("Optimal y is:")
    print(y_opt_)
