import numpy as np
from numpy import ndarray


def top_eigenvector(matrix: ndarray) -> ndarray:
    """Compute the top eigenvector of a matrix based on the largest eigenvalue."""
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = np.argmax(np.abs(eigenvalues))
    top_vector = eigenvectors[:, index]
    return top_vector


if __name__ == "__main__":
    # Example
    A = np.array([[1, 2], [3, 4]])
    vector = top_eigenvector(A)
    print("Top Eigenvector:", vector)
