import numpy as np

def matrix_inverse_with_adj(matrix):
    """
    Compute the inverse of a square matrix using the adjugate method.
    Parameters:
        matrix (numpy.ndarray): The square matrix to invert.
    Returns:
        numpy.ndarray: The inverse of the matrix.
    """
    # Step 1: Compute the determinant
    det = np.linalg.det(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")

    # Step 2: Compute the matrix of cofactors
    cofactors = np.zeros(matrix.shape)
    n = matrix.shape[0]
    for i in range(n):
        for j in range(n):
            # Compute the minor
            minor = np.delete(np.delete(matrix, i, axis=0), j, axis=1)
            cofactors[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)

    # Step 3: Compute the adjugate (transpose of the cofactor matrix)
    adjugate = cofactors.T

    # Step 4: Compute the inverse
    inverse = adjugate / det
    return inverse

# Example usage
A = np.array([
    [4, 7, 2],
    [3, 6, 1],
    [2, 5, 9]
])

A_inverse = matrix_inverse_with_adj(A)
print("Matrix A:")
print(A)
print("\nInverse of A:")
print(A_inverse)
