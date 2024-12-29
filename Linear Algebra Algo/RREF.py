import numpy as np

def rref(matrix):
    """
    Perform Gaussian elimination to compute the Reduced Row-Echelon Form (RREF)
    of a matrix.

    Parameters:
        matrix (numpy.ndarray): The input matrix to be reduced.
    
    Returns:
        numpy.ndarray: The matrix in reduced row-echelon form.
    """
    # Convert to a float type for division operations
    matrix = matrix.astype(float)
    rows, cols = matrix.shape
    pivot_row = 0

    for col in range(cols):
        # Find the pivot (non-zero) element in the current column
        max_row = pivot_row + np.argmax(np.abs(matrix[pivot_row:, col]))
        if matrix[max_row, col] != 0:
            # Swap the current row with the row containing the maximum element
            matrix[[pivot_row, max_row]] = matrix[[max_row, pivot_row]]
            
            # Scale the pivot row to make the pivot element equal to 1
            matrix[pivot_row] = matrix[pivot_row] / matrix[pivot_row, col]

            # Eliminate all other entries in the current column
            for r in range(rows):
                if r != pivot_row:
                    matrix[r] -= matrix[r, col] * matrix[pivot_row]
            
            # Move to the next row
            pivot_row += 1
            
        # Stop if we have processed all rows
        if pivot_row == rows:
            break

    return matrix

# Example usage
A = np.array([
    [2, 1, -1, 8],
    [3, -1, 2, 1],
    [1, 2, -3, -2]
])

rref_matrix = rref(A)
print("Reduced Row-Echelon Form:")
print(rref_matrix)