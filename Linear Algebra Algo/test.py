import numpy as np

A = np.array([
    [ 1 , 2 ,-1, -4],
    [ 2 , 3 ,-1 ,-11],
    [-2 , 0, -3,  22]
])
rows, cols = A.shape

print(np.abs(A[0:, 0]))
print(np.argmax(np.abs(A[0:, 0])))
print(A[0])