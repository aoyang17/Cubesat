import numpy as np

A = 2 * np.ones((3, 5))
B = 3 * np.ones((3, 5))

# print(A)
# print(B)

C = np.einsum('ni,ni->i', A, B)
print(C)

D = np.repeat(np.arange(4), 3)
print(D)
E = np.arange(3 * 4)
print(E)