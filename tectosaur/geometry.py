import numpy as np

def linear_basis_tri(xhat, yhat):
    return np.array([1.0 - xhat - yhat, xhat, yhat])

def tri_pt(basis, tri):
    return np.array([
        sum([basis[j] * tri[j][i] for j in range(3)])
        for i in range(3)
    ])

def cross(x, y):
    return np.array([
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0]
    ])

def tri_normal(tri, normalize = False):
    n = cross(
        [tri[2][i] - tri[0][i] for i in range(3)],
        [tri[2][i] - tri[1][i] for i in range(3)]
    )
    if normalize:
        n = n / np.linalg.norm(n)
    return n
