import matplotlib.pyplot as plt
import numpy as np
import cppimport
import tectosaur.quadrature as quad
from tectosaur.geometry import tri_normal
import interpolate
adaptive_integrate = cppimport.imp('adaptive_integrate')

tri1 = [[0,0,0],[1,0,0],[0,1,0]]

n_theta = 13
theta = np.linspace(0, 2 * np.pi, n_theta)[1:-1]
y = np.cos(theta)
z = np.sin(theta)
rho = 0.5 * np.tan(np.deg2rad(20))

def remove_proj(V, b):
    return V - (V.dot(b) * b) / np.linalg.norm(b)

def vec_angle(v1, v2):
    return np.arccos(v1.dot(v2) / np.linalg.norm(v1) / np.linalg.norm(v2))

def get_offset(tri1, tri2):
    p = tri1[1] - tri1[0]
    L1 = tri1[2] - tri1[0]
    L2 = tri2[2] - tri2[0]
    T1 = remove_proj(L1, p)
    T2 = remove_proj(L2, p)
    n1 = tri_normal(tri1, normalize = True)

    theta = vec_angle(T1, T2)
    samedir = n1.dot(T2 - T1) > 0

    v = n1
    if not samedir and theta < np.pi * 1.5:
        v = -(L1 + L2) / 2.0
        v /= np.linalg.norm(v)
    return v

def plot_offset(Y, Z, offset):
    eps = 0.05
    plt.plot([0, Y], [0, Z])
    plt.plot([0, 1], [0, 0])
    plt.plot([-offset[1] * eps, 1 - offset[1] * eps], [-offset[2] * eps, -offset[2] * eps])
    plt.xlim([-1.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.show()

for i in range(theta.shape[0]):
    Y = y[i]
    Z = z[i]
    print(Y,Z)

    tri2 = [[1,0,0],[0,0,0],[0,Y,Z]]
    offset = get_offset(np.array(tri1), np.array(tri2))

    K = "U"
    rho_order = 40
    eps = 0.1
    tol = 0.01
    rho_gauss = quad.gaussxw(rho_order)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)

    res = np.array(adaptive_integrate.integrate_adjacent(
        K, tri1, tri2, offset.tolist(),
        tol, eps, 1.0, 0.25, rho_q[0].tolist(), rho_q[1].tolist()
    ))
    print(res[0])

    from tectosaur.dense_integral_op import DenseIntegralOp
    op = DenseIntegralOp(
        [eps], 5, 20, 5, 5, 5, 5, K, 1.0, 0.25,
        np.vstack((tri1, tri2)), np.array([[0,1,2],[1,0,5]])
    )
    print(op.mat[0,9])
    #
    # A = op.mat[:9,9:].reshape((81))
    # B = res.reshape((81))
    # for i in range(81):
    #     print(A[i], B[i], A[i] - B[i])
