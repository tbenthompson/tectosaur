import numpy as np
import cppimport
import tectosaur.quadrature as quad
adaptive_integrate = cppimport.imp('adaptive_integrate')



tri1 = [[0,0,0],[1,0,0],[0,1,0]]

def find_adj_pts(tri):

    adjacent_pt


n_theta = 13
theta = np.linspace(0,2 * np.pi, 13)
rho = 0.5 * np.tan(np.deg2rad(20))
y = rho * np.cos(theta)
z = rho * np.sin(theta)
print(y,z)


for i in range(n_theta):
    tri2 = [[1,0,0],[0,0,0],[0,-1,0]]
    rho_order = 40
    rho_gauss = quad.gaussxw(rho_order)
    eps = 0.0001
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)

    res = np.array(adaptive_integrate.integrate_adjacent(
        "H", tri1, tri2,
        0.01, eps, 1.0, 0.25, rho_q[0].tolist(), rho_q[1].tolist()
    ))
    print(res[0])

# from tectosaur.dense_integral_op import DenseIntegralOp
# op = DenseIntegralOp(
#     [eps], 5, 20, 5, 5, 5, 5, "H", 1.0, 0.25,
#     np.vstack((tri1, tri2)), np.array([[0,1,2],[1,0,5]])
# )
#
# A = op.mat[:9,9:].reshape((81))
# B = res.reshape((81))
# for i in range(81):
#     print(A[i], B[i], A[i] - B[i])
