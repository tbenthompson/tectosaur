import numpy as np

import tectosaur.quadrature as quad
import cppimport
adaptive_integrate = cppimport.imp('tectosaur.adaptive_integrate').adaptive_integrate
adaptive_integrate2 = cppimport.imp('tectosaur.adaptive_integrate2').adaptive_integrate2

from test_decorators import golden_master, slow


@slow
@golden_master
def test_coincident_integral():
    tri = [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]]
    eps = 0.01
    rho_gauss = quad.gaussxw(50)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    theta_q = quad.gaussxw(50)
    res = adaptive_integrate.integrate_coincident(
        'H', tri, 0.001, eps, 1.0, 0.25,
        rho_q[0].tolist(), rho_q[1].tolist(),
        theta_q[0].tolist(), theta_q[1].tolist()
    )
    return np.array(res)

@slow
@golden_master
def test_adjacent_integral():
    obs_tri = [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]]
    src_tri = [[1.2, 0, 0], [0, 0, 0], [0.3, -1.1, 0]]
    eps = 0.01
    rho_gauss = quad.gaussxw(50)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    res = adaptive_integrate.integrate_adjacent(
        'H', obs_tri, src_tri, 0.001, eps, 1.0, 0.25,
        rho_q[0].tolist(), rho_q[1].tolist()
    )
    return np.array(res)

def f1d(x):
    return np.sin(x)

def test_adaptive_1d():
    res = adaptive_integrate2.integrate(f1d, [0], [1], 0.01)
    np.testing.assert_almost_equal(res[0], 1 - np.cos(1))

def f2d(x):
    return np.array([np.sin(x[:,0])*x[:,1]]).T

def test_adaptive_2d():
    res = adaptive_integrate2.integrate(f2d, [0,0], [1,1], 0.01)
    np.testing.assert_almost_equal(res[0], np.sin(0.5) ** 2)

def vec_out(x):
    return np.array([x[:,0] + 1, x[:,0] - 1]).T

def test_adaptive_vector_out():
    res = adaptive_integrate2.integrate(vec_out, [0], [1], 0.01)
    np.testing.assert_almost_equal(res[0], [1.5, -0.5])
