from tectosaur.triangle_rules import *
from tectosaur.quadrature import *
from tectosaur.geometry import *
import tectosaur.mesh as mesh
from slow_test import slow
from laplace import laplace

from tectosaur.gpu import load_gpu
import pycuda.driver as drv

def test_simple():
    eps = 0.01

    q = coincident_quad(0.01, 10, 10, 10, 10)

    result = quadrature(lambda x: x[:, 2], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 2)

    result = quadrature(lambda x: x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 2)

    result = quadrature(lambda x: x[:, 2] * x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 48.0, 2)

@slow
def test_kernel():
    tri = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]

    eps = 0.01
    for i in range(3):
        for j in range(i):
            K = lambda pts: laplace(tri, tri, i, j, eps, pts)
            exact = quadrature(K, coincident_quad(eps, 18, 15, 15, 18))
            def tryit(n1,n2,n3,n4):
                q = coincident_quad(eps, n1, n2, n3, n4)
                result = quadrature(K, q)
                return np.abs((result - exact) / exact)
            assert(tryit(7,6,4,10) < 0.005)
            assert(tryit(14,9,7,10) < 0.0005)
            assert(tryit(13,12,11,13) < 0.00005)

def test_coincident_gpu():
    N = 16

    pts, tris = mesh.make_strip(N)
    pts = np.array(pts).astype(np.float32)
    tris = np.array(tris).astype(np.int32)
    q = richardson_quad([0.1, 0.01], lambda e: coincident_quad(e, 8, 8, 5, 10))
    qx = q[0].astype(np.float32)
    qw = q[1].astype(np.float32)

    result = np.empty((tris.shape[0], 3, 3, 3, 3)).astype(np.float32)

    mod = load_gpu('tectosaur/integrals.cu')
    coincident = mod.get_function('single_pairsSH')

    block = (32, 1, 1)
    grid = (int(tris.shape[0]/block[0]),1)
    runtime = coincident(
        drv.Out(result),
        np.int32(q[0].shape[0]),
        drv.In(qx),
        drv.In(qw),
        drv.In(pts),
        drv.In(tris),
        drv.In(tris),
        np.float32(1.0),
        np.float32(0.25),
        block = block,
        grid = grid,
        time_kernel = True
    )
    # Simple golden master test
    result2 = np.load('tests/correct_coincident.npy')
    np.testing.assert_almost_equal(result, result2)
