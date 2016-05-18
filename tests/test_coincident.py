from tectosaur.coincident import *
from tectosaur.quadrature import *
from tectosaur.basis import *

def test_simple():
    eps = 0.01

    q = coincident_quad(0.01, 10, 10, 10, 10)

    result = quadrature(lambda x: x[:, 2], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 4)

    result = quadrature(lambda x: x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 4)

    result = quadrature(lambda x: x[:, 2] * x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 48.0, 4)

def test_kernel():
    tri = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]
    eps = 0.01
    i = 0
    j = 0

    def K(pts):
        obsxhat = pts[:, 0]
        obsyhat = pts[:, 1]
        srcxhat = pts[:, 2]
        srcyhat = pts[:, 3]

        obsbasis = linear_basis_tri(obsxhat, obsyhat)
        srcbasis = linear_basis_tri(srcxhat, srcyhat)

        obsn = tri_normal(tri)

        obspt = (tri_pt(obsbasis, tri).T - eps * obsn).T
        srcpt = tri_pt(srcbasis, tri)

        R = np.sqrt(
            (obspt[0,:] - srcpt[0,:]) ** 2 +
            (obspt[1,:] - srcpt[1,:]) ** 2 +
            (obspt[2,:] - srcpt[2,:]) ** 2
        )
        K = (1.0 / 4.0 / np.pi) * ((1 / R ** 3) - (3.0 * eps ** 2 / R ** 5))
        return obsbasis[i, :] * srcbasis[j, :] * K

    exact = quadrature(K, coincident_quad(eps, 18, 15, 15, 18))
    # exact = -0.33151084315265733
    def tryit(n1,n2,n3,n4):
        q = coincident_quad(eps, n1, n2, n3, n4)
        result = quadrature(K, q)
        print(np.abs((result - exact) / exact))
        print(q[0].shape)
    tryit(6,5,4,10)
    tryit(9,7,5,10)
    # np.testing.assert_almost_equal(result, exact, 4)
