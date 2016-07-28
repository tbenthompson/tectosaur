from tectosaur.quadrature import *
from tectosaur.triangle_rules import *
from tectosaur.geometry import *

def laplace(tri1, tri2, i, j, eps, pts):
    obsxhat = pts[:, 0]
    obsyhat = pts[:, 1]
    srcxhat = pts[:, 2]
    srcyhat = pts[:, 3]

    obsbasis = linear_basis_tri(obsxhat, obsyhat)
    srcbasis = linear_basis_tri(srcxhat, srcyhat)

    obsn = tri_normal(tri1)
    srcn = tri_normal(tri2)

    obspt = (tri_pt(obsbasis, tri1).T - np.outer(eps, obsn)).T
    srcpt = tri_pt(srcbasis, tri2)

    R = np.sqrt(
        (obspt[0,:] - srcpt[0,:]) ** 2 +
        (obspt[1,:] - srcpt[1,:]) ** 2 +
        (obspt[2,:] - srcpt[2,:]) ** 2
    )
    K = (1.0 / 4.0 / np.pi) * ((1 / R ** 2) - (3.0 * eps ** 2 / R ** 4))
    return obsbasis[i, :] * srcbasis[j, :] * K


def test_coincident_laplace():
    tri = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]).astype(np.float64)

    eps = 0.01
    for i in range(3):
        for j in range(i + 1):
            K = lambda pts: laplace(tri, tri, i, j, np.float64(eps), pts.astype(np.float64))
            # q_accurate = coincident_quad(eps, 28, 28, 28, 28)
            # exact = quadrature(K, q_accurate)
            def tryit(n1,n2,n3,n4):
                q = coincident_quad(eps, n1, n2, n3, n4)
                p = q[1].argsort()[::-1]
                qxs = q[0][p,:].astype(np.float64)
                qws = q[1][p].astype(np.float64)
                # result = quadrature(K, (qxs, qws))
                result = quadrature(K, q)
                print(result)
                return result
            exactish = tryit(18,18,18,18)
            assert(np.abs((tryit(8,7,6,11) - exactish)/exactish) < 0.005)
            assert(np.abs((tryit(14,9,7,10) - exactish)/exactish) < 0.0005)
            assert(np.abs((tryit(13,12,11,13) - exactish)/exactish) < 0.00005)
            print(tryit(15,15,15,15) - tryit(15,15,25,15))
            # assert(tryit(13,12,11,13) < 0.000005)
test_coincident_laplace()
