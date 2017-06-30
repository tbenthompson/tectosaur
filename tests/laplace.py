from tectosaur.util.geometry import *
import numpy as np

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
    K = (1.0 / 4.0 / np.pi) * ((1 / R ** 3) - (3.0 * eps ** 2 / R ** 5))
    return obsbasis[i, :] * srcbasis[j, :] * K

