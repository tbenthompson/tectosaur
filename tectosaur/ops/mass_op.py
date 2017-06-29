import tectosaur.geometry as geometry
from tectosaur.quadrature import gauss2d_tri
import numpy as np
import scipy.sparse

class MassOp:
    #TODO: Can this just be done using interp_galerkin_mat??
    def __init__(self, nq, pts, tris):
        qx, qw = gauss2d_tri(nq)
        self.mat = scipy.sparse.dok_matrix((tris.shape[0] * 9, tris.shape[0] * 9))

        tri_pts = pts[tris]
        basis = geometry.linear_basis_tri_arr(qx)

        unscaled_normals = geometry.unscaled_normals(tri_pts)
        jacobians = geometry.jacobians(unscaled_normals)

        for b1 in range(3):
            for b2 in range(3):
                basis_factor = sum(qw * (basis[:,b1]*basis[:,b2]))
                for i in range(tris.shape[0]):
                    entry = jacobians[i] * basis_factor
                    for d in range(3):
                        self.mat[9 * i + 3 * b1 + d, 9 * i + 3 * b2 + d] = entry
        self.shape = self.mat.shape
        self.mat = self.mat.tocsr()

    def dot(self, v):
        return self.mat.dot(v)

    def nearfield_dot(self, v):
        return self.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.dot(v)

    def farfield_dot(self, v):
        shape = [self.shape[0]]
        shape.extend(v.shape[1:])
        return np.zeros(shape)
