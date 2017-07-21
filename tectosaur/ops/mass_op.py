import tectosaur.util.geometry as geometry
from tectosaur.util.quadrature import gauss2d_tri
import numpy as np
import scipy.sparse

from cppimport import cppimport
_mass_op = cppimport("tectosaur.ops._mass_op")

class MassOp:
    def __init__(self, nq, pts, tris):
        qx, qw = gauss2d_tri(nq)

        tri_pts = pts[tris]
        basis = geometry.linear_basis_tri_arr(qx)

        unscaled_normals = geometry.unscaled_normals(tri_pts)
        jacobians = geometry.jacobians(unscaled_normals)

        basis_factors = []
        for b1 in range(3):
            for b2 in range(3):
                basis_factors.append(np.sum(qw * (basis[:,b1]*basis[:,b2])))
        basis_factors = np.array(basis_factors)
        rows, cols, vals = _mass_op.build_op(basis_factors, jacobians)

        n_rows = tris.shape[0] * 9
        self.shape = (n_rows, n_rows)
        self.mat = scipy.sparse.csr_matrix((vals, (rows, cols)), shape = self.shape)

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
