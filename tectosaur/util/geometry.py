import numpy as np

from tectosaur.util.cpp import imp
_geometry = imp('tectosaur.util._geometry')
get_edge_lens = _geometry.get_edge_lens
get_longest_edge = _geometry.get_longest_edge
vec_angle = _geometry.vec_angle
triangle_internal_angles = _geometry.triangle_internal_angles

def random_rotation():
    axis = np.random.rand(3) * 2 - 1.0
    axis /= np.linalg.norm(axis)
    theta = np.random.rand(1) * 2 * np.pi
    R = np.array(_geometry.rotation_matrix(axis.tolist(), theta[0]))
    return R

def projection(V, b):
    return (V.dot(b) * b) / (np.linalg.norm(b) ** 2)

def linear_basis_tri(xhat, yhat):
    return np.array([1.0 - xhat - yhat, xhat, yhat])

def linear_basis_tri_arr(pts):
    return np.array([1 - pts[:, 0] - pts[:, 1], pts[:, 0], pts[:, 1]]).T

def unscaled_normals(tri_pts):
    return np.cross(
        tri_pts[:,2,:] - tri_pts[:,0,:],
        tri_pts[:,2,:] - tri_pts[:,1,:]
    )

def jacobians(unscaled_normals):
    return np.linalg.norm(unscaled_normals, axis = 1)

def tri_area(tri):
    return np.linalg.norm(unscaled_normals(tri[np.newaxis,:,:]), axis = 1) / 2.0

#TODO: Replace tri_pt with this.
def element_pt(basis, el):
    return np.array([
        sum([basis[j] * el[j][i] for j in range(el.shape[0])])
        for i in range(el.shape[0])
    ])

def tri_pt(basis, tri):
    return element_pt(basis, tri)

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
