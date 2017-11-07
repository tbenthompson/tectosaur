import numpy as np

#TODO: Should this module even exist? Can it be deleted and spread out? Or should it simply be a shim over the C++ variants of each of these functions?
import cppimport
_geometry = cppimport.cppimport('tectosaur.util._geometry')

def random_rotation():
    axis = np.random.rand(3) * 2 - 1.0
    axis /= np.linalg.norm(axis)
    theta = np.random.rand(1) * 2 * np.pi
    R = np.array(_geometry.rotation_matrix(axis.tolist(), theta[0]))
    return R

def projection(V, b):
    return (V.dot(b) * b) / (np.linalg.norm(b) ** 2)

#TODO: Duplicated with standardize c++ stuff
def vec_angle(v1, v2):
    v1L = np.linalg.norm(v1)
    v2L = np.linalg.norm(v2)
    v1d2 = v1.dot(v2)
    arg = min(max(v1d2 / (v1L * v2L),-1),1)
    return np.arccos(arg)

#TODO: Duplicated with standardize c++ stuff
def get_edge_lens(tri):
    L0 = np.sum((tri[1,:] - tri[0,:])**2)
    L1 = np.sum((tri[2,:] - tri[1,:])**2)
    L2 = np.sum((tri[2,:] - tri[0,:])**2)
    return L0, L1, L2

#TODO: Duplicated with standardize c++ stuff
def get_longest_edge(lens):
    if lens[0] >= lens[1] and lens[0] >= lens[2]:
        return 0
    elif lens[1] >= lens[0] and lens[1] >= lens[2]:
        return 1
    elif lens[2] >= lens[0] and lens[2] >= lens[1]:
        return 2

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
