import numpy as np

def projection(V, b):
    return (V.dot(b) * b) / (np.linalg.norm(b) ** 2)

def vec_angle(v1, v2):
    v1L = np.linalg.norm(v1)
    v2L = np.linalg.norm(v2)
    v1d2 = v1.dot(v2)
    arg = min(max(v1d2 / (v1L * v2L),-1),1)
    return np.arccos(arg)

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

def tri_pt(basis, tri):
    return np.array([
        sum([basis[j] * tri[j][i] for j in range(3)])
        for i in range(3)
    ])

def xyhat_from_pt(pt, tri):
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    pt_trans = pt - tri[0]
    xhat, yhat = np.linalg.lstsq(np.array([v1,v2]).T, pt_trans)[0]
    assert(xhat + yhat <= 1.0 + 1e-15)
    assert(xhat >= -1e-15)
    assert(yhat >= -1e-15)

    return xhat, yhat

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

class Side:
    front = 0
    behind = 1
    intersect = 2

def which_side_point(tri, pt):
    normal = tri_normal(tri)
    dot_val = (pt - tri[0]).dot(normal)
    if dot_val > 0:
        return Side.front
    elif dot_val < 0:
        return Side.behind
    else:
        return Side.intersect

def segment_side(sides):
    if sides[0] == sides[1]:
        return sides[0]
    elif sides[0] == Side.intersect:
        return sides[1]
    elif sides[1] == Side.intersect:
        return sides[0]
    else:
        return Side.intersect

def tri_side(s):
    edge0 = segment_side([s[0], s[1]]);
    edge1 = segment_side([s[0], s[2]]);
    edge2 = segment_side([s[1], s[2]]);
    if edge0 == Side.intersect and edge1 == edge2:
        return edge1;
    if edge1 == Side.intersect and edge2 == edge0:
        return edge2;
    if edge2 == Side.intersect and edge0 == edge1:
        return edge0;
    return edge0;
