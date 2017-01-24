import numpy as np

import tectosaur.geometry as geometry
import tectosaur.nearfield_op as nearfield_op
from tectosaur.adjacency import rotate_tri
from tectosaur.table_lookup import *
from tectosaur.standardize import standardize, rotation_matrix
from tectosaur.dense_integral_op import DenseIntegralOp

# Most of the tests for the table lookup are indirect through the integral op tests,
# these are just for specific sub-functions

def test_adjacent_theta():
    rho = 0.2
    for i in range(10):
        theta = np.random.rand(1)[0] * 2 * np.pi
        out = get_adjacent_theta(
            np.array([[0,0,0],[1,0,0],[0.5,rho,0.0]]),
            np.array([[0,0,0],[1,0,0],[0.5,rho*np.cos(theta),rho*np.sin(theta)]]),
        )
        np.testing.assert_almost_equal(out, theta)

def random_rotation():
    axis = np.random.rand(3) * 2 - 1.0
    axis /= np.linalg.norm(axis)
    theta = np.random.rand(1) * 2 * np.pi
    R = rotation_matrix(axis, theta)
    return R

def test_internal_angles():
    angles = triangle_internal_angles(np.array([[0,0,0],[1,0,0],[0,1,0]]))
    np.testing.assert_almost_equal(angles, [np.pi / 2, np.pi / 4, np.pi / 4])

def test_get_split_pt():
    split_pt = get_split_pt(np.array([[0,0,0],[1,0,0],[0.4,0.8,0.0]]))
    np.testing.assert_almost_equal(split_pt, [0.5, min_angle_isoceles_height, 0.0])

def test_get_split_pt_rotated():
    R = random_rotation()
    scale = np.random.rand(1) * 10.0
    tri = np.array([[0,0,0],[1,0,0],[0.4,0.8,0.0]])
    rot_tri = scale * R.dot(tri.T).T
    split_pt = get_split_pt(rot_tri)
    rot_correct = scale * R.dot([0.5, min_angle_isoceles_height, 0.0])
    np.testing.assert_almost_equal(split_pt, rot_correct)

def interp_pts_wts_test(pts, wts, f):
    fvs = f(pts)
    max_err = 0
    for i in range(10):
        test_pt = np.random.rand(1, pts.shape[1]) * 2 - 1
        correct = f(test_pt)
        res = fast_lookup.barycentric_evalnd(pts, wts, fvs, test_pt)
        print(test_pt, res, correct)
        err = np.abs(res - correct)
        max_err = max(err, max_err)
    return max_err

def test_coincident_interp_pts_wts():
    pts, wts = coincident_interp_pts_wts(10,10,9)
    f = lambda xs: (np.sin(xs[:,0]) * np.exp(np.cos(xs[:,1]) * xs[:,2]))[:, np.newaxis]
    max_err = interp_pts_wts_test(pts, wts, f)
    print(max_err)

def test_adjacent_interp_pts_wts():
    pts, wts = adjacent_interp_pts_wts(10,9)
    f = lambda xs: (np.sin(xs[:,0]) * np.cos(xs[:,1]))[:, np.newaxis]
    max_err = interp_pts_wts_test(pts, wts, f)
    print(max_err)

def test_separate():
    i = 5
    while i > 0:
        pts = np.random.rand(4,3)

        obs_tri = pts[:3,:]
        src_tri = pts[[1,0,3],:]

        # ensure the random triangles are legal triangles
        if standardize(obs_tri, 20) is None:
            continue
        if standardize(src_tri, 20) is None:
            continue

        pts, obs_set, src_set, obs_basis_tris, src_basis_tris = separate_tris(obs_tri, src_tri)
        obs0_angles = triangle_internal_angles(pts[obs_set[0]])
        src0_angles = triangle_internal_angles(pts[src_set[0]])

        np.testing.assert_almost_equal(obs0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(obs0_angles[1], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[1], np.deg2rad(table_min_internal_angle))
        i -= 1

def test_coincident_lookup():
    np.random.seed(10)
    A = 0.6
    B = 0.4
    pr = np.random.rand(1)[0] * 0.5
    A,B,pr = (0.2919530424374, 0.8318957778585687, 0.25)

    pts = 1.0 * np.array([[0,0,0],[1,0,0],[A,B,0]], dtype = np.float64)
    tris = np.array([[0,1,2]])
    eps_scale = np.sqrt(np.linalg.norm(tri_normal(pts)))
    eps = 0.02 * (2.0 ** -np.arange(4))# / eps_scale
    op = DenseIntegralOp(
        eps, 20, 15, 10, 3, 10, 3.0, 'H', 1.0, pr,
        pts, tris, remove_sing = True
    )
    op2 = DenseIntegralOp(
        eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr,
        pts, tris, use_tables = True, remove_sing = True
    )
    print(op.mat[0,0],op2.mat[0,0])
    err = np.abs((op.mat[0,0] - op2.mat[0,0]) / op.mat[0,0])
    assert(err < 0.05)


def test_adjacent_table_lookup():
    # np.random.seed(12)
    K = 'H'
    # We want theta in [0, 3*pi/2] because the old method doesn't work for
    # theta >= 3*np.pi/2

    # theta=2.1719140566792428,pr=0.34567085809127246
    # is a node of the interpolation, so using this
    # theta eliminates any inteporlation error for the moment.
    theta = np.random.rand(1)[0] * 1.3 * np.pi + 0.2
    pr = np.random.rand(1)[0] * 0.5
    non_standard_loc = True
    alpha = 2
    beta = 2

    theta = 2.902453144249991
    pr = 0.461939765

    H = min_angle_isoceles_height
    pts = np.array([
        [0,0,0],[1,0,0],
        [0.5,alpha*H*np.cos(theta),alpha*H*np.sin(theta)],
        [0.5,beta*H,0]
    ])

    if non_standard_loc:
        scale = np.random.rand(1)[0] * 3
        translation = np.random.rand(3)
        R = random_rotation()
        print(theta, pr, scale, translation)
        pts = (translation + R.dot(pts.T).T * scale).copy()

    tris = np.array([[0,1,3],[1,0,2]])

    eps = 0.01 * (2.0 ** -np.arange(4))
    eps_scale = np.sqrt(np.linalg.norm(tri_normal(pts[tris[0]])))

    op2 = DenseIntegralOp(
        eps, 3, 3, 10, 3, 10, 3.0, K, 1.0, pr,
        pts, tris, use_tables = True, remove_sing = True
    )
    print(op2.mat[0:9,9])
    op = DenseIntegralOp(
        eps, 15, 23, 10, 3, 10, 3.0, K, 1.0, pr,
        pts, tris, remove_sing = True
    )
    print(op.mat[0:9,9])
    err = np.abs((op.mat[0,9] - op2.mat[0,9]) / op.mat[0,9])

    # import cppimport
    # adaptive_integrate = cppimport.imp('tables.adaptive_integrate').adaptive_integrate
    # import tectosaur.quadrature as quad
    # from tectosaur.limit import limit
    # rho_order = 50
    # rho_gauss = quad.gaussxw(rho_order)
    # Is = []
    # for e in eps:
    #     rho_q = quad.sinh_transform(rho_gauss, -1, e * 2)
    #     Is.append(adaptive_integrate.integrate_adjacent(
    #         K, pts[tris[1]].tolist(), pts[tris[0]].tolist(), 0.001, e,
    #         1.0, pr, rho_q[0].tolist(), rho_q[1].tolist()
    #     ))
    # Is = np.array(Is)
    # limits = np.empty(81)
    # for i in range(81):
    #     # limits[i] = limit(eps, Is[:, i], False)
    #     val, log_coeff = limit(eps, Is[:, i], True)
    #     limits[i] = val + np.log(eps_scale) * log_coeff
    # print(limits[:9].tolist())
    # import ipdb; ipdb.set_trace()

    # assert(err < 0.05)

def test_sub_basis():
    xs = np.linspace(0.0, 1.0, 11)
    for x in xs:
        pts = np.array([[0,0],[1,0],[0.5,0.5],[0,1]])
        I1 = sub_basis(np.ones((3,3,3,3)), pts[[0,1,2]], pts[[0,1,3]])
        I2 = sub_basis(np.ones((3,3,3,3)), pts[[0,2,3]], pts[[0,1,3]])
        result = I1 + I2
        np.testing.assert_almost_equal(result, 2.0)

def test_sub_basis_identity():
    A = np.random.rand(81).reshape((3,3,3,3))
    B = sub_basis(A, np.array([[0,0],[1,0],[0,1]]), np.array([[0,0],[1,0],[0,1]]))
    np.testing.assert_almost_equal(A, B)

def test_sub_basis_rotation():
    A = np.random.rand(81).reshape((3,3,3,3))
    B = sub_basis(A, np.array([[0,0],[1,0],[0,1]]), np.array([[0,1],[0,0],[1,0]]))
    np.testing.assert_almost_equal(A[:,:,[1,2,0],:], B)
