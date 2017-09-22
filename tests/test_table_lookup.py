import pytest
import numpy as np

import tectosaur.util.geometry as geometry
from tectosaur.mesh.find_near_adj import rotate_tri
import tectosaur.nearfield.nearfield_op as nearfield_op
from tectosaur.nearfield.standardize import standardize, rotation_matrix, BadTriangleError
from tectosaur.nearfield.interpolate import to_interval
from tectosaur.ops.dense_integral_op import DenseIntegralOp

from tectosaur.nearfield.table_lookup import *

from tectosaur.util.test_decorators import golden_master, slow, kernel

float_type = np.float64

def test_find_va_rotations():
    res = fast_lookup.find_va_rotations([1,3,5],[2,3,4])
    np.testing.assert_equal(res, [[1,2,0],[1,2,0]])

    res = fast_lookup.find_va_rotations([3,1,5],[2,3,4])
    np.testing.assert_equal(res, [[0,1,2],[1,2,0]])

    res = fast_lookup.find_va_rotations([1,5,3],[2,4,3])
    np.testing.assert_equal(res, [[2,0,1],[2,0,1]])

def test_adjacent_phi():
    rho = 0.2
    for i in range(10):
        phi = np.random.rand(1)[0] * 2 * np.pi
        out = fast_lookup.get_adjacent_phi(
            [[0,0,0],[1,0,0],[0.5,rho,0.0]],
            [[0,0,0],[1,0,0],[0.5,rho*np.cos(phi),rho*np.sin(phi)]],
        )
        np.testing.assert_almost_equal(out, phi)

def random_rotation():
    axis = np.random.rand(3) * 2 - 1.0
    axis /= np.linalg.norm(axis)
    theta = np.random.rand(1) * 2 * np.pi
    R = np.array(rotation_matrix(axis, theta[0]))
    return R

def test_internal_angles():
    angles = fast_lookup.triangle_internal_angles([[0,0,0],[1,0,0],[0,1,0]])
    np.testing.assert_almost_equal(angles, [np.pi / 2, np.pi / 4, np.pi / 4])

def test_get_split_pt():
    split_pt = fast_lookup.get_split_pt([[0,0,0],[1,0,0],[0.4,0.8,0.0]])
    np.testing.assert_almost_equal(split_pt, [0.5, min_angle_isoceles_height, 0.0])

def test_get_split_pt_rotated():
    for i in range(50):
        R = random_rotation()
        scale = np.random.rand(1) * 10.0
        tri = np.array([[0,0,0],[1,0,0],[np.random.rand(1)[0] * 0.5,np.random.rand(1)[0],0.0]])
        rot_tri = scale * R.dot(tri.T).T
        split_pt = fast_lookup.get_split_pt(rot_tri.tolist())
        rot_correct = scale * R.dot([0.5, min_angle_isoceles_height, 0.0])
        np.testing.assert_almost_equal(split_pt, rot_correct)

def interp_pts_wts_test(pts, wts, f):
    fvs = f(pts)
    max_err = 0
    for i in range(10):
        test_pt = np.random.rand(1, pts.shape[1]) * 2 - 1
        correct = f(test_pt)
        res = fast_lookup.barycentric_evalnd(pts, wts, fvs, test_pt)
        # print(test_pt, res, correct)
        err = np.abs(res - correct)
        max_err = max(err, max_err)
    return max_err

def test_coincident_interp_pts_wts():
    pts, wts = coincident_interp_pts_wts(10,10,9)
    f = lambda xs: (np.sin(xs[:,0]) * np.exp(np.cos(xs[:,1]) * xs[:,2]))[:, np.newaxis]
    max_err = interp_pts_wts_test(pts, wts, f)
    # print(max_err)

def test_adjacent_interp_pts_wts():
    pts, wts = adjacent_interp_pts_wts(10,9)
    f = lambda xs: (np.sin(xs[:,0]) * np.cos(xs[:,1]))[:, np.newaxis]
    max_err = interp_pts_wts_test(pts, wts, f)
    # print(max_err)

def test_separate():
    i = 5
    while i > 0:
        pts = np.random.rand(4,3)

        obs_tri = pts[:3,:]
        src_tri = pts[[1,0,3],:]

        # ensure the random triangles are legal triangles
        try:
            standardize(obs_tri, 20, True)
            standardize(src_tri, 20, True)
        except BadTriangleError:
            continue

        pts, obs_set, src_set, obs_basis_tris, src_basis_tris =\
            fast_lookup.separate_tris(obs_tri.tolist(), src_tri.tolist())
        obs0_angles = fast_lookup.triangle_internal_angles(np.array(pts)[obs_set[0]].tolist())
        src0_angles = fast_lookup.triangle_internal_angles(np.array(pts)[src_set[0]].tolist())

        np.testing.assert_almost_equal(obs0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(obs0_angles[1], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[1], np.deg2rad(table_min_internal_angle))
        i -= 1

def test_sub_basis():
    xs = np.linspace(0.0, 1.0, 11)
    for x in xs:
        pts = np.array([[0,0],[1,0],[x,1-x],[0,1]])
        tri1 = pts[[0,1,2]].tolist()
        tri2 = pts[[0,2,3]].tolist()
        full_tri = pts[[0,1,3]].tolist()
        input = np.ones(81).tolist()
        tri1area = np.linalg.norm(geometry.tri_normal(np.hstack((tri1, np.zeros((3,1))))))
        tri2area = np.linalg.norm(geometry.tri_normal(np.hstack((tri2, np.zeros((3,1))))))
        I1 = np.array(fast_lookup.sub_basis(input, tri1, full_tri))
        I2 = np.array(fast_lookup.sub_basis(input, tri2, full_tri))
        result = tri1area * I1 + tri2area * I2
        np.testing.assert_almost_equal(result, 1.0)

def test_sub_basis_identity():
    A = np.random.rand(81).tolist()
    B = fast_lookup.sub_basis(
        A, [[0,0],[1,0],[0,1]], [[0,0],[1,0],[0,1]]
    )
    np.testing.assert_almost_equal(A, B)

def test_sub_basis_rotation():
    A = np.random.rand(81).reshape((3,3,3,3))
    B = fast_lookup.sub_basis(A.flatten().tolist(), [[0,0],[1,0],[0,1]], [[0,1],[0,0],[1,0]])
    np.testing.assert_almost_equal(A[:,:,[1,2,0],:], np.array(B).reshape((3,3,3,3)))

def coincident_lookup_helper(K, correct_digits, n_tests = 10):
    np.random.seed(113)


    results = []
    for i in range(n_tests):
        try:
            A = np.random.rand(1)[0] * 0.5
            B = np.random.rand(1)[0]
            pr = np.random.rand(1)[0] * 0.5
            scale = np.random.rand(1)[0]
            flip = np.random.rand(1) > 0.5

            params = [1.0, pr]

            R = random_rotation()
            # print(R)

            pts = scale * np.array([[0,0,0],[1,0,0],[A,B,0]], dtype = np.float64)
            pts = R.dot(pts)

            if flip:
                tris = np.array([[0,2,1]])
            else:
                tris = np.array([[0,1,2]])

            op = DenseIntegralOp(
                10, 3, 10, 3.0, K, params,
                pts, tris, float_type
            )
            results.append(op.mat)
        except BadTriangleError as e:
            print("Bad tri: " + str(e.code))
    return np.array(results)

@golden_master(4)
def test_coincident_fast_lookup(request, kernel):
    return coincident_lookup_helper(kernel, 5, 1)

@slow
@golden_master(4)
def test_coincident_lookup(request, kernel):
    return coincident_lookup_helper(kernel, 5)

def adjacent_lookup_helper(K, correct_digits, n_tests = 10):
    np.random.seed(973)


    results = []
    for i in range(n_tests):
        # We want phi in [0, 3*pi/2] because the old method doesn't work for
        # phi >= 3*np.pi/2
        phi = to_interval(min_intersect_angle, 1.4 * np.pi, np.random.rand(1)[0])
        pr = np.random.rand(1)[0] * 0.5

        params = [1.0, pr]
        alpha = np.random.rand(1)[0] * 3 + 1
        beta = np.random.rand(1)[0] * 3 + 1

        scale = np.random.rand(1)[0] * 3
        translation = np.random.rand(3)
        R = random_rotation()

        # print(alpha, beta, phi, pr, scale, translation.tolist(), R.tolist())

        H = min_angle_isoceles_height
        pts = np.array([
            [0,0,0],[1,0,0],
            [0.5,alpha*H*np.cos(phi),alpha*H*np.sin(phi)],
            [0.5,beta*H,0]
        ])

        pts = (translation + R.dot(pts.T).T * scale).copy()

        tris = np.array([[0,1,3],[1,0,2]])

        op = DenseIntegralOp(
            10, 3, 10, 3.0, K, params, pts, tris, float_type
        )
        results.append(op.mat[:9,9:])
    return np.array(results)

@golden_master()
def test_adjacent_fast_lookup(request, kernel):
    return adjacent_lookup_helper(kernel, 5, 1)

@slow
@golden_master()
def test_adjacent_lookup(request, kernel):
    return adjacent_lookup_helper(kernel, 5)

def fault_surface_experiment():
    pts = np.array([[-1, 0, 0], [0, -1, 0], [0, 1, 0], [1, 0, 0], [0, 0, -1]])
    tris = np.array([[0, 1, 2], [2, 1, 3], [1, 2, 4]])
    K = 'elasticT3'
    params = [1.0, 0.25]
    op = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris, float_type)
    A = op.mat[:9,18:]
    B = op.mat[9:18,18:]
    import matplotlib.pyplot as plt
    plt.imshow(np.sign(A) * np.log10(np.abs(A)), interpolation = 'none')
    plt.colorbar()
    plt.figure()
    plt.imshow(np.sign(B) * np.log10(np.abs(B)), interpolation = 'none')
    plt.title('B')
    plt.show()
    import ipdb
    ipdb.set_trace()

def adj_theta_dependence_experiment():
    ts = np.linspace(0.2 * np.pi, 1.8 * np.pi, 20)
    K = 'elasticT3'
    params = [1.0, 0.25]
    v1 = []
    v2 = []
    for theta in ts:
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0],
            [0.5, 0.5 * np.cos(theta), 0.5 * np.sin(theta)]
        ])
        tris = np.array([[0,1,2],[1,0,3]])
        op = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris, float_type)
        v1.append(op.mat[:9,9:].reshape((3,3,3,3)))
        v2.append(op.mat[9:,:9].reshape((3,3,3,3)))
    v1 = np.array(v1)
    v2 = np.array(v2)
    vv1,vv2 = np.load('adj_theta_dependence.npy')
    import matplotlib.pyplot as plt
    for b1 in range(3):
        for d1 in range(3):
            for b2 in range(3):
                for d2 in range(3):
                    plt.plot(ts, v1[:,b1,d1,b2,d2], 'r-')
                    plt.plot(ts, v2[:,b1,d1,b2,d2], 'b-')
                    plt.plot(ts, vv1[:,b1,d1,b2,d2], 'r-.')
                    plt.plot(ts, vv2[:,b1,d1,b2,d2], 'b-.')
                    plt.title(' '.join([str(x) for x in [b1,d1,b2,d2]]))
                    plt.show()

def adj_flipping_experiment():
    theta = np.pi / 2.0
    K = 'elasticT3'
    params = [1.0, 0.25]
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0],
        [0.5, 0.5 * np.cos(theta), 0.5 * np.sin(theta)]
    ])
    print(pts)
    tris = np.array([[0,1,2],[1,0,3]])
    tris2 = np.array([[0,1,2],[0,1,3]])
    op = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris, float_type)
    op2 = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris2, float_type)
    A = op.mat[:9,9:].reshape((3,3,3,3))
    B = op2.mat[:9,9:].reshape((3,3,3,3))

    AG,BG = np.load('flipped_experiment_90.npy')
    AG = AG.reshape((3,3,3,3))
    BG = BG.reshape((3,3,3,3))
    CG = BG.copy()
    CG[:,:,0,:] = BG[:,:,1,:]
    CG[:,:,1,:] = BG[:,:,0,:]
    CG *= -1
    import ipdb
    ipdb.set_trace()

def test_orient_simple():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, -0.5, 0.0]])
    tris = np.array([[0,1,2], [1,0,3]])
    result = fast_lookup.orient_adj_tris(pts, tris, 0, 1)
    assert(result[0] == 0)
    np.testing.assert_almost_equal(result[1], pts[tris[0]])
    assert(result[2] == 0)
    assert(not result[3])
    np.testing.assert_almost_equal(result[4], pts[tris[1]])

def test_orient_rot():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, -0.5, 0.0]])
    tris = np.array([[1,2,0], [1,0,3]])
    result = fast_lookup.orient_adj_tris(pts, tris, 0, 1)
    assert(result[0] == 2)
    np.testing.assert_almost_equal(result[1], pts[[0,1,2]])
    assert(result[2] == 0)
    assert(not result[3])
    np.testing.assert_almost_equal(result[4], pts[tris[1]])

def test_orient_flip():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, -0.5, 0.0]])
    tris = np.array([[0,1,2], [0,1,3]])
    result = fast_lookup.orient_adj_tris(pts, tris, 0, 1)
    assert(result[0] == 0);
    np.testing.assert_almost_equal(result[1], pts[tris[0]])
    assert(result[2] == 0);
    assert(result[3])
    np.testing.assert_almost_equal(result[4], pts[[1,0,3]])

if __name__ == "__main__":
    adj_flipping_experiment()
    # adj_theta_dependence_experiment()
