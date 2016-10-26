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

    pts = 1.0 * np.array([[0,0,0],[1,0,0],[A,B,0]], dtype = np.float64)
    tris = np.array([[0,1,2]])
    eps = 0.08 * (2.0 ** -np.arange(4))
    op = DenseIntegralOp(
        eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr,
        pts, tris, remove_sing = True
    )
    op2 = DenseIntegralOp(
        eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr,
        pts, tris, use_tables = True, remove_sing = True
    )
    err = np.abs((op.mat[0,0] - op2.mat[0,0]) / op.mat[0,0])
    assert(err < 0.05)


def test_adjacent_table_lookup():
    # np.random.seed(12)
    K = 'H'
    for i in range(1):
        # We want theta in [0, 3*pi/2] because the old method doesn't work for
        # theta >= 3*np.pi/2

        # theta=2.1719140566792428,pr=0.34567085809127246
        # is a node of the interpolation, so using this
        # theta eliminates any inteporlation error for the moment.
        theta = 2.1719140566792428#np.random.rand(1)[0] * 1.3 * np.pi + 0.2
        pr = 0.34567085809127246#np.random.rand(1)[0] * 0.5

        # scale = np.random.rand(1)[0] * 3
        # translation = np.random.rand(3)
        # R = random_rotation()
        # print(theta, pr, scale, translation)

        H = min_angle_isoceles_height
        pts = np.array([
            [0,0,0],[1,0,0],
            [0.5,1*H*np.cos(theta),1*H*np.sin(theta)],
            [0.5,1*H,0]
        ])
        # pts = (translation + R.dot(pts.T).T * scale).copy()
        tris = np.array([[0,1,3],[1,0,2]])

        result, pairs = adjacent_table(
            K, 1.0, pr, pts, np.array([tris[0]]), np.array([tris[1]]), True
        )
        print(result[0,0,0,0,0])

        eps_scale = np.sqrt(np.linalg.norm(tri_normal(pts[tris[0]])))
        print("eps_scale: " + str(eps_scale))
        eps = 0.08 * (2.0 ** -np.arange(4)) / 40
        op = DenseIntegralOp(
            eps, 15, 30, 10, 3, 10, 3.0, K, 1.0, pr,
            pts, tris, remove_sing = True
        )
        print(op.mat[0,9])

        nq = 7
        for i,pts,ot,st,obt,sbt in pairs:
            otA = np.linalg.norm(geometry.tri_normal(pts[ot]))
            stA = np.linalg.norm(geometry.tri_normal(pts[st]))
            print(otA * stA)
            if otA * stA < 1e-10:
                continue

            ot_clicks = 0
            st_clicks = 0
            for d in range(3):
                matching_vert = np.where(st == ot[d])[0]
                if matching_vert.shape[0] > 0:
                    ot_clicks = d
                    st_clicks = matching_vert[0]
                    break
                if d == 2:
                    print("NOTTOUCHING")

            ot_rot = rotate_tri(ot_clicks)
            st_rot = rotate_tri(st_clicks)
            Iv = nearfield_op.vert_adj(
                nq, K, 1.0, pr, pts.copy(),
                np.array([ot[ot_rot]]),
                np.array([st[st_rot]])
            )
            Iv = sub_basis(Iv[0], obt[ot_rot], sbt[st_rot])
            result += Iv

        est = result[0,0,0,0,0]
        correct = op.mat[0,9]
        err = np.abs((est - correct) / correct)
        print(est, correct, err)
        assert(err < 0.05)

        # print(opE[0,0,0,0,0] + opV[0,0,0,0,0])
        # print(op.mat[0,9])

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

def test_summation():
    K = 'H'

    eps = 0.08 * (2.0 ** -np.arange(4))
    abc = 0.5
    pts = np.array([[0,0,0],[1,0,0],[abc,1 - abc,0],[0,1,0],[0,-1,0],[0.5,-0.5,0]])
    obs_basis = pts[:,:2]
    src_basis = np.array([[1,0],[0,0],[0,0],[0,0],[0,1],[0.0,0.5]])

    # tris = np.array([[0,2,3],[0,4,1]])
    tris = np.array([[0,1,3],[1,0,4]])

    nqe = 20
    nqv = 10

    op = DenseIntegralOp(eps, 15, nqe, nqv, nqv, nqv, 3.0, K, 1.0, 0.25, pts, tris)

    # Split obs
    obs_e_adj_tri = [0,1,2]
    src_e_adj_tri = [1,0,4]
    obs_v_adj_tri = [0,2,3]
    src_v_adj_tri = [0,4,1]
    # Split src
    # obs_e_adj_tri = [0,1,3]
    # src_e_adj_tri = [1,0,5]
    # obs_v_adj_tri = [0,1,3]
    # src_v_adj_tri = [0,4,5]
    opE = nearfield_op.edge_adj(
        nqe, eps * np.sqrt(1.0 / (1 - abc)), K, 1.0, 0.25, pts,
        np.array([obs_e_adj_tri]), np.array([src_e_adj_tri]),
    )
    opV = nearfield_op.vert_adj(
        nqv, K, 1.0, 0.25, pts,
        np.array([obs_v_adj_tri]), np.array([src_v_adj_tri]),
    )

    opEsub = sub_basis(opE[0], obs_basis[obs_e_adj_tri], src_basis[src_e_adj_tri])
    opVsub = sub_basis(opV[0], obs_basis[obs_v_adj_tri], src_basis[src_v_adj_tri])

    A = op.mat[:9,9:].reshape((3,3,3,3))
    B = opEsub + opVsub

    print('')
    print(B[:,0,:,0])
    print(A[:,0,:,0])

def test_edge_adj_split():
    K = 'H'
    nqe = 15
    nqv = 7

    eps = 0.08 * (2.0 ** -np.arange(4))
    for abc in np.linspace(0.1,0.9,5):
        pts = np.array([[0,0,0],[1,0,0],[abc,1-abc,0],[0,1,0],[0,-1,0],[0.5,-0.5,0],[0.25,0.75,0]])
        obs_basis = pts[:,:2]
        src_basis = np.array([[1,0],[0,0],[0,0],[0,0],[0,1],[0.0,0.5]])

        tris = np.array([[0,1,3],[0,4,1]])
        op = DenseIntegralOp(eps, 15, nqe, nqv, nqv, nqv, 3.0, K, 1.0, 0.25, pts, tris)

        tris2 = np.array([[0,1,2],[0,2,3],[0,4,1]])
        op2 = DenseIntegralOp(eps * np.sqrt(1.0 / (1 - abc)), 15, nqe, nqv, nqv, nqv, 3.0, K, 1.0, 0.25, pts, tris2)

        A = op.mat[0,9]
        B = op2.mat[0,18] + op2.mat[9,18]
        err = np.abs((A - B) / A)
        assert(err < 0.01)
