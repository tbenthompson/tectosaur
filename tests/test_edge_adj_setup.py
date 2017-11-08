import numpy as np
import tectosaur.util.geometry as geometry
from tectosaur.nearfield.table_params import table_min_internal_angle, min_angle_isoceles_height

from tectosaur.util.cpp import imp
standardize = imp('tectosaur.nearfield.standardize')
edge_adj_setup = imp('tectosaur.nearfield.edge_adj_setup')

def test_adjacent_phi():
    rho = 0.2
    for i in range(10):
        phi = np.random.rand(1)[0] * 2 * np.pi
        out = edge_adj_setup.get_adjacent_phi(
            [[0,0,0],[1,0,0],[0.5,rho,0.0]],
            [[0,0,0],[1,0,0],[0.5,rho*np.cos(phi),rho*np.sin(phi)]],
        )
        np.testing.assert_almost_equal(out, phi)

def test_xyhat_from_pt_simple():
    P = np.array([0.5,0.5,0.0])
    T = np.array([[0,0,0],[1,0,0],[0,1,0]])
    xyhat = edge_adj_setup.xyhat_from_pt(P.tolist(), T.tolist())
    np.testing.assert_almost_equal(xyhat, [0.5, 0.5])

def test_xyhat_from_pt_harder():
    P = np.array([0,2.0,0.0])
    T = np.array([[0,2,0],[0,3,0],[0,2,1]])
    xyhat = edge_adj_setup.xyhat_from_pt(P.tolist(), T.tolist())
    np.testing.assert_almost_equal(xyhat, [0.0, 0.0])

    P = np.array([0,2.5,0.5])
    T = np.array([[0,2,0],[0,3,0],[0,2,1]])
    xyhat = edge_adj_setup.xyhat_from_pt(P.tolist(), T.tolist())
    np.testing.assert_almost_equal(xyhat, [0.5, 0.5])

    P = np.array([0,3.0,0.5])
    T = np.array([[0,2,0],[0,4,0],[0,2,1]])
    xyhat = edge_adj_setup.xyhat_from_pt(P.tolist(), T.tolist())
    np.testing.assert_almost_equal(xyhat, [0.5, 0.5])

def test_xyhat_from_pt_random():
    for i in range(20):
        xhat = np.random.rand(1)[0]
        yhat = np.random.rand(1)[0] * (1 - xhat)
        T = np.random.rand(3,3)
        P = geometry.tri_pt(geometry.linear_basis_tri(xhat, yhat), T)
        xhat2, yhat2 = edge_adj_setup.xyhat_from_pt(P.tolist(), T.tolist())
        np.testing.assert_almost_equal(xhat, xhat2)
        np.testing.assert_almost_equal(yhat, yhat2)

def test_get_split_pt():
    tri = [[0,0,0],[1,0,0],[0.4,0.8,0.0]]
    split_pt = edge_adj_setup.get_split_pt(tri)
    np.testing.assert_almost_equal(split_pt, [0.5, min_angle_isoceles_height, 0.0])

def test_get_split_pt_rotated():
    for i in range(50):
        R = geometry.random_rotation()
        scale = np.random.rand(1) * 10.0
        tri = np.array([[0,0,0],[1,0,0],[np.random.rand(1)[0] * 0.5,np.random.rand(1)[0],0.0]])
        rot_tri = scale * R.dot(tri.T).T
        split_pt = edge_adj_setup.get_split_pt(rot_tri.tolist())
        rot_correct = scale * R.dot([0.5, min_angle_isoceles_height, 0.0])
        np.testing.assert_almost_equal(split_pt, rot_correct)

def test_separate():
    i = 5
    while i > 0:
        pts = np.random.rand(4,3)

        obs_tri = pts[:3,:]
        src_tri = pts[[1,0,3],:]

        # ensure the random triangles are legal triangles
        try:
            standardize.standardize(obs_tri.tolist(), 20, True)
            standardize.standardize(src_tri.tolist(), 20, True)
        except standardize.BadTriangleException:
            continue

        pts, obs_set, src_set, obs_basis_tris, src_basis_tris = edge_adj_setup.separate_tris(
            obs_tri.tolist(), src_tri.tolist()
        )
        obs0_angles = geometry.triangle_internal_angles(np.array(pts)[obs_set[0]].tolist())
        src0_angles = geometry.triangle_internal_angles(np.array(pts)[src_set[0]].tolist())

        np.testing.assert_almost_equal(obs0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(obs0_angles[1], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[1], np.deg2rad(table_min_internal_angle))
        i -= 1

def test_orient_simple():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, -0.5, 0.0]])
    tris = np.array([[0,1,2], [1,0,3]])
    result = edge_adj_setup.orient_adj_tris(pts, tris, 0, 1)
    assert(result[0] == 0)
    np.testing.assert_almost_equal(result[1], pts[tris[0]])
    assert(result[2] == 0)
    assert(not result[3])
    np.testing.assert_almost_equal(result[4], pts[tris[1]])

def test_orient_rot():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, -0.5, 0.0]])
    tris = np.array([[1,2,0], [1,0,3]])
    result = edge_adj_setup.orient_adj_tris(pts, tris, 0, 1)
    assert(result[0] == 2)
    np.testing.assert_almost_equal(result[1], pts[[0,1,2]])
    assert(result[2] == 0)
    assert(not result[3])
    np.testing.assert_almost_equal(result[4], pts[tris[1]])

def test_orient_flip():
    pts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.5, 0], [0.5, -0.5, 0.0]])
    tris = np.array([[0,1,2], [0,1,3]])
    result = edge_adj_setup.orient_adj_tris(pts, tris, 0, 1)
    assert(result[0] == 0);
    np.testing.assert_almost_equal(result[1], pts[tris[0]])
    assert(result[2] == 0);
    assert(result[3])
    np.testing.assert_almost_equal(result[4], pts[[1,0,3]])
