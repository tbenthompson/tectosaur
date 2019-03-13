import numpy as np
import tectosaur as tct
from tectosaur.stress_constraints import *

def test_rot_mat():
    tri = np.random.rand(3,3)
    x_to_xp = rot_mat(tri)
    np.testing.assert_almost_equal(x_to_xp.dot(x_to_xp.T), np.eye(3))
    v1 = tri[1] - tri[0]
    v1 /= np.linalg.norm(v1)
    np.testing.assert_almost_equal(x_to_xp.T.dot([1,0,0]), v1)

def test_jacobian():
    for i in range(10):
        tri = np.random.rand(3,3)
        x_to_xhat = jacobian(tri)
        xhat_to_x = inv_jacobian(tri)
        np.testing.assert_almost_equal(x_to_xhat.dot([1,0]), tri[1] - tri[0])
        np.testing.assert_almost_equal(x_to_xhat.dot([0,1]), tri[2] - tri[0])
        np.testing.assert_almost_equal(xhat_to_x.dot(tri[1] - tri[0]), [1,0])
        np.testing.assert_almost_equal(xhat_to_x.dot(tri[2] - tri[0]), [0,1])

def test_calc_gradient():
    tri = np.array([
        [0,0,0],
        [1.1,0,1],
        [1,0,0]
    ]).astype(np.float)
    disp = np.array([
        [0,0,0],
        [1.1,0,0],
        [1,0,0]
    ]).astype(np.float)
    disp_xp_dxp, x_to_xp = calc_gradient(tri, disp)

    np.testing.assert_almost_equal(
        x_to_xp.T.dot(disp_xp_dxp).dot(x_to_xp),
        [[1,0,0],[0,0,0],[0,0,0]]
    )

def test_derive_stress():
    tri = np.array([
        [0,0,0],
        [1.1,0,1],
        [1,0,0]
    ]).astype(np.float)
    disp = np.array([
        [0,0,0],
        [1.1,0,0],
        [1.0,0,0]
    ]).astype(np.float)
    strain, stress, x_to_xp = derive_stress((tri,disp,0,0), 1.0, 0.25)

    strain_x = x_to_xp.T.dot(strain[:,:,0].dot(x_to_xp))
    correct = np.zeros((3,3))
    correct[0,0] = 1.0
    correct[1,1] = -1/3.
    np.testing.assert_almost_equal(strain_x, correct)

def test_stress_constraints():
    tri1 = np.array([
        [0,0,0],
        [1.1,0,1],
        [1,0,0]
    ]).astype(np.float)
    disp1 = np.array([
        [0,0,0],
        [1.1,0,0],
        [1.0,0,0]
    ]).astype(np.float)
    tri2 = np.array([
        [0,0,0],
        [0,1.1,1],
        [0,1,0]
    ]).astype(np.float)
    disp2 = np.array([
        [0,0,0],
        [-1.1,0,0],
        [-1,0,0]
    ]).astype(np.float)
    stress_constraints((tri1,disp1,0,0),(tri2,disp2,1,0), 1.0, 0.25)

def test_stress_funky():
    from numpy import array, float32
    tri_data1 = (array([[0.03333333, 0.01666667, 0.33333333],
        [0.        , 0.        , 0.33333333],
        [0.        , 0.        , 0.36666667]]), array([[0.2454905, 0.       , 0.       ],
        [0.25     , 0.       , 0.       ],
        [0.1654347, 0.       , 0.       ]], dtype=float32), 3519, 1)

    tri_data2 = (array([[ 0.        ,  0.        ,  0.36666667],
        [ 0.        ,  0.        ,  0.33333333],
        [-0.03333333,  0.01666667,  0.36666667]]), array([[0.1654347 , 0.        , 0.        ],
        [0.25      , 0.        , 0.        ],
        [0.16191977, 0.        , 0.        ]], dtype=float32), 3638, 1)

    E1, S1, M1 = derive_stress(tri_data1, 1.0, 0.25)
    E1, S2, M2 = derive_stress(tri_data2, 1.0, 0.25)
    S1x = rotate_tensor(S1, M1)
    S2x = rotate_tensor(S2, M2)

    stress_constraints(tri_data1, tri_data2, 1.0, 0.25)
