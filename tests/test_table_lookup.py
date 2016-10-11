from tectosaur.table_lookup import *
from tectosaur.standardize import standardize, rotation_matrix

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
        if standardize(obs_tri) is None:
            continue
        if standardize(src_tri) is None:
            continue

        obs_set, src_set = separate_tris(obs_tri, src_tri)
        obs0_angles = triangle_internal_angles(obs_set[0])
        src0_angles = triangle_internal_angles(src_set[0])

        np.testing.assert_almost_equal(obs0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(obs0_angles[1], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[0], np.deg2rad(table_min_internal_angle))
        np.testing.assert_almost_equal(src0_angles[1], np.deg2rad(table_min_internal_angle))
        i -= 1

# def test_coincident_lookup():
    # A = 0.0
    # B = 1.0
    # pr = 0.3
    #
    # pts = np.array([[0,0,0],[1,0,0],[A,B,0]], dtype = np.float64)
    # tris = np.array([[0,1,2]])
    # eps = 0.1 * (2.0 ** -np.arange(1))
    # op = DenseIntegralOp(eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
    # # op3 = DenseIntegralOp(eps, 16, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
    # op2 = DenseIntegralOp(eps, 15, 15, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris, use_tables = True)
    #
    # # rho_order = 50
    # # rho_gauss = quad.gaussxw(rho_order)
    # # rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    # # res = adaptive_integrate.integrate_coincident(
    # #     'H', pts[tris][0].tolist(), 0.04, 0.1, 1.0, pr,
    # #     rho_q[0].tolist(), rho_q[1].tolist()
    # # )
    #
    # print(op.mat[0,0])
    # print(op2.mat[0,0])
    # print(res[0])

def test_adjacent_table_lookup():
    rho = 0.5 * np.tan(np.deg2rad(20))
    theta = 2.93114584997056#0.9
    pr = 0.4665063509461097#0.25

    factor = 2.0
    translation =

    pts = factor * np.array([[0,0,0],[1,0,0],[0.5,rho,0.0],[0.5,rho*np.cos(theta),rho*np.sin(theta)]])
    tris = np.array([[0,1,2],[1,0,3]])

    from tectosaur.table_lookup import adjacent_table

    result, pairs = adjacent_table('H', 1.0, pr, pts, np.array([tris[0]]), np.array([tris[1]]))
    print(result[0,0,0,0,0])

    eps = 0.08 * (2.0 ** -np.arange(4))
    op = DenseIntegralOp(eps, 15, 16, 10, 3, 10, 3.0, 'H', 1.0, pr, pts, tris)
    print(op.mat[0,9])
