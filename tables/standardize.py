import numpy as np

# THE PLAN!
# Find longest edge
# relabel to put that longest edge as the 0-1 edge
# flip to put the 2 vertex closer to the 0 vertex than the 1 vertex
# translate 0 vertex to origin
# rotate 1 vertex to be at (A, 0, 0) and store rotation
# rotate 2 vertex to be at (B, C, 0) and store rotation
# scale triangle so that 1 vertex is at (1, 0, 0) and store scale factor
# check that triangle internal angles are greater than 20 degrees

def get_edge_lens(tri):
    L0 = np.sum((tri[1,:] - tri[0,:])**2)
    L1 = np.sum((tri[2,:] - tri[1,:])**2)
    L2 = np.sum((tri[2,:] - tri[0,:])**2)
    return L0, L1, L2

def get_longest_edge(lens):
    if lens[0] >= lens[1] and lens[0] >= lens[2]:
        return 0
    elif lens[1] >= lens[0] and lens[1] >= lens[2]:
        return 1
    elif lens[2] >= lens[0] and lens[2] >= lens[1]:
        return 2

def get_origin_vertex(lens):
    longest = get_longest_edge(lens)
    if longest == 0 and lens[1] >= lens[2]:
        return 0
    if longest == 0 and lens[2] >= lens[1]:
        return 1
    if longest == 1 and lens[2] >= lens[0]:
        return 1
    if longest == 1 and lens[0] >= lens[2]:
        return 2
    if longest == 2 and lens[0] >= lens[1]:
        return 2
    if longest == 2 and lens[1] >= lens[0]:
        return 0

def relabel(tri, ov, longest_edge):
    if longest_edge == ov:
        labels = [ov, (ov + 1) % 3, (ov + 2) % 3]
    elif (longest_edge + 1) % 3 == ov:
        labels = [ov, (ov + 2) % 3, (ov + 1) % 3]
    else:
        raise Exception("BAD!")
    return np.array([tri[L] for L in labels]), labels

def translate(tri):
    return tri - tri[0,:]

def rotation_matrix(axis, theta):
    cross_mat = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    outer_mat = np.outer(axis, axis)
    id = np.identity(3)
    rot_mat = np.cos(theta) * id + np.sin(theta) * cross_mat + (1 - np.cos(theta)) * outer_mat
    return rot_mat

def rotate1_to_xaxis(tri):
    # rotate 180deg around the axis halfway in between the 0-1 vector and
    # the 0-(L,0,0) vector, where L is the length of the 0-1 vector
    to_pt1 = tri[1] - tri[0]
    pt1L = np.linalg.norm(to_pt1)
    to_target = np.array([pt1L, 0, 0])
    axis = (to_pt1 + to_target) / 2.0
    axis_mag = np.linalg.norm(axis)
    if axis_mag == 0.0:
        axis = np.array([0,0,1])
    else:
        axis /= axis_mag
    theta = np.pi
    rot_mat = rotation_matrix(axis, theta)
    return rot_mat.dot(tri.T).T, rot_mat

def rotate2_to_xyplane(tri):
    xaxis = np.array([1, 0, 0])
    yaxis = np.array([0, 1, 0])
    ydot2 = yaxis[1:].dot(tri[2][1:]) / np.linalg.norm(tri[2][1:])
    theta = np.arccos(ydot2)
    rot_mat = rotation_matrix(xaxis, theta)
    out_tri = rot_mat.dot(tri.T).T
    if np.abs(out_tri[2][2]) > 1e-10:
        theta = -np.arccos(ydot2)
        rot_mat = rotation_matrix(xaxis, theta)
        out_tri = rot_mat.dot(tri.T).T
    return out_tri, rot_mat

def scale(tri):
    return tri / tri[1][0], 1.0 / tri[1][0]

def lawcos(a, b, c):
    return np.arccos((a**2 + b**2 - c**2) / (2*a*b))

# Checks for acceptable edge lengths and internal angles
def check_bad_tri(tri, angle_lim):
    a = tri[2][0]
    b = tri[2][1]

    # filter out when L2 < 1
    L2 = np.sqrt((a-1)**2 + b**2)
    if L2 > 1:
        return 1

    # filter out when L3 < 1
    L3 = np.sqrt(a**2 + b**2)
    if L3 > 1:
        return 2

    # filter out when T1 < 20
    A1 = lawcos(1.0, L3, L2)
    if np.rad2deg(A1) < angle_lim:
        return 3

    # filter out when A2 < 20
    A2 = lawcos(1.0, L2, L3)
    if np.rad2deg(A2) < angle_lim:
        return 4

    # filter out when A3 < 20
    A3 = lawcos(L2, L3, 1.0)
    if np.rad2deg(A3) < angle_lim:
        return 5
    return 0

def standardize(tri):
    ls = get_edge_lens(tri)
    longest = get_longest_edge(ls)
    ov = get_origin_vertex(ls)
    relabeled, labels = relabel(tri, ov, longest)
    trans = translate(relabeled)
    rot1, rot_mat1 = rotate1_to_xaxis(trans)
    rot2, rot_mat2 = rotate2_to_xyplane(rot1)
    np.testing.assert_almost_equal(0, rot2[2][2])
    sc, factor = scale(rot2)
    if check_bad_tri(sc, 20) > 0:
        raise Exception("Bad tri!")
    return sc, labels, rot_mat2.dot(rot_mat1), factor

def test_origin_vertex():
    assert(get_origin_vertex(get_edge_lens(np.array([[0,0,0],[1,0,0],[0.2,0.5,0]]))) == 0)
    assert(get_origin_vertex(get_edge_lens(np.array([[0,0,0],[1,0,0],[0.8,0.5,0]]))) == 1)
    assert(get_origin_vertex(get_edge_lens(np.array([[1,0,0],[0,0,0],[0.2,0.5,0]]))) == 1)
    assert(get_origin_vertex(get_edge_lens(np.array([[1,0,0],[0,0,0],[0.8,0.5,0]]))) == 0)
    assert(get_origin_vertex(get_edge_lens(np.array([[0.8,0.5,0],[0,0,0],[1,0,0]]))) == 2)

def test_longest_edge():
    assert(get_longest_edge(get_edge_lens(np.array([[0,0,0],[1,0,0],[0.5,0.5,0]]))) == 0)
    assert(get_longest_edge(get_edge_lens(np.array([[0,0,0],[0.5,0.5,0],[1,0,0]]))) == 2)

def test_translate():
    out = translate(np.array([[0,1,0],[0,0,0],[0,2,0]]))
    np.testing.assert_almost_equal(out, [[0,0,0], [0,-1,0],[0,1,0]])

def test_relabel():
    out = relabel(np.array([[0,0,0],[0.2,0,0],[0.4,0.5,0]]), 0, 2)
    np.testing.assert_almost_equal(out, [[0,0,0],[0.4,0.5,0],[0.2,0,0]])

def test_rotate1():
    out = rotate1_to_xaxis(np.array([[0,0,0], [1.0,1,1], [0,1,1]], dtype = np.float64))
    np.testing.assert_almost_equal(out,
        [[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), -np.sqrt(1 / 3.0), -np.sqrt(1 / 3.0)]])

def test_rotate2():
    out1 = rotate1_to_xaxis(np.array([[0,0,0], [1.0,1,1], [0,1,1]], dtype = np.float64))
    out2 = rotate2_to_xyplane(out1)
    np.testing.assert_almost_equal(out2,
        [[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), np.sqrt(2.0 / 3.0), 0]])

def test_scale():
    out = scale([[0,0,0],[np.sqrt(3),0,0],[np.sqrt(1 + 1.0 / 3), np.sqrt(2.0 / 3.0), 0]])
    np.testing.assert_almost_equal(out, [[0,0,0],[1.0,0,0],[np.sqrt(4 / 9.0), np.sqrt(2 / 9.0),0]])

def test_standardize():
    # out = standardize(np.array([[0,0,0],[0.2,0,0],[0.4,0.5,0]]))
    # np.testing.assert_almost_equal(out, [[0,0,0],[0.4,0.5,0],[0.2,0,0]])

    out,_,_ = standardize(np.array([[0,0,0],[1,0.0,0],[0.0,0.5,0]]))
    np.testing.assert_almost_equal(out, [[0,0,0],[0.4,0.5,0],[0.2,0.1,0]])

def test_rotation():
    sm = 1.0

    def test_tri(tri):
        standard_tri, labels, R, factor = standardize.standardize(np.array(tri))
        np.testing.assert_almost_equal(
            standard_tri,
            [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
        )

        correct_full = calc(
            K, tri, rho_order, tol, eps_start, eps_step, n_steps, sm, pr
        ).reshape((3,3,3,3))

        standard_full = calc(
            K, standard_tri.tolist(), rho_order, tol, eps_start, eps_step, n_steps, 1.0, pr
        ).reshape((3,3,3,3))

        for sb1 in range(3):
            for sb2 in range(3):
                standardized = R.T.dot(standard_full[sb1,:,sb2,:]).dot(R) / (sm * factor ** 3)
                cb1 = labels[sb1]
                cb2 = labels[sb2]
                correct = correct_full[cb1,:,cb2,:]
                np.testing.assert_almost_equal(correct, standardized, 4)

        # np.testing.assert_almost_equal(correct, standardized, 4)

    test_tris = [
        [[0.0,0.0,0.0], [1.1,0.0,0.0], [0.4,0.3,0.0]]
        ,[[0.0,0.0,0.0], [0.0,1.1,0.0], [-0.3,0.4,0.0]]
        ,[[0.0,0.0,0.0], [0.0,0.0,1.1], [0.0,-0.3,0.4]]
        ,[[0.0,0.0,0.0], [0.0,0.3,1.1], [0.0,-0.3,0.4]]
        ,[[0.0,0.0,0.0], [0.0,-0.3,0.4], [0.0,0.35,1.1]]
        ,[[0.0,0.35,1.1], [0.0,0.0,0.0], [0.0,-0.3,0.4]]
        ,[[0.0, -0.3, 0.4], [0.0,0.35,1.1], [0.0,0.0,0.0]]
        ,[[1.0,0.0,0.0], [0.0,-0.3,0.45], [0.0,0.35,1.1]]
    ]
    for t in test_tris:
        test_tri(t):

    n_checks = 10
    while True:
        tri = np.random.rand(3,3).tolist()

        try:
            test_tri(tri)
            print("SUCCESS!")
        except Exception as e:
            # Exception implies the triangle was malformed (angle < 20 degrees)
            continue

        n_checks -= 1
        if n_checks == 0:
            break
