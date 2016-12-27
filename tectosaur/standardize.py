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

# import cppimport
# fast_lookup = cppimport.imp("tectosaur.fast_lookup").fast_lookup
#
# def tolist_args(f):
#     def wrapper(*args):
#         new_args = []
#         for a in args:
#             new_args.append(a.tolist() if type(a) == np.ndarray else a)
#         return f(*new_args)
#     return wrapper
#
# get_edge_lens = tolist_args(fast_lookup.get_edge_lens)
# get_longest_edge = tolist_args(fast_lookup.get_longest_edge)
# get_origin_vertex = tolist_args(fast_lookup.get_origin_vertex)
# relabel = tolist_args(fast_lookup.relabel)
# check_bad_tri = tolist_args(fast_lookup.check_bad_tri)
# transform_from_standard = tolist_args(fast_lookup.transform_from_standard)
# rotation_matrix = tolist_args(fast_lookup.rotation_matrix)
# translate = tolist_args(fast_lookup.translate)
# rotate1_to_xaxis = tolist_args(fast_lookup.rotate1_to_xaxis)
# rotate2_to_xyplane = tolist_args(fast_lookup.rotate2_to_xyplane)
# full_standardize_rotate = tolist_args(fast_lookup.full_standardize_rotate)
# scale = tolist_args(fast_lookup.scale)
# standardize = tolist_args(fast_lookup.standardize);

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
    translation = -tri[0,:]
    return tri + translation, translation

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
    eps = 1e-10

    a = tri[2][0]
    b = tri[2][1]

    # filter out when L2 > 1
    L2 = np.sqrt((a-1)**2 + b**2)
    if L2 > 1 + eps:
        return 1

    # filter out when L3 > 1
    L3 = np.sqrt(a**2 + b**2)
    if L3 >= 1 + eps:
        return 2

    # filter out when T1 < 20
    A1 = lawcos(1.0, L3, L2)
    if np.rad2deg(A1) < angle_lim - eps:
        return 3

    # filter out when A2 < 20
    A2 = lawcos(1.0, L2, L3)
    if np.rad2deg(A2) < angle_lim - eps:
        return 4

    # filter out when A3 < 20
    A3 = lawcos(L2, L3, 1.0)
    if np.rad2deg(A3) < angle_lim - eps:
        return 5
    return 0

def relabel_longest_edge01_and_shortestedge02(tri):
    ls = get_edge_lens(tri)
    longest = get_longest_edge(ls)
    ov = get_origin_vertex(ls)
    return relabel(tri, ov, longest)

def standardize(tri, angle_lim, should_relabel = True):
    if should_relabel:
        relabeled, labels = relabel_longest_edge01_and_shortestedge02(tri)
    else:
        relabeled = tri
        labels = [0,1,2]
    trans, translation = translate(relabeled)
    rot1, rot_mat1 = rotate1_to_xaxis(trans)
    rot2, rot_mat2 = rotate2_to_xyplane(rot1)
    np.testing.assert_almost_equal(0, rot2[2][2])
    sc, factor = scale(rot2)
    if should_relabel:
        code = check_bad_tri(sc, angle_lim)
        if code > 0:
            print("Bad tri: " + str(code))
            return None
    return sc, labels, translation, rot_mat2.dot(rot_mat1), factor

kernel_properties = dict()
kernel_properties['U'] = (1, -3)
kernel_properties['T'] = (0, -2)
kernel_properties['A'] = (0, -2)
kernel_properties['H'] = (-1, -1)

def transform_from_standard(I, K, sm, labels, translation, R, scale):
    is_flipped = not (labels[1] == ((labels[0] + 1) % 3))
    sm_power = kernel_properties[K][0]
    scale_power = kernel_properties[K][1]

    out = np.empty((3,3,3,3))
    for sb1 in range(3):
        for sb2 in range(3):
            cb1 = labels[sb1]
            cb2 = labels[sb2]
            I_rot = R.T.dot(I[sb1,:,sb2,:]).dot(R)
            I_scale = I_rot * (scale ** scale_power)
            I_sm = I_scale * (sm ** sm_power)

            # Why does flipping the (1,0),(2,0),(0,1),(0,2) work?
            if is_flipped:
                d1 = np.tile(np.arange(3), (3, 1)).T
                d2 = d1.T
                I_final = I_sm * np.where(np.logical_xor(d1 == 0, d2 == 0), -1, 1)
            else:
                I_final = I_sm
            out[cb1,:,cb2,:] = I_final
    return out
