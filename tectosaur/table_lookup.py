import time
import numpy as np
from tectosaur.interpolate import barycentric_evalnd, cheb, cheb_wts, from_interval
from tectosaur.standardize import standardize, transform_from_standard
from tectosaur.geometry import tri_normal, xyhat_from_pt, projection, vec_angle
from tectosaur.adjacency import rotate_tri
import tectosaur.limit as limit
# import tectosaur.nearfield_op as nearfield_op

import cppimport
fast_lookup = cppimport.imp("tectosaur.fast_lookup").fast_lookup


table_min_internal_angle = 20 + 1e-11
min_angle_isoceles_height = 0.5 * np.tan(np.deg2rad(table_min_internal_angle))

minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01

def adjacent_interp_pts_wts(n_phi, n_pr):
    phihats = cheb(-1, 1, n_phi)
    prhats = cheb(-1, 1, n_pr)
    Ph,Nh = np.meshgrid(phihats,prhats)
    interp_pts = np.array([Ph.ravel(), Nh.ravel()]).T

    phiwts = cheb_wts(-1, 1, n_phi)
    prwts = cheb_wts(-1, 1, n_pr)
    interp_wts = np.outer(prwts, phiwts).ravel()
    return interp_pts.copy(), interp_wts.copy()

def coincident_interp_pts_wts(n_A, n_B, n_pr):
    Ahats = cheb(-1, 1, n_A)
    Bhats = cheb(-1, 1, n_B)
    prhats = cheb(-1, 1, n_pr)
    Ah,Bh,Nh = np.meshgrid(Ahats, Bhats, prhats)
    interp_pts = np.array([Ah.ravel(),Bh.ravel(), Nh.ravel()]).T

    Awts = cheb_wts(-1,1,n_A)
    Bwts = cheb_wts(-1,1,n_B)
    prwts = cheb_wts(-1,1,n_pr)
    # meshgrid behaves in a slightly strange manner such that Bwts must go first
    # in this outer product!
    interp_wts = np.outer(np.outer(Bwts, Awts),prwts).ravel()
    return interp_pts.copy(), interp_wts.copy()

def interp_limit(eps_start, n_steps, remove_sing, interp_pts, interp_wts, table, pt):
    out = np.empty(81)
    log_coeffs = np.empty(81)
    epsvs = eps_start * (2.0 ** -np.arange(n_steps))
    total = 0
    for j in range(81):
        sequence_vals = []
        for eps_idx in range(4):
            start = time.time()
            sequence_vals.append(fast_lookup.barycentric_evalnd(
                interp_pts, interp_wts, table[:, eps_idx, j], pt
            ))
            total += time.time() - start
        if remove_sing:
            out[j], log_coeffs[j] = limit.limit(epsvs, sequence_vals, remove_sing)
        else:
            out[j] = limit.limit(epsvs, sequence_vals, remove_sing)
    print(total)
    if remove_sing:
        return out, log_coeffs
    else:
        return out

def coincident_lookup(table_and_pts_wts, K, sm, pr, tri, eps_start, n_steps, remove_sing):
    table_limits, table_log_coeffs, interp_pts, interp_wts = table_and_pts_wts

    start = time.time()
    standard_tri, labels, translation, R, scale = standardize(
        tri, table_min_internal_angle, True
    )
    print("i2: " + str(time.time() - start)); start = time.time()

    A, B = standard_tri[2][0:2]

    Ahat = from_interval(minlegalA, 0.5, A)
    Bhat = from_interval(minlegalB, maxlegalB, B)
    prhat = from_interval(0.0, 0.5, pr)
    pt = np.array([[Ahat, Bhat, prhat]])
    print("i3: " + str(time.time() - start)); start = time.time()

    interp_vals = fast_lookup.barycentric_evalnd(
        interp_pts, interp_wts, table_limits, pt
    )
    log_coeffs = fast_lookup.barycentric_evalnd(
        interp_pts, interp_wts, table_log_coeffs, pt
    )
    print("i4: " + str(time.time() - start)); start = time.time()

    standard_scale = np.sqrt(np.linalg.norm(tri_normal(standard_tri)))
    interp_vals += np.log(standard_scale) * log_coeffs
    print("i5: " + str(time.time() - start)); start = time.time()

    # interp_vals = interp_vals.reshape((3,3,3,3))
    print("i6: " + str(time.time() - start)); start = time.time()

    out = transform_from_standard(interp_vals, K, sm, labels, translation, R, scale)
    out = np.array(out).reshape((3,3,3,3))
    print("i7: " + str(time.time() - start)); start = time.time()
    return out

def coincident_table(kernel, sm, pr, pts, tris, remove_sing):
    # filename = '_50_0.080000_4_0.010000_3_3_3_coincidenttable.npy'
    filename = '_50_0.001000_4_0.000100_4_4_4_coincidenttable.npy'

    params = filename.split('_')

    rho_order = int(params[1])
    eps_start = float(params[2])
    n_steps = int(params[3])
    tol = float(params[4])
    n_A = int(params[5])
    n_B = int(params[6])
    n_pr = int(params[7])

    interp_pts, interp_wts = coincident_interp_pts_wts(n_A, n_B, n_pr)

    tri_pts = pts[tris]

    start = time.time()
    table_sequences = np.load('data/' + kernel + filename)
    table_limits = np.empty((table_sequences.shape[0], table_sequences.shape[2]))
    table_log_coeffs = np.empty((table_sequences.shape[0], table_sequences.shape[2]))
    epsvs = eps_start * (2.0 ** -np.arange(n_steps))
    for i in range(table_sequences.shape[0]):
        for j in range(table_sequences.shape[2]):
            table_limits[i,j], table_log_coeffs[i,j] = limit.limit(
                epsvs, table_sequences[i,:,j], remove_sing
            )
    print("o1: " + str(time.time() - start)); start = time.time()

    out = np.empty((tris.shape[0], 3, 3, 3, 3))

    for i in range(tris.shape[0]):
        out[i,:,:,:,:] = coincident_lookup(
            (table_limits, table_log_coeffs, interp_pts, interp_wts),
            kernel, sm, pr, tri_pts[i,:,:],
            eps_start, n_steps, remove_sing
        )
    print("o2: " + str(time.time() - start)); start = time.time()

    return out

def get_adjacent_theta(obs_tri, src_tri):
    p = obs_tri[1] - obs_tri[0]
    L1 = obs_tri[2] - obs_tri[0]
    L2 = src_tri[2] - src_tri[0]
    T1 = L1 - projection(L1, p)
    T2 = L2 - projection(L2, p)

    n1 = tri_normal(obs_tri, normalize = True)
    samedir = n1.dot(T2 - T1) > 0
    theta = vec_angle(T1, T2)

    if samedir:
        return theta
    else:
        return 2 * np.pi - theta

def triangle_internal_angles(tri):
    v01 = tri[1] - tri[0]
    v02 = tri[2] - tri[0]
    v12 = tri[2] - tri[1]

    L01 = np.linalg.norm(v01)
    L02 = np.linalg.norm(v02)
    L12 = np.linalg.norm(v12)

    A1 = np.arccos(v01.dot(v02) / (L01 * L02))
    A2 = np.arccos(-v01.dot(v12) / (L01 * L12))
    A3 = np.pi - A1 - A2

    return A1, A2, A3

def get_split_pt(tri):
    base_vec = tri[1] - tri[0]
    midpt = base_vec / 2.0 + tri[0]
    to2 = tri[2] - midpt
    V = to2 - projection(to2, base_vec)
    V = (V / np.linalg.norm(V)) * np.linalg.norm(base_vec) * min_angle_isoceles_height
    return midpt + V

def separate_tris(obs_tri, src_tri):
    np.testing.assert_almost_equal(obs_tri[0], src_tri[1])
    np.testing.assert_almost_equal(obs_tri[1], src_tri[0])

    obs_split_pt = get_split_pt(obs_tri)
    obs_split_pt_xyhat = xyhat_from_pt(obs_split_pt, obs_tri)

    src_split_pt = get_split_pt(src_tri)
    src_split_pt_xyhat = xyhat_from_pt(src_split_pt, src_tri)

    pts = np.array(
        [obs_tri[0], obs_tri[1], obs_tri[2], src_tri[2], obs_split_pt, src_split_pt]
    )
    obs_tris = np.array([[0, 1, 4], [4, 1, 2], [0, 4, 2]])
    src_tris = np.array([[1, 0, 5], [5, 0, 3], [1, 5, 3]])
    obs_basis_tris = np.array([
        [[0,0],[1,0],obs_split_pt_xyhat],
        [obs_split_pt_xyhat, [1,0],[0,1]],
        [[0,0],obs_split_pt_xyhat,[0,1]]
    ])
    src_basis_tris = np.array([
        [[0,0],[1,0],src_split_pt_xyhat],
        [src_split_pt_xyhat, [1,0],[0,1]],
        [[0,0],src_split_pt_xyhat,[0,1]]
    ])
    return pts, obs_tris, src_tris, obs_basis_tris, src_basis_tris

def adjacent_lookup(table_and_pts_wts, sm, pr, orig_obs_tri, obs_tri, src_tri,
        eps_start, n_steps, remove_sing):
    table, interp_pts, interp_wts = table_and_pts_wts

    standard_tri, _, translation, R, scale = standardize(
        orig_obs_tri, table_min_internal_angle, should_relabel = False
    )

    theta = get_adjacent_theta(obs_tri, src_tri)

    # factor = 1
    # #TODO: HANDLE FLIPPING!
    # if theta > np.pi:
    #     theta = 2 * np.pi - theta
    #     # factor *= -1

    thetahat = from_interval(0, np.pi, theta)
    prhat = from_interval(0, 0.5, pr)
    pt = np.array([[thetahat, prhat]])


    if remove_sing:
        interp_vals, log_coeffs = interp_limit(
            eps_start, n_steps, remove_sing, interp_pts, interp_wts, table, pt
        )

        standard_scale = np.sqrt(np.linalg.norm(tri_normal(standard_tri)))
        interp_vals += np.log(standard_scale) * log_coeffs
    else:
        interp_vals = interp_limit(
            eps_start, n_steps, remove_sing, interp_pts, interp_wts, table, pt
        )

    interp_vals = interp_vals.reshape((3,3,3,3))

    out = np.empty((3,3,3,3))
    for b1 in range(3):
        for b2 in range(3):
            correct = R.T.dot(interp_vals[b1,:,b2,:]).dot(R) / (sm * scale ** 1)
            out[b1,:,b2,:] = correct
    return out

def sub_basis(I, obs_basis_tri, src_basis_tri):
    out = np.zeros((3,3,3,3))
    bfncs = [
        lambda x,y: 1 - x - y,
        lambda x,y: x,
        lambda x,y: y
    ]

    for ob1 in range(3):
        for sb1 in range(3):
            for ob2 in range(3):
                for sb2 in range(3):
                    obv = bfncs[ob1](*obs_basis_tri[ob2,:])
                    sbv = bfncs[sb1](*src_basis_tri[sb2,:])
                    out[ob1,:,sb1,:] += I[ob2,:,sb2,:] * obv * sbv
    return out

def find_va_rotations(ot, st):
    ot_clicks = 0
    st_clicks = 0
    for d in range(3):
        matching_vert = np.where(st == ot[d])[0]
        if matching_vert.shape[0] > 0:
            ot_clicks = d
            st_clicks = matching_vert[0]
            break
    # If the loop finds no shared vertices, then the triangles are not touching
    # and no rotations will be performed.
    ot_rot = rotate_tri(ot_clicks)
    st_rot = rotate_tri(st_clicks)
    return ot_rot, st_rot

def adjacent_table(nq_va, kernel, sm, pr, pts, obs_tris, src_tris, remove_sing):
    # filename = '_50_0.080000_4_0.010000_3_3_adjacenttable.npy'
    # filename = '_50_0.080000_4_0.001000_4_4_adjacenttable.npy'
    filename = '_50_0.001000_4_0.000100_4_4_adjacenttable.npy'
    # filename = '_50_0.010000_4_0.010000_4_4_adjacenttable.npy'
    # filename = '_50_0.004000_4_0.001000_4_4_adjacenttable.npy'

    params = filename.split('_')
    rho_order = int(params[1])
    eps_start = float(params[2])
    n_steps = int(params[3])
    tol = float(params[4])
    n_theta = int(params[5])
    n_pr = int(params[6])

    interp_pts, interp_wts = adjacent_interp_pts_wts(n_theta, n_pr)

    table = np.load('data/' + kernel + filename)

    out = np.empty((obs_tris.shape[0], 3, 3, 3, 3))

    va_pts = []
    va_which_pair = []
    va_obs_tris = []
    va_src_tris = []
    va_obs_basis = []
    va_src_basis = []
    for i in range(obs_tris.shape[0]):
        obs_tri = pts[obs_tris[i]]
        src_tri = pts[src_tris[i]]
        split_pts, split_obs, split_src, obs_basis_tris, src_basis_tris = separate_tris(
            obs_tri, src_tri
        )
        lookup_result = adjacent_lookup(
            (table, interp_pts, interp_wts), sm, pr,
            obs_tri, split_pts[split_obs[0]], split_pts[split_src[0]],
            eps_start, n_steps, remove_sing
        )
        out[i] = sub_basis(lookup_result, obs_basis_tris[0], src_basis_tris[0])

        va_pts.extend(split_pts.tolist())
        for j in range(3):
            for k in range(3):
                if k == 0 and j == 0:
                    continue
                otA = np.linalg.norm(tri_normal(split_pts[split_obs[j]]))
                stA = np.linalg.norm(tri_normal(split_pts[split_src[k]]))
                if otA * stA < 1e-10:
                    continue

                ot = split_obs[j]
                st = split_src[k]
                ot_rot, st_rot = find_va_rotations(ot, st)

                va_obs_tris.append((ot[ot_rot] + 6 * i).tolist())
                va_src_tris.append((st[st_rot] + 6 * i).tolist())
                va_which_pair.append(i)
                va_obs_basis.append(obs_basis_tris[j][ot_rot])
                va_src_basis.append(src_basis_tris[k][st_rot])

    Iv = nearfield_op.vert_adj(
        nq_va, kernel, sm, pr,
        np.array(va_pts), np.array(va_obs_tris), np.array(va_src_tris)
    )
    for i in range(Iv.shape[0]):
        Iv[i] = sub_basis(Iv[i], va_obs_basis[i], va_src_basis[i])
        out[va_which_pair[i]] += Iv[i]

    return out
