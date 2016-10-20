import time
import numpy as np
from tectosaur.interpolate import barycentric_evalnd, cheb, cheb_wts, from_interval
from tectosaur.standardize import standardize
from tectosaur.geometry import tri_normal, xyhat_from_pt, projection, vec_angle

table_min_internal_angle = 20 + 1e-11
min_angle_isoceles_height = 0.5 * np.tan(np.deg2rad(table_min_internal_angle))

minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01
filename, n_A, n_B, n_pr = ('coincidenttable_limits.npy', 8, 8, 8)
filename, n_A, n_B, n_pr = ('testtable.npy', 3, 3, 3)


def coincident_interp_pts_wts(n_A, n_B, n_pr):
    Ahats = cheb(-1, 1, n_A)
    Bhats = cheb(-1, 1, n_B)
    prhats = cheb(-1, 1, n_pr)
    Ah,Bh,Nh = np.meshgrid(Ahats, Bhats,prhats)
    interp_pts = np.array([Ah.ravel(),Bh.ravel(), Nh.ravel()]).T

    Awts = cheb_wts(-1,1,n_A)
    Bwts = cheb_wts(-1,1,n_B)
    prwts = cheb_wts(-1,1,n_pr)
    interp_wts = np.outer(Awts,np.outer(Bwts,prwts)).ravel()
    return interp_pts, interp_wts

def coincident_lookup(table_and_pts_wts, sm, pr, tri):
    table, interp_pts, interp_wts = table_and_pts_wts

    standard_tri, labels, translation, R, scale = standardize(
        tri, table_min_internal_angle
    )

    A, B = standard_tri[2][0:2]

    Ahat = from_interval(minlegalA, 0.5, A)
    Bhat = from_interval(minlegalB, maxlegalB, B)
    prhat = from_interval(0.0, 0.5, pr)

    interp_vals = np.empty(81)
    for j in range(81):
        interp_vals[j] = barycentric_evalnd(
            interp_pts, interp_wts, table[:, j], np.array([[Ahat, Bhat, prhat]])
        )[0]
    interp_vals = interp_vals.reshape((3,3,3,3))

    out = np.empty((3,3,3,3))
    for sb1 in range(3):
        for sb2 in range(3):
            cb1 = labels[sb1]
            cb2 = labels[sb2]
            correct = R.T.dot(interp_vals[sb1,:,sb2,:]).dot(R) / (sm * scale ** 1)
            out[cb1,:,cb2,:] = correct
    return out

def coincident_table(kernel, sm, pr, pts, tris):
    interp_pts, interp_wts = coincident_interp_pts_wts(n_A, n_B, n_pr)

    tri_pts = pts[tris]

    table = np.load('data/' + kernel + filename)
    out = np.empty((tris.shape[0], 3, 3, 3, 3))

    for i in range(tris.shape[0]):
        out[i,:,:,:,:] = coincident_lookup(
            (table, interp_pts, interp_wts), sm, pr, tri_pts[i,:,:]
        )

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

def adjacent_interp_pts_wts(n_theta, n_pr):
    thetahats = cheb(-1, 1, n_theta)
    prhats = cheb(-1, 1, n_pr)
    Th,Nh = np.meshgrid(thetahats,prhats)
    interp_pts = np.array([Th.ravel(), Nh.ravel()]).T

    thetawts = cheb_wts(-1, 1, n_theta)
    prwts = cheb_wts(-1, 1, n_pr)
    interp_wts = np.outer(thetawts, prwts).ravel()
    return interp_pts, interp_wts

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

def adjacent_lookup(table_and_pts_wts, sm, pr, obs_tri, src_tri):
    table, interp_pts, interp_wts = table_and_pts_wts

    standard_tri, _, translation, R, scale = standardize(
        obs_tri, table_min_internal_angle, should_relabel = False
    )

    theta = get_adjacent_theta(obs_tri, src_tri)

    #TODO: HANDLE FLIPPING!
    if theta > np.pi:
        theta = 2 * np.pi - theta

    thetahat = from_interval(0, np.pi, theta)
    prhat = from_interval(0, 0.5, pr)

    interp_vals = np.empty(81)
    for j in range(81):
        interp_vals[j] = barycentric_evalnd(
            interp_pts, interp_wts, table[:, j], np.array([[thetahat, prhat]])
        )[0]
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

def sub_basis_simple(I, obs_basis_tri, src_basis_tri):
    out = np.zeros((3,3,3,3))
    bfncs = [
        lambda x,y: 1 - x - y,
        lambda x,y: x,
        lambda x,y: y
    ]

    for ob1 in range(3):
        for ob2 in range(3):
            out[ob1] += I[ob2] * bfncs[ob1](*obs_basis_tri[ob2,:])
    return out

def adjacent_table(kernel, sm, pr, pts, obs_tris, src_tris):
    n_theta = n_pr = 3
    interp_pts, interp_wts = adjacent_interp_pts_wts(n_theta, n_pr)

    table = np.load(kernel + '_adj_' + filename)

    out = np.empty((obs_tris.shape[0], 3, 3, 3, 3))

    vert_adj_pairs = []
    for i in range(obs_tris.shape[0]):
        obs_tri = pts[obs_tris[i]]
        src_tri = pts[src_tris[i]]
        split_pts, split_obs, split_src, obs_basis_tris, src_basis_tris = separate_tris(
            obs_tri, src_tri
        )
        lookup_result = adjacent_lookup(
            (table, interp_pts, interp_wts), sm, pr,
            split_pts[split_obs[0]], split_pts[split_src[0]]
        )
        out[i] = sub_basis(lookup_result, obs_basis_tris[0], src_basis_tris[0])

        for j in range(3):
            for k in range(3):
                if k == 0 and j == 0:
                    continue
                vert_adj_pairs.append((
                    i, split_pts, split_obs[j], split_src[k],
                    obs_basis_tris[j], src_basis_tris[k]
                ))

    return out, vert_adj_pairs
        #Perform vertex adjacent for all other pairs
