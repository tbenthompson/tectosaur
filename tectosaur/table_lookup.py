import time
import numpy as np
from tectosaur.interpolate import barycentric_evalnd, cheb, cheb_wts, from_interval
from tectosaur.standardize import standardize, transform_from_standard
from tectosaur.geometry import tri_normal, xyhat_from_pt, projection, vec_angle
from tectosaur.adjacency import rotate_tri
import tectosaur.limit as limit
import tectosaur.nearfield_op as nearfield_op
from tectosaur.util.timer import Timer
import tectosaur.util.gpu as gpu

from tectosaur.table_params import *

import cppimport
fast_lookup = cppimport.imp("tectosaur.fast_lookup").fast_lookup



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

def coincident_lookup_interpolation_gpu(table_limits, table_log_coeffs,
        interp_pts, interp_wts, pts):

    float_type = np.float64
    gpu_cfg = {'float_type': gpu.np_to_c_type(float_type)}
    module = gpu.load_gpu('table_lookup.cl', tmpl_args = gpu_cfg)
    fnc = module.coincident_lookup_interpolation

    n_tris = pts.shape[0]

    gpu_table_limits = gpu.to_gpu(table_limits, float_type)
    gpu_table_log_coeffs = gpu.to_gpu(table_log_coeffs, float_type)
    gpu_interp_pts = gpu.to_gpu(interp_pts, float_type)
    gpu_interp_wts = gpu.to_gpu(interp_wts, float_type)
    gpu_pts = gpu.to_gpu(pts, float_type)

    gpu_result = gpu.empty_gpu(n_tris * 81 * 2, float_type)

    fnc(
        gpu.gpu_queue, (n_tris,), None,
        gpu_result.data,
        np.int32(gpu_interp_pts.shape[0]),
        gpu_table_limits.data,
        gpu_table_log_coeffs.data,
        gpu_interp_pts.data,
        gpu_interp_wts.data,
        gpu_pts.data
    )

    out = gpu_result.get().reshape((n_tris, 81, 2))
    return out[:, :, 0], out[:, :, 1]

def coincident_table(kernel, sm, pr, pts, tris, remove_sing):
    filename = 'data/H_100_0.003125_6_0.000001_12_17_9_coincidenttable.npy'

    params = filename.split('_')

    n_A = int(params[5])
    n_B = int(params[6])
    n_pr = int(params[7])

    interp_pts, interp_wts = coincident_interp_pts_wts(n_A, n_B, n_pr)

    tri_pts = pts[tris]

    start = time.time()
    table_data = np.load(filename)
    table_limits = table_data[:,:,0]
    table_log_coeffs = table_data[:,:,1]
    print("o1: " + str(time.time() - start)); start = time.time()

    # Shift to a three step process
    # 1) Get interpolation points
    pts, standard_tris = fast_lookup.coincident_lookup_pts(tri_pts, pr);

    # 2) Perform interpolation --> GPU!
    # interp_vals, log_coeffs = fast_lookup.coincident_lookup_interpolation(
    #     table_limits, table_log_coeffs, interp_pts, interp_wts, pts
    # )
    interp_vals, log_coeffs = coincident_lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, pts
    )

    # 3) Transform to real space
    out = fast_lookup.coincident_lookup_from_standard(
        standard_tris, interp_vals, log_coeffs, kernel, sm
    ).reshape((-1, 3, 3, 3, 3))

    print("o2: " + str(time.time() - start)); start = time.time()

    return out

def get_adjacent_phi(obs_tri, src_tri):
    p = obs_tri[1] - obs_tri[0]
    L1 = obs_tri[2] - obs_tri[0]
    L2 = src_tri[2] - src_tri[0]
    T1 = L1 - projection(L1, p)
    T2 = L2 - projection(L2, p)

    n1 = tri_normal(obs_tri, normalize = True)
    samedir = n1.dot(T2 - T1) > 0
    phi = vec_angle(T1, T2)

    if samedir:
        return phi
    else:
        return 2 * np.pi - phi

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

def separate_tris(obs_tri, src_tri):
    # np.testing.assert_almost_equal(obs_tri[0], src_tri[1])
    # np.testing.assert_almost_equal(obs_tri[1], src_tri[0])

    obs_split_pt = fast_lookup.get_split_pt(obs_tri.tolist())
    obs_split_pt_xyhat = xyhat_from_pt(obs_split_pt, obs_tri)

    src_split_pt = fast_lookup.get_split_pt(src_tri.tolist())
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

def adjacent_lookup(table_and_pts_wts, K, sm, pr, orig_obs_tri, obs_tri, src_tri,
        remove_sing):
    t = Timer(silent = True)

    table_limits, table_log_coeffs, interp_pts, interp_wts = table_and_pts_wts

    code, standard_tri, labels, translation, R, scale = standardize(
        orig_obs_tri, table_min_internal_angle, False
    )
    t.report("standardize")

    phi = get_adjacent_phi(obs_tri, src_tri)

    # factor = 1
    # #TODO: HANDLE FLIPPING! Probably depends on kernel.
    if phi > np.pi:
        phi = 2 * np.pi - phi
    #     # factor *= -1

    assert(min_intersect_angle <= phi and phi <= np.pi)
    phihat = from_interval(min_intersect_angle, np.pi, phi)
    prhat = from_interval(0, 0.5, pr)
    pt = np.array([phihat, prhat])
    t.report("get interp pt")

    interp_vals = fast_lookup.barycentric_evalnd(interp_pts, interp_wts, table_limits, pt)
    log_coeffs = fast_lookup.barycentric_evalnd(interp_pts, interp_wts, table_log_coeffs, pt)
    t.report("interp")

    standard_scale = np.sqrt(np.linalg.norm(tri_normal(standard_tri)))
    interp_vals += np.log(standard_scale) * log_coeffs
    t.report("log correct")

    out = transform_from_standard(interp_vals, K, sm, labels, translation, R, scale)
    out = np.array(out).reshape((3,3,3,3))
    t.report("transform from standard")
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
    filename = 'data/H_50_0.010000_200_0.000000_14_6_adjacenttable.npy'
    to = Timer()

    params = filename.split('_')
    n_phi = int(params[5])
    n_pr = int(params[6])

    interp_pts, interp_wts = adjacent_interp_pts_wts(n_phi, n_pr)

    table_data = np.load(filename)
    table_limits = table_data[:,:,0]
    table_log_coeffs = table_data[:,:,1]

    out = np.empty((obs_tris.shape[0], 3, 3, 3, 3))

    start = time.time()
    va_pts = []
    va_which_pair = []
    va_obs_tris = []
    va_src_tris = []
    va_obs_basis = []
    va_src_basis = []
    for i in range(obs_tris.shape[0]):
        t = Timer(silent = True)
        obs_tri = pts[obs_tris[i]]
        src_tri = pts[src_tris[i]]
        split_pts, split_obs, split_src, obs_basis_tris, src_basis_tris = separate_tris(
            obs_tri, src_tri
        )
        t.report("separate tris")
        lookup_result = adjacent_lookup(
            (table_limits, table_log_coeffs, interp_pts, interp_wts),
            kernel, sm, pr,
            obs_tri, split_pts[split_obs[0]], split_pts[split_src[0]],
            remove_sing
        )
        t.report("lookup")
        out[i] = np.array(fast_lookup.sub_basis(
            lookup_result.flatten().tolist(),
            obs_basis_tris[0].tolist(),
            src_basis_tris[0].tolist())
        ).reshape((3,3,3,3))
        t.report("sub basis")

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
        t.report("create va subpairs")
    to.report("split and edge integrals")
    start = time.time()

    Iv = nearfield_op.vert_adj(
        nq_va, kernel, sm, pr,
        np.array(va_pts), np.array(va_obs_tris), np.array(va_src_tris)
    )
    to.report('vert adj subpairs')
    for i in range(Iv.shape[0]):
        Iv[i] = np.array(fast_lookup.sub_basis(
            Iv[i].flatten().tolist(),
            va_obs_basis[i].tolist(),
            va_src_basis[i].tolist())
        ).reshape((3,3,3,3))
        out[va_which_pair[i]] += Iv[i]
    to.report('vert adj subbasis')

    return out
