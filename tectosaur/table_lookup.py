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

def lookup_interpolation_gpu(table_limits, table_log_coeffs,
        interp_pts, interp_wts, pts):

    t = Timer(silent = True)

    float_type = np.float64
    gpu_cfg = {'float_type': gpu.np_to_c_type(float_type)}
    module = gpu.load_gpu('table_lookup.cl', tmpl_args = gpu_cfg)
    dims = interp_pts.shape[1]
    fnc = getattr(module, 'lookup_interpolation' + str(dims))

    t.report("load module")

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
    t.report("run interpolation for " + str(n_tris) + " tris")
    return out[:, :, 0], out[:, :, 1]

def coincident_table(kernel, sm, pr, pts, tris):
    t = Timer(prefix = 'coincident')
    if kernel is 'U':
        filename = 'data/U_25_0.010000_16_0.000000_8_13_8_coincidenttable.npy'
    elif kernel is 'T':
        filename = 'data/T_25_0.000000_3_0.000000_12_13_7_coincidenttable.npy'
    elif kernel is 'A':
        filename = 'data/A_25_0.000000_3_0.000000_12_13_7_coincidenttable.npy'
    elif kernel is 'H':
        filename = 'data/H_100_0.003125_6_0.000001_12_17_9_coincidenttable.npy'

    params = filename.split('_')

    n_A = int(params[5])
    n_B = int(params[6])
    n_pr = int(params[7])

    interp_pts, interp_wts = coincident_interp_pts_wts(n_A, n_B, n_pr)

    tri_pts = pts[tris]

    table_data = np.load(filename)
    table_limits = table_data[:,:,0]
    table_log_coeffs = table_data[:,:,1]
    t.report("load table")

    # Shift to a three step process
    # 1) Get interpolation points
    pts, standard_tris = fast_lookup.coincident_lookup_pts(tri_pts, pr);
    t.report("get pts")

    # 2) Perform interpolation --> GPU!
    interp_vals, log_coeffs = lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, pts
    )
    t.report("interpolate")

    # 3) Transform to real space
    out = fast_lookup.coincident_lookup_from_standard(
        standard_tris, interp_vals, log_coeffs, kernel, sm
    ).reshape((-1, 3, 3, 3, 3))
    t.report("from standard")


    return out

def adjacent_table(nq_va, kernel, sm, pr, pts, obs_tris, src_tris):
    if obs_tris.shape[0] == 0:
        return np.zeros((0,3,3,3,3))

    if kernel is 'U':
        filename = 'data/U_25_0.010000_16_0.000000_7_8_adjacenttable.npy'
    elif kernel is 'T':
        filename = 'data/T_25_0.000000_3_0.000000_12_7_adjacenttable.npy'
    elif kernel is 'A':
        filename = 'data/A_25_0.000000_3_0.000000_12_7_adjacenttable.npy'
    elif kernel is 'H':
        filename = 'data/H_50_0.010000_200_0.000000_14_6_adjacenttable.npy'
    t = Timer(prefix = 'adjacent')

    params = filename.split('_')
    n_phi = int(params[5])
    n_pr = int(params[6])

    interp_pts, interp_wts = adjacent_interp_pts_wts(n_phi, n_pr)
    t.report("generate interp pts wts")

    table_data = np.load(filename)
    table_limits = table_data[:,:,0]
    table_log_coeffs = table_data[:,:,1]
    t.report("load table")

    obs_tris_pts = pts[obs_tris]
    src_tris_pts = pts[src_tris]
    va, ea = fast_lookup.adjacent_lookup_pts(obs_tris_pts, src_tris_pts, pr)
    t.report("get pts")

    interp_vals, log_coeffs = lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, np.array(ea.pts)
    )
    t.report("interpolation")

    out = fast_lookup.adjacent_lookup_from_standard(
        obs_tris_pts, interp_vals, log_coeffs, ea, kernel, sm
    ).reshape((-1, 3, 3, 3, 3))

    t.report("from standard")

    Iv = nearfield_op.vert_adj(
        nq_va, kernel, sm, pr,
        np.array(va.pts), np.array(va.obs_tris), np.array(va.src_tris)
    )
    t.report('vert adj subpairs')
    fast_lookup.vert_adj_subbasis(out, Iv, va);
    t.report('vert adj subbasis')

    return out
