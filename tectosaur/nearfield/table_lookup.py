import os

import time
import numpy as np

import tectosaur
from tectosaur.util.timer import Timer
import tectosaur.nearfield.limit as limit
import tectosaur.nearfield.vert_adj as nearfield_op
import tectosaur.util.gpu as gpu

from tectosaur.nearfield.table_params import *

import cppimport
fast_lookup = cppimport.imp("tectosaur.nearfield.fast_lookup").nearfield.fast_lookup

def lookup_interpolation_gpu(table_limits, table_log_coeffs,
        interp_pts, interp_wts, pts):

    t = Timer(silent = True)

    float_type = np.float64
    gpu_cfg = {'float_type': gpu.np_to_c_type(float_type)}
    module = gpu.load_gpu('nearfield/table_lookup.cl', tmpl_args = gpu_cfg)
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

def coincident_table(kernel, params, pts, tris):
    t = Timer(prefix = 'coincident')
    if kernel is 'elasticU':
        filename = 'elasticU_25_0.010000_16_0.000000_8_13_8_coincidenttable.npy'
    elif kernel is 'elasticT':
        filename = 'elasticT_25_0.000000_3_0.000000_12_13_7_coincidenttable.npy'
    elif kernel is 'elasticA':
        filename = 'elasticA_25_0.000000_3_0.000000_12_13_7_coincidenttable.npy'
    elif kernel is 'elasticH':
        filename = 'elasticH_100_0.003125_6_0.000001_12_17_9_coincidenttable.npy'
    filepath = tectosaur.get_data_filepath(filename)

    tableparams = filename.split('_')

    n_A = int(tableparams[5])
    n_B = int(tableparams[6])
    n_pr = int(tableparams[7])

    interp_pts, interp_wts = coincident_interp_pts_wts(n_A, n_B, n_pr)

    tri_pts = pts[tris]

    table_data = np.load(filepath)
    table_limits = table_data[:,:,0]
    table_log_coeffs = table_data[:,:,1]
    t.report("load table")

    # Shift to a three step process
    # 1) Get interpolation points
    pts, standard_tris = fast_lookup.coincident_lookup_pts(tri_pts, params[1])
    t.report("get pts")

    # 2) Perform interpolation --> GPU!
    interp_vals, log_coeffs = lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, pts
    )
    t.report("interpolate")

    # 3) Transform to real space
    out = fast_lookup.coincident_lookup_from_standard(
        standard_tris, interp_vals, log_coeffs, kernel, params[0]
    ).reshape((-1, 3, 3, 3, 3))
    t.report("from standard")


    return out

def adjacent_table(nq_va, kernel, params, pts, obs_tris, src_tris):
    if obs_tris.shape[0] == 0:
        return np.zeros((0,3,3,3,3))

    flip_symmetry = False
    if kernel is 'elasticU':
        filename = 'elasticU_25_0.010000_16_0.000000_7_8_adjacenttable.npy'
        flip_symmetry = True
    elif kernel is 'elasticT':
        filename = 'elasticT_25_0.000000_3_0.000000_16_7_adjacenttable.npy'
    elif kernel is 'elasticA':
        filename = 'elasticA_25_0.000000_3_0.000000_16_7_adjacenttable.npy'
    elif kernel is 'elasticH':
        filename = 'elasticH_50_0.010000_200_0.000000_14_6_adjacenttable.npy'
        flip_symmetry = True
    filepath = tectosaur.get_data_filepath(filename)

    t = Timer(prefix = 'adjacent')

    tableparams = filename.split('_')
    n_phi = int(tableparams[5])
    n_pr = int(tableparams[6])

    interp_pts, interp_wts = adjacent_interp_pts_wts(n_phi, n_pr)
    t.report("generate interp pts wts")

    table_data = np.load(filepath)
    table_limits = table_data[:,:,0]
    table_log_coeffs = table_data[:,:,1]
    t.report("load table")

    obs_tris_pts = pts[obs_tris]
    src_tris_pts = pts[src_tris]
    va, ea = fast_lookup.adjacent_lookup_pts(obs_tris_pts, src_tris_pts, params[1], flip_symmetry)
    t.report("get pts")

    interp_vals, log_coeffs = lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, np.array(ea.pts)
    )
    t.report("interpolation")

    out = fast_lookup.adjacent_lookup_from_standard(
        obs_tris_pts, interp_vals, log_coeffs, ea, kernel, params[0]
    ).reshape((-1, 3, 3, 3, 3))

    t.report("from standard")

    # np.save('playground/vert_adj_test2.npy', (np.array(va.pts), np.array(va.obs_tris), np.array(va.src_tris)))
    # import sys; sys.exit()
    Iv = nearfield_op.vert_adj(
        nq_va, kernel, params,
        np.array(va.pts), np.array(va.obs_tris), np.array(va.src_tris)
    )
    t.report('vert adj subpairs')
    fast_lookup.vert_adj_subbasis(out, Iv, va);
    t.report('vert adj subbasis')

    return out
