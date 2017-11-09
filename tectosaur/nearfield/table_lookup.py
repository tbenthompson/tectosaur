import numpy as np

from tectosaur import get_data_filepath
from tectosaur.util.timer import Timer

from tectosaur.nearfield.interpolate import barycentric_evalnd
from tectosaur.nearfield.pairs_integrator import PairsIntegrator
from tectosaur.nearfield.table_params import *

from tectosaur.util.cpp import imp
standardize = imp('tectosaur.nearfield.standardize')
edge_adj_setup = imp('tectosaur.nearfield.edge_adj_setup')
fast_lookup = imp('tectosaur.nearfield.fast_lookup')

def lookup_interpolation_gpu(table_limits, table_log_coeffs,
        interp_pts, interp_wts, pts, float_type):
    vals = np.hstack((table_limits, table_log_coeffs)).copy()
    out = barycentric_evalnd(interp_pts.copy(), interp_wts.copy(), vals.copy(), pts.copy(), float_type)
    out = out.reshape((-1, 2, 81))
    return out[:,0,:], out[:,1,:]

def coincident_table(kernel, params, pts, tris, float_type):
    t = Timer(prefix = 'coincident')
    if kernel is 'elasticU3':
        filename = 'elasticU_25_0.010000_16_0.000000_8_13_8_coincidenttable.npy'
    elif kernel is 'elasticT3':
        filename = 'elasticT_25_0.000000_3_0.000000_12_13_7_coincidenttable.npy'
    elif kernel is 'elasticA3':
        filename = 'elasticA_25_0.000000_3_0.000000_12_13_7_coincidenttable.npy'
    elif kernel is 'elasticH3':
        filename = 'elasticH_100_0.003125_6_0.000001_12_17_9_coincidenttable.npy'
    filepath = get_data_filepath(filename)

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
        table_limits, table_log_coeffs, interp_pts, interp_wts, pts, float_type
    )
    t.report("interpolate")

    # 3) Transform to real space
    out = fast_lookup.coincident_lookup_from_standard(
        standard_tris, interp_vals, log_coeffs, kernel, params[0]
    ).reshape((-1, 3, 3, 3, 3))
    t.report("from standard")


    return out

def adjacent_table(nq_va, kernel, params, pts, tris, ea_tri_indices, float_type):
    if ea_tri_indices.shape[0] == 0:
        return np.zeros((0,3,3,3,3))

    flip_symmetry = False
    if kernel is 'elasticU3':
        filename = 'elasticU_25_0.010000_16_0.000000_7_8_adjacenttable.npy'
        flip_symmetry = True
    elif kernel is 'elasticT3':
        filename = 'elasticT_25_0.000000_3_0.000000_16_7_adjacenttable.npy'
    elif kernel is 'elasticA3':
        filename = 'elasticA_25_0.000000_3_0.000000_16_7_adjacenttable.npy'
    elif kernel is 'elasticH3':
        filename = 'elasticH_50_0.010000_200_0.000000_14_6_adjacenttable.npy'
        flip_symmetry = True
    filepath = get_data_filepath(filename)

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

    va, ea = fast_lookup.adjacent_lookup_pts(pts, tris, ea_tri_indices, params[1], flip_symmetry)
    t.report("get pts")

    interp_vals, log_coeffs = lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, ea.pts, float_type
    )
    t.report("interpolation")

    out = fast_lookup.adjacent_lookup_from_standard(
        interp_vals, log_coeffs, ea, kernel, params[0]
    ).reshape((-1, 3, 3, 3, 3))

    t.report("from standard")

    pairs_int = PairsIntegrator(kernel, params, float_type, 1, 1, va.pts, va.tris)
    Iv = pairs_int.vert_adj(nq_va, va.pairs)
    t.report('vert adj subpairs')

    fast_lookup.vert_adj_subbasis(out, Iv, va, ea);
    t.report('vert adj subbasis')

    return out
