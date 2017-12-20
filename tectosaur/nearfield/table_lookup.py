import numpy as np

from tectosaur import get_data_filepath
from tectosaur.util.timer import Timer

from tectosaur.nearfield.interpolate import barycentric_evalnd
from tectosaur.nearfield.pairs_integrator import PairsIntegrator
from tectosaur.nearfield.table_params import *

from tectosaur.util.cpp import imp

from tectosaur.kernels import kernels

import cppimport.import_hook

import tectosaur.nearfield.standardize
import tectosaur.nearfield.edge_adj_setup
from tectosaur.nearfield._table_lookup import coincident_lookup_pts, coincident_lookup_from_standard,\
    adjacent_lookup_pts, adjacent_lookup_from_standard, vert_adj_subbasis

def lookup_interpolation_gpu(table_limits, table_log_coeffs,
        interp_pts, interp_wts, pts, float_type):
    vals = np.hstack((table_limits, table_log_coeffs)).copy()
    out = barycentric_evalnd(interp_pts.copy(), interp_wts.copy(), vals.copy(), pts.copy(), float_type)
    out = out.reshape((-1, 2, 81))
    return out[:,0,:], out[:,1,:]

def coincident_table(K, params, tri_pts, float_type):
    if tri_pts.shape[0] == 0:
        return np.empty((0, 3, 3, 3, 3))

    t = Timer(prefix = 'coincident')
    filename = kernels[K].co_table_filename
    filepath = get_data_filepath(filename)

    tableparams = filename.split('_')

    n_A = int(tableparams[5])
    n_B = int(tableparams[6])
    n_pr = int(tableparams[7])

    interp_pts, interp_wts = coincident_interp_pts_wts(n_A, n_B, n_pr)

    table_data = np.load(filepath)
    table_limits = table_data[:,:,0]
    table_log_coeffs = table_data[:,:,1]
    t.report("load table")

    # Shift to a three step process
    # 1) Get interpolation points
    lookup_pts, standard_tris = coincident_lookup_pts(tri_pts, params[1])
    t.report("get pts")

    # 2) Perform interpolation --> GPU!
    interp_vals, log_coeffs = lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, lookup_pts, float_type
    )
    t.report("interpolate")

    # 3) Transform to real space
    out = coincident_lookup_from_standard(
        standard_tris, interp_vals, log_coeffs, K, params[0]
    ).reshape((-1, 3, 3, 3, 3))
    t.report("from standard")


    return out

def adjacent_table(nq_va, K, params, pts, tris, ea_tri_indices, float_type):
    if ea_tri_indices.shape[0] == 0:
        return np.zeros((0,3,3,3,3))

    flip_symmetry = not kernels[K].flip_negate
    filename = kernels[K].adj_table_filename
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

    va, ea = adjacent_lookup_pts(pts, tris, ea_tri_indices, params[1], flip_symmetry)
    t.report("get pts")

    interp_vals, log_coeffs = lookup_interpolation_gpu(
        table_limits, table_log_coeffs, interp_pts, interp_wts, ea.pts, float_type
    )
    t.report("interpolation")

    out = adjacent_lookup_from_standard(
        interp_vals, log_coeffs, ea, K, params[0]
    ).reshape((-1, 3, 3, 3, 3))

    t.report("from standard")

    pairs_int = PairsIntegrator(K, params, float_type, 1, 1, va.pts, va.tris)
    Iv = pairs_int.vert_adj(nq_va, va.pairs)
    t.report('vert adj subpairs')

    vert_adj_subbasis(out, Iv, va, ea);
    t.report('vert adj subbasis')

    return out
