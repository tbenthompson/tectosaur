import logging
import numpy as np
import uuid

from scipy.optimize import fsolve

import tectosaur as tct
import cppimport.import_hook
from .pt_average import pt_averageD
from . import newton

def get_farfield_op(cfg):
    if cfg['use_fmm']:
        return tct.FMMFarfieldOp(
            mac = cfg['fmm_mac'],
            pts_per_cell = cfg['pts_per_cell'],
            order = cfg['fmm_order']
        )
    else:
        return tct.TriToTriDirectFarfieldOp

def build_elastic_op(m, cfg, K, obs_subset = None, src_subset = None):
    op_cfg = cfg['tectosaur_cfg']
    if K == 'U':
        fullKname = 'elastic' + K + '3'
    else:
        fullKname = 'elasticR' + K + '3'
    return tct.RegularizedSparseIntegralOp(
        op_cfg['quad_coincident_order'],
        op_cfg['quad_edgeadj_order'],
        op_cfg['quad_vertadj_order'],
        op_cfg['quad_far_order'],
        op_cfg['quad_near_order'],
        op_cfg['quad_near_threshold'],
        fullKname, fullKname, [1.0, cfg['pr']], m.pts, m.tris, op_cfg['float_type'],
        farfield_op_type = get_farfield_op(op_cfg),
        obs_subset = obs_subset,
        src_subset = src_subset,
    )

def print_length_scales(model):
    sigma_n = model.cfg['additional_normal_stress']

    mesh_L = np.max(np.sqrt(model.tri_size))
    Lb = model.cfg['sm'] * model.cfg['Dc'] / (sigma_n * model.cfg['b'])

    #TODO: Remove and replace with empirical version directly from matrix.
    hstar = (
        (np.pi * model.cfg['sm'] * model.cfg['Dc']) /
        (sigma_n * (model.cfg['b'] - model.cfg['a']))
    )
    hstarRA = (
        (2.0 / np.pi) * model.cfg['sm'] * model.cfg['b'] * model.cfg['Dc']
        / ((model.cfg['b'] - model.cfg['a']) ** 2 * sigma_n)
    )
    hstarRA3D = np.pi ** 2 / 4.0 * hstarRA

    # all_fields = np.vstack((Lb, hstar, np.ones_like(hstar) * mesh_L)).T
    # plot_fields(m, all_fields)
    print('hstar (2d antiplane, erickson and dunham 2014)', np.min(np.abs(hstar)))
    print('hstar_RA (2d antiplane, rubin and ampuero 2005)', np.min(np.abs(hstarRA)))
    print('hstar_RA3D (3d strike slip, lapusta and liu 2009)', np.min(np.abs(hstarRA3D)))
    print('cohesive zone length scale', np.min(Lb))
    print('mesh length scale', mesh_L)

def calc_derived_constants(cfg):
    if type(cfg['tectosaur_cfg']['log_level']) is str:
        log_level = getattr(logging, cfg['tectosaur_cfg']['log_level'])
        cfg['tectosaur_cfg']['log_level'] = log_level

    out_cfg = cfg.copy()

    # Shear wave speed (m/s)
    out_cfg['cs'] = np.sqrt(out_cfg['sm'] / out_cfg['density'])

    # The radiation damping coefficient (kg / (m^2 * s))
    out_cfg['eta'] = out_cfg['sm'] / (2 * out_cfg['cs'])
    return out_cfg

def remember(f):
    unique_id = uuid.uuid4().hex[:6].upper()
    def g(self, *args, **kwargs):
        if not hasattr(self, unique_id):
            setattr(self, unique_id, f(self, *args, **kwargs))
        return getattr(self, unique_id)
    return g

def rate_state_solve(model, traction, state):
    timer = model.cfg['Timer']()
    V = np.empty_like(model.field_inslipdir)
    newton.rate_state_solver(
        model.tri_normals, traction, state, V,
        model.cfg['a'], model.cfg['eta'],
        model.cfg['V0'], model.cfg.get('C', 0.0),
        model.cfg['additional_normal_stress'],
        1e-12, 50, int(model.n_dofs / model.n_tris),
        model.cfg.get('rs_separate_dims', False)
    )
    timer.report('newton')

    if model.basis_dim == 1:
        ptavg_V = V
    else:
        ptavg_V = np.empty_like(V)
        for d in range(3):
            ptavg_V.reshape((-1,3))[:,d] = pt_averageD(
                model.m.pts, model.m.get_tris('fault'), V.reshape((-1,3))[:,d].copy()
            )
    timer.report('pt avg')

    inslipdir_speed = np.sum(ptavg_V.reshape((-1,3)) * model.field_inslipdir_interior.reshape((-1,3)), axis = 1)
    inslipdir_speed /= np.linalg.norm(model.field_inslipdir_interior.reshape((-1,3)), axis = 1)
    if model.cfg.get('no_backslip', False):
        ptavg_V.reshape((-1,3))[inslipdir_speed < 0,:] = 0.0
        ptavg_V.reshape((-1,3))[np.isnan(inslipdir_speed)] = 0.0

    out = (
        model.field_inslipdir_edges * model.cfg['plate_rate']
        + ptavg_V.flatten()
    )
    timer.report('rate_state_solve --> out')
    return out

# State evolution law -- aging law.
def aging_law(cfg, V, state):
    return (cfg['b'] * cfg['V0'] / cfg['Dc']) * (
        np.exp((cfg['f0'] - state) / cfg['b']) - (V / cfg['V0'])
    )

def state_evolution(cfg, V, state):
    V_mag = np.linalg.norm(V.reshape(-1,3), axis = 1)
    return aging_law(cfg, V_mag, state)

def init_creep(model):
    V_i = model.cfg['plate_rate']
    def f(state):
        return aging_law(model.cfg, V_i, state)
    state_i = fsolve(f, 0.7)[0]
    sigma_n = model.cfg['additional_normal_stress']
    tau_i = newton.F(V_i, sigma_n, state_i, model.cfg['a'][0], model.cfg['V0'], model.cfg['C'])
    init_traction = tau_i * model.field_inslipdir_interior
    init_slip_deficit = model.traction_to_slip(init_traction)
    init_state =  state_i * np.ones((model.m.n_tris('fault') * 3))
    return 0, -init_slip_deficit, init_state

def check_naninf(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))
