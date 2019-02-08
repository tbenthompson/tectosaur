import numpy as np
from scipy.sparse.linalg import cg
import scipy.sparse

import tectosaur as tct
import tectosaur_topo
from tectosaur.util.geometry import unscaled_normals
from tectosaur.util.timer import Timer
from tectosaur.constraints import ConstraintEQ, Term

from . import siay
from .full_model import setup_logging
from .model_helpers import (
    calc_derived_constants, remember, rate_state_solve,
    state_evolution, build_elastic_op)
from .plotting import plot_fields

def check_naninf(x):
    return np.any(np.isnan(x)) or np.any(np.isinf(x))

class TopoModel:
    def __init__(self, m, cfg):#, which_side):
        cfg = calc_derived_constants(cfg)
        # self.which_side = which_side
        self.cfg = cfg
        self.cfg['Timer'] = self.cfg.get(
            'Timer',
            lambda: Timer(output_fnc = lambda x: None)
        )
        self.setup_mesh(m)
        self.setup_edge_bcs()
        self.disp_deficit = np.zeros(m.n_dofs('surf'))
        self.old_slip_deficit = np.zeros(m.n_dofs('fault'))

    def restart(self, t, y):
        slip, state = self.get_components(y)
        self.old_slip_deficit = slip
        self.disp_deficit = self.m.get_dofs(self.slip_to_disp(slip), 'surf')

    def make_derivs(self):
        return self.get_derivs

    def get_derivs(self, t, y):
        if check_naninf(y):
            return np.inf * y
        data = self.solve_for_full_state(t, y)
        if not data:
            return np.inf * y
        disp_slip, state, traction, fault_V, dstatedt = data

        slip_deficit_rate = (
            self.field_inslipdir_interior * self.cfg['plate_rate']
            - self.ones_interior * fault_V
        )
        return np.concatenate((slip_deficit_rate, dstatedt))

    def get_components(self, y):
        deficit_end = self.m.n_dofs('fault')
        slip_deficit = y[:deficit_end]
        state = y[deficit_end:]
        return slip_deficit, state

    def do_fast_step(self):
        return self.disp_deficit is not None and self.cfg['fast_surf_disp']

    def solve_for_full_state(self, t, y):
        timer = self.cfg['Timer']()
        slip_deficit, state = self.get_components(y)
        if np.any(state < 0) or np.any(state > 1.2):
            print("BAD STATE VALUES")
            print(state)

        do_fast_step = self.do_fast_step()
        if do_fast_step:
            deficit = np.concatenate((self.disp_deficit, slip_deficit))
        else:
            deficit = self.slip_to_disp(slip_deficit)
        timer.report(f'disp_slip(fast={do_fast_step})')

        traction = self.disp_slip_to_traction(deficit)
        fault_traction = traction[self.m.n_dofs('surf'):].copy()
        timer.report('traction')

        fault_V = rate_state_solve(self, fault_traction, state)
        timer.report('fault_V')

        dstatedt = state_evolution(self.cfg, fault_V, state)
        timer.report('dstatedt')

        if check_naninf(fault_V):
            return False

        out = deficit, state, traction, fault_V, dstatedt
        return out

    def post_step(self, ts, ys, rk):
        t = self.cfg['Timer']()
        slip_deficit, _ = self.get_components(ys[-1])
        if len(ts) % self.cfg['refresh_disp_freq'] == 0:
            full_deficit = self.m.get_dofs(
                self.slip_to_disp(slip_deficit),
                'surf'
            )
            self.old_slip_deficit = slip_deficit
        else:
            self.disp_deficit = self.slip_to_disp_one_jacobi_step(
                self.disp_deficit, slip_deficit, self.old_slip_deficit
            )
            self.old_slip_deficit = slip_deficit
        t.report('post step')

    def display(self, t, y, plotter = plot_fields):
        print(t / siay)
        data = self.solve_for_full_state(t, y)
        disp_slip, state, traction, fault_V, dstatedt = data
        print('slip deficit')
        plotter(self, self.m.get_dofs(disp_slip, 'fault'), dims = [0,1])
        print('surface displacement deficit')
        plotter(self, self.m.get_dofs(disp_slip, 'surf'), which = 'surf', dims = [0,1])
        print('fault V')
        plotter(self, np.log10(np.abs(fault_V) + 1e-40), dims = [0,1])
        print('traction on fault')
        plotter(self, self.m.get_dofs(traction, 'fault'), dims = [0,1])
        print('traction on surface')
        plotter(self, self.m.get_dofs(traction, 'surf'), which = 'surf', dims = [0,1])
        print('state')
        plotter(self, state, dims = [0,1])

    @property
    @remember
    def slip_to_disp(self):
        return get_slip_to_disp(self.m, self.cfg, self.H())

    @property
    @remember
    def slip_to_disp_one_jacobi_step(self):
        return get_slip_to_disp_one_jacobi_step(self.m, self.cfg, self.H())

    @property
    @remember
    def disp_slip_to_traction(self):
        return get_disp_slip_to_traction(self.m, self.cfg, self.H())

    @property
    @remember
    def full_traction_to_slip(self):
        return get_traction_to_slip(self.m, self.cfg, self.H())

    @property
    def traction_to_slip(self):
        def f(traction):
            full_traction = np.zeros(self.m.n_dofs())
            full_traction[self.m.n_dofs('surf'):] = traction
            return self.full_traction_to_slip(full_traction)
        return f

    @remember
    def H(self):
        setup_logging(self.cfg)
        return build_elastic_op(self.m, self.cfg, 'H')

    def setup_mesh(self, m):
        self.m = m
        self.fault_start_idx = m.get_start('fault')
        fault_tris = self.m.get_tris('fault')

        self.unscaled_tri_normals = unscaled_normals(self.m.pts[fault_tris])
        self.tri_size = np.linalg.norm(self.unscaled_tri_normals, axis = 1)
        self.tri_normals = self.unscaled_tri_normals / self.tri_size[:, np.newaxis]

        self.n_tris = self.m.tris.shape[0]
        self.basis_dim = 3
        self.n_dofs = self.basis_dim * self.n_tris

    def setup_edge_bcs(self):
        cs = tct.free_edge_constraints(self.m.tris)
        cm, c_rhs, _ = tct.build_constraint_matrix(cs, self.m.n_dofs())
        constrained_slip = np.ones(cm.shape[1])
        self.ones_interior = cm.dot(constrained_slip)[self.m.n_dofs('surf'):]

        self.field_inslipdir_interior = self.ones_interior.copy()
        self.field_inslipdir = self.ones_interior.copy()
        for d in range(3):
            val = self.cfg['slipdir'][d]
            self.field_inslipdir_interior.reshape((-1,3))[:,d] *= val
            self.field_inslipdir.reshape((-1,3))[:,d] = val

        self.field_inslipdir_edges = self.field_inslipdir - self.field_inslipdir_interior

        self.surf_tri_centers = np.mean(self.m.pts[self.m.get_tris('surf')], axis = 1)

    @property
    @remember
    def locked_fault_surf_disp_deficit(self):
        return self.m.get_dofs(
            self.slip_to_disp(self.field_inslipdir_interior),
            'surf'
        )

def get_slip_to_disp_one_jacobi_step(m, cfg, H):
    base_cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    base_cs.extend(tct.free_edge_constraints(m.tris))
    cs = base_cs + tct.all_bc_constraints(
        m.n_tris('surf'), m.n_tris(), np.zeros(m.n_dofs('fault'))
    )
    cm, _, rhs_mat = tct.build_constraint_matrix(cs, m.n_dofs())

    near = H.nearfield.full_scipy_mat_no_correction()
    diag = cm.T.dot(near.dot(cm)).diagonal()

    def f(disp, slip, old_slip):
        t = cfg['Timer']()
        c_rhs = rhs_mat.dot(np.concatenate((np.zeros(len(base_cs)), slip)))
        c_rhs_old = rhs_mat.dot(np.concatenate((np.zeros(len(base_cs)), old_slip)))
        t.report('constraints')

        start_val = np.concatenate((disp, slip))
        delta = cm.dot(
            (1.0 / diag) * (cm.T.dot(H.dot(start_val)))
        )
        out = start_val - delta - c_rhs_old + c_rhs
        t.report('jacobi step')
        return m.get_dofs(out, 'surf')
    return f

def get_slip_to_disp(m, cfg, H):
    base_cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    base_cs.extend(tct.free_edge_constraints(m.tris))

    def f(slip):
        cs = base_cs + tct.all_bc_constraints(
            m.n_tris('surf'), m.n_tris(), slip
        )
        cm, c_rhs, _ = tct.build_constraint_matrix(cs, m.n_dofs())

        rhs = -H.dot(c_rhs)
        out = tectosaur_topo.solve.iterative_solve(
            H, cm, rhs, lambda x: x, dict(solver_tol = 1e-4)
        )
        return out + c_rhs
    return f

def get_disp_slip_to_traction(m, cfg, H):
    csS = tct.all_bc_constraints(0, m.n_tris('surf'), np.zeros(m.n_dofs('surf')))
    csS.extend(tct.free_edge_constraints(m.get_tris('surf')))
    csF = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    csF.extend(tct.free_edge_constraints(m.get_tris('fault')))
    cs = tct.build_composite_constraints((csS, 0), (csF, m.n_dofs('surf')))
    cm, c_rhs, _ = tct.build_constraint_matrix(cs, m.n_dofs())

    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    constrained_traction_mass_op = cm.T.dot(traction_mass_op.mat.dot(cm))

    def f(disp_slip):
        np.testing.assert_almost_equal(c_rhs, 0.0)
        def callback(x):
            callback.iter += 1
            print(callback.iter)
        callback.iter = 0

        rhs = -H.dot(disp_slip)
        constrained_rhs = cm.T.dot(rhs)
        soln = cg(constrained_traction_mass_op, constrained_rhs)#, callback = callback)
        out = cfg['sm'] * cm.dot(soln[0])
        return out
    return f

def fault_surf_intersection_traction_constraints(m):
    surf_tris = m.get_tris('surf')
    unscaled_ns = tct.util.geometry.unscaled_normals(m.pts[surf_tris])
    ns = tct.util.geometry.normalize(unscaled_ns)
    pt_ns = np.zeros((m.pts.shape[0], 3))
    for i in range(m.n_tris('surf')):
        for d in range(3):
            pt_ns[m.tris[i,d]] += ns[i]
    n_ns = np.ones((m.pts.shape[0]))
    unique, counts = np.unique(surf_tris, return_counts=True)
    n_ns[unique] = counts
    pt_ns /= n_ns[:,np.newaxis]
    cs = []
    for i in m.get_tri_idxs('fault'):
        for d in range(3):
            n = pt_ns[m.tris[i,d]]
            if np.all(n == 0):
                continue
            assert(np.where(surf_tris == m.tris[i,d])[0].shape[0] > 0)
            ts = []
            for d2 in range(3):
                if n[d2] == 0.0:
                    continue
                fault_dof = i * 9 + d * 3 + d2
                ts.append(Term(n[d2], fault_dof))
            cs.append(ConstraintEQ(ts, 0))
    return cs


def get_traction_to_slip(m, cfg, H):
    t = cfg['Timer']()
    csS = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    csF = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    cs = tct.build_composite_constraints((csS, 0), (csF, m.n_dofs('surf')))
    cs.extend(tct.free_edge_constraints(m.tris))

    cm, c_rhs, _ = tct.build_constraint_matrix(cs, m.n_dofs())
    cm = cm.tocsr()
    cmT = cm.T.tocsr()
    t.report('t2s -- build constraints')

    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    np.testing.assert_almost_equal(c_rhs, 0.0)
    t.report('t2s -- build massop')

    # nearfield_H = H.nearfield.full_scipy_mat_no_correction()
    # diag_H = nearfield_H.diagonal()
    # def prec(x):
    #     return cm.T.dot(cm.dot(x) / diag_H)

    # nearfield_H = H.nearfield.full_scipy_mat_no_correction()
    # constrained_nearfield_H = cmT.dot(nearfield_H.dot(cm))
    # t.report('t2s -- constrained nearfield')
    # spilu = scipy.sparse.linalg.spilu(constrained_nearfield_H)
    # t.report('t2s -- spilu')
    # def prec(x):
    #     return spilu.solve(x)

    # U = build_elastic_op(m, cfg, 'U')
    # nearfield_U = U.nearfield.full_scipy_mat_no_correction()
    # diag_U = nearfield_U.diagonal()
    # def prec(x):
    #     return cmT.dot(U.dot(cm.dot(x)))

    def prec(x):
        return x

    def f(traction):
        rhs = -traction_mass_op.dot(traction / cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            H, cm, rhs, prec, dict(solver_tol = 1e-4)
        )
        return out
    f.H = H
    f.cm = cm
    f.traction_mass_op = traction_mass_op
    return f

def refine_mesh_and_initial_conditions(m, slip_deficit):
    fields = [slip_deficit.reshape((-1,3,3))[:,:,d] for d in range(3)]
    m_refined, new_fields = tct.mesh.refine.refine_to_size(
        (m.pts, m.tris), 0.000000001, recurse = False, fields = fields
    )
    slip_deficit2 = np.swapaxes(np.swapaxes(np.array(new_fields), 0, 2), 0, 1)
    surf_tris = m_refined[1][:m.n_tris('surf') * 4].copy()
    fault_tris = m_refined[1][m.n_tris('surf') * 4:].copy()
    m2 = tct.CombinedMesh.from_named_pieces([
        ('surf', (m_refined[0], surf_tris)),
        ('fault', (m_refined[0], fault_tris))
    ])
    return m2, slip_deficit2
