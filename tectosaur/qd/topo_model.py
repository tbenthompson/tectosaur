import numpy as np
from scipy.sparse.linalg import cg, lsmr, gmres
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
    state_evolution, build_elastic_op, check_naninf)
from .plotting import plot_fields

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
            return False

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

        if check_naninf(fault_V):
            return False

        dstatedt = state_evolution(self.cfg, fault_V, state)
        timer.report('dstatedt')

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

    @remember
    def T(self):
        setup_logging(self.cfg)
        return build_elastic_op(self.m, self.cfg, 'T')

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
        slipdir = self.cfg['slipdir']
        if type(slipdir) is tuple:
            for d in range(3):
                val = self.cfg['slipdir'][d]
                self.field_inslipdir_interior.reshape((-1,3))[:,d] *= val
                self.field_inslipdir.reshape((-1,3))[:,d] = val
        else:
            assert(slipdir.shape[0] == self.field_inslipdir.shape[0])
            self.field_inslipdir = slipdir
            self.field_inslipdir_interior *= slipdir

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

def get_slip_to_disp(m, cfg, T):
    base_cs = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    # base_cs.extend(tct.free_edge_constraints(m.tris))

    mass_op = tct.MultOp(tct.MassOp(3, m.pts, m.tris), 0.5)
    iop = tct.SumOp([T, mass_op])

    def f(slip):
        cs = base_cs + tct.all_bc_constraints(
            m.n_tris('surf'), m.n_tris(), slip
        )
        cm, c_rhs, _ = tct.build_constraint_matrix(cs, m.n_dofs())

        rhs = -iop.dot(c_rhs)
        out = tectosaur_topo.solve.iterative_solve(
            iop, cm, rhs, lambda x: x, dict(solver_tol = 1e-6)
        )
        return out + c_rhs
    return f

def depth_traction_constraints(m):
    from tectosaur.faultea.mesh_fncs import get_surf_fault_edges, get_surf_fault_pts
    from tectosaur.continuity import find_touching_pts
    from tectosaur.constraint_builders import find_free_edges,free_edge_dofs
    fe_dofs = free_edge_dofs(m.tris, find_free_edges(m.tris))
    sf_pts = get_surf_fault_pts(m.get_tris('surf'), m.get_tris('fault'))
    touching_pts = find_touching_pts(m.get_tris('fault'))

    pairs = []
    for p in sf_pts:
        tri_idx1, corner_idx1 = touching_pts[p][0]
        dof1 = m.n_tris('surf') * 3 + tri_idx1 * 3 + corner_idx1
        if dof1 in fe_dofs:
            continue

        connected_pts = []
        for tri_idx, corner_idx in touching_pts[p]:
            connected_pts.extend(m.get_tris('fault')[tri_idx].tolist())
        connected_pts = list(set(connected_pts) - {p})
        vs = m.pts[p] - m.pts[connected_pts]
        max_depth_idx = np.argmax(vs[:,2])
        tri_idx2, corner_idx2 = touching_pts[connected_pts[max_depth_idx]][0]
        for i in range(vs.shape[0]):
            dof2 = m.n_tris('surf') * 3 + tri_idx2 * 3 + corner_idx2
            if dof2 not in fe_dofs:
                break
            max_depth_idx = (max_depth_idx + 1) % vs.shape[0]
            tri_idx2, corner_idx2 = touching_pts[connected_pts[max_depth_idx]][0]

        for d in range(3):
            terms = [
                m.n_dofs('surf') + tri_idx1 * 9 + corner_idx1 * 3 + d,
                m.n_dofs('surf') + tri_idx2 * 9 + corner_idx2 * 3 + d
            ]
            pairs.append(terms)
    return pairs

def surface_fault_admissibility(m):
    from tectosaur.faultea.mesh_fncs import get_surf_fault_edges, get_surf_fault_pts
    from tectosaur.continuity import (
        find_touching_pts, constant_stress_constraint, equilibrium_constraint
    )
    from tectosaur.constraint_builders import find_free_edges,free_edge_dofs
    fe_dofs = free_edge_dofs(m.tris, find_free_edges(m.tris))
    sf_pts = get_surf_fault_pts(m.get_tris('surf'), m.get_tris('fault'))
    touching_pts = find_touching_pts(m.tris)

    cs = []
    for p in sf_pts:
        surf_idxs = [t for t in touching_pts[p] if t[0] < m.n_tris('surf')]
        fault_idxs = [t for t in touching_pts[p] if t[0] >= m.n_tris('surf')]
        for f in fault_idxs:
            tri_data1 = (m.pts[m.tris[f[0]]], f[0], f[1])
            for s in surf_idxs:
                tri_data2 = (m.pts[m.tris[s[0]]], s[0], s[1])
                cs.append(constant_stress_constraint(tri_data1, tri_data2))
                # cs.append(equilibrium_constraint(tri_data1))
                # cs.append(equilibrium_constraint(tri_data2))
    return cs


def get_disp_slip_to_traction(m, cfg, H):
    csTS = tct.continuity_constraints(m.pts, m.get_tris('surf'), int(1e9))
    # csTS = tct.all_bc_constraints(0, m.n_tris('surf'), np.zeros(m.n_dofs('surf')))
    csTF = tct.continuity_constraints(m.pts, m.get_tris('fault'), int(1e9))
    csT = tct.build_composite_constraints((csTS, 0), (csTF, m.n_dofs('surf')))
    csT.extend(tct.free_edge_constraints(m.tris))
    cmT, c_rhsT, _ = tct.build_constraint_matrix(csT, m.n_dofs())
    np.testing.assert_almost_equal(c_rhsT, 0.0)

    cmU = cmT

    traction_mass_op = tct.MassOp(
        cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris
    )
    constrained_traction_mass_op = cmU.T.dot(traction_mass_op.mat.dot(cmT))

    def f(disp_slip):
        t = cfg['Timer']()
        def callback(x):
            callback.iter += 1
            print(callback.iter)
        callback.iter = 0

        rhs = -cmU.T.dot(H.dot(disp_slip.flatten()))
        t.report('rhs')

        soln = cg(constrained_traction_mass_op, rhs)
        #soln = lsmr(constrained_traction_mass_op, rhs)#, callback = callback)
        t.report('lsmr')

        out = cfg['sm'] * (cmT.dot(soln[0]) + c_rhsT)
        t.report('out')
        return out
    return f

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
