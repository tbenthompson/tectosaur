import logging
import numpy as np

from scipy.sparse.linalg import cg

import tectosaur as tct
import tectosaur_topo
from tectosaur.mesh.combined_mesh import CombinedMesh
from tectosaur.util.geometry import unscaled_normals
from tectosaur.constraint_builders import free_edge_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.util.timer import Timer

from . import siay
from .model_helpers import (
    calc_derived_constants, remember, build_elastic_op,
    rate_state_solve, state_evolution)
from .plotting import plot_fields

class FullspaceModel:
    def __init__(self, m, cfg):
        cfg = calc_derived_constants(cfg)
        self.cfg = cfg
        if not 'Timer' in self.cfg or self.cfg['Timer'] is None:
            self.cfg['Timer'] = lambda: Timer(output_fnc = lambda x: None)
        self.setup_mesh(m)
        self.setup_edge_bcs()

    def make_derivs(self):
        def derivs(t, y):
            data = self.solve_for_full_state(t, y)
            slip, slip_deficit, state, traction, V, dstatedt = data
            return np.concatenate((V, dstatedt))
        return derivs

    def get_components(self, y):
        slip_end = self.m.n_dofs('fault')
        slip = y[:slip_end]
        state = y[slip_end:]
        return slip, state

    def solve_for_full_state(self, t, y):
        out = dict()
        timer = self.cfg['Timer']()

        n_total_dofs = y.shape[0]
        n_slip_dofs = n_total_dofs // 4 * 3
        slip, state = y[:n_slip_dofs], y[n_slip_dofs:]
        timer.report('separate_slip_state')

        plate_motion = (t * self.cfg['plate_rate']) * self.field_inslipdir
        slip_deficit = self.ones_interior * (plate_motion.reshape(-1) - slip)
        timer.report('get_slip_deficit')

        traction = self.slip_to_traction(slip_deficit)
        timer.report('slip_to_traction')

        V = rate_state_solve(self, traction, state)
        timer.report('rate_state_solve')

        dstatedt = state_evolution(self.cfg, V, state)
        timer.report('state_evolution')

        return slip, slip_deficit, state, traction, V, dstatedt

    def post_step(self, ts, ys, rk):
        pass

    def display(self, t, y, plotter = plot_fields):
        print(t / siay)
        data = self.solve_for_full_state(t, y)
        slip, slip_deficit, state, traction, V, dstatedt = data
        print('slip')
        plotter(self, slip)
        print('V')
        plotter(self, np.log10(np.abs(V) + 1e-40))
        print('traction')
        plotter(self, traction)
        print('state')
        plotter(self, state)

    @property
    @remember
    def slip_to_traction(self):
        return get_slip_to_traction(self.m, self.cfg)

    @property
    @remember
    def traction_to_slip(self):
        return get_traction_to_slip(self.m, self.cfg)

    def setup_mesh(self, m):
        if type(m) is CombinedMesh:
            self.m = m
        else:
            self.m = CombinedMesh.from_named_pieces([('fault', m)])

        self.unscaled_tri_normals = unscaled_normals(self.m.pts[self.m.tris])
        self.tri_size = np.linalg.norm(self.unscaled_tri_normals, axis = 1)
        self.tri_normals = self.unscaled_tri_normals / self.tri_size[:, np.newaxis]

        self.n_tris = self.m.tris.shape[0]
        self.basis_dim = 3
        self.n_dofs = self.basis_dim * self.n_tris

    def setup_edge_bcs(self):
        cs = free_edge_constraints(self.m.get_tris('fault'))
        cm, c_rhs, _ = build_constraint_matrix(cs, self.m.n_dofs('fault'))

        constrained_slip = np.ones(cm.shape[1])
        self.ones_interior = cm.dot(constrained_slip)

        self.field_inslipdir_interior = self.ones_interior.copy()
        self.field_inslipdir = self.field_inslipdir_interior.copy()
        for d in range(3):
            val = self.cfg.get('slipdir', (1.0, 0.0, 0.0))[d]
            self.field_inslipdir_interior.reshape(-1,3)[:,d] *= val
            self.field_inslipdir.reshape(-1,3)[:,d] = val

        self.field_inslipdir_edges = (
            self.field_inslipdir - self.field_inslipdir_interior
        )


def setup_slip_traction(m, cfg):
    setup_logging(cfg)
    cm = build_continuity(m, cfg)
    H = build_elastic_op(m, cfg, 'H')
    traction_mass_op = tct.MassOp(cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris)
    return H, traction_mass_op, cm

def setup_logging(cfg):
    tct.logger.setLevel(cfg['tectosaur_cfg']['log_level'])
    tectosaur_topo.logger.setLevel(cfg['tectosaur_cfg']['log_level'])

def build_continuity(m, cfg):
    cs = tct.continuity_constraints(m.pts, m.tris, m.tris.shape[0])
    cs.extend(free_edge_constraints(m.get_tris('fault')))
    cm, c_rhs, _ = build_constraint_matrix(cs, m.n_dofs('fault'))
    return cm

def get_slip_to_traction(m, cfg):
    def f(slip):
        t = cfg['Timer']()
        rhs = -f.H.dot(slip)
        t.report('H.dot')
        soln = cg(f.constrained_traction_mass_op, f.cm.T.dot(rhs))
        out = cfg['sm'] * f.cm.dot(soln[0])
        t.report('solve')

        if cfg.get('only_x', False):
            out.reshape((-1,3))[:,1] = 0.0
            out.reshape((-1,3))[:,2] = 0.0
        t.report('return')
        return out

    setup_logging(cfg)
    f.H, f.traction_mass_op, f.cm = setup_slip_traction(m, cfg)
    f.constrained_traction_mass_op = f.cm.T.dot(f.traction_mass_op.mat.dot(f.cm))

    return f

def get_traction_to_slip(m, cfg):
    def f(traction):
        rhs = -f.traction_mass_op.dot(traction / cfg['sm'])
        out = tectosaur_topo.solve.iterative_solve(
            f.H, f.cm, rhs, lambda x: x, dict(solver_tol = 1e-8)
        )
        return out
    f.H, f.traction_mass_op, f.cm = setup_slip_traction(m, cfg)
    return f
