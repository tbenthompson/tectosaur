import numpy as np
import scipy.sparse.linalg

import cutde.fullspace

from tectosaur.mesh.combined_mesh import CombinedMesh
from .helpers import tri_normal_info

from .model_helpers import calc_derived_constants

class TDEModel:
    def __init__(self, m, cfg):
        cfg = calc_derived_constants(cfg)
        self.cfg = cfg
        self.setup_mesh(m)
        self.setup_edge_bcs()

    @property
    def tde_matrix(self):
        if getattr(self, '_tde_matrix', None) is None:
            self._tde_matrix = tde_matrix(self)
        return self._tde_matrix

    @property
    def inv_tde_matrix(self):
        if getattr(self, '_inv_tde_matrix', None) is None:
            self._inv_tde_matrix = np.linalg.inv(self.tde_matrix)
        return self._inv_tde_matrix

    @property
    def slip_to_traction(self):
        if getattr(self, '_slip_to_traction', None) is None:
            self._slip_to_traction = get_tde_slip_to_traction(self.tde_matrix, self.cfg)
        return self._slip_to_traction

    def setup_mesh(self, m):
        if type(m) is CombinedMesh:
            self.m = m
        else:
            self.m = CombinedMesh.from_named_pieces([('fault', m)])
        self.unscaled_normals, self.tri_size, self.tri_normals = \
            tri_normal_info(self.m)
        self.tri_centers = np.mean(self.m.pts[self.m.tris], axis = 1)

        self.n_tris = self.m.tris.shape[0]
        self.basis_dim = 1
        self.n_dofs = self.basis_dim * self.n_tris

    def setup_edge_bcs(self):
        self.ones_interior = np.ones(self.n_tris * 3)
        self.field_inslipdir_interior = np.empty(self.n_tris * 3)
        for d in range(3):
            val = self.cfg.get('slipdir', (1.0, 0.0, 0.0))[d]
            self.field_inslipdir_interior.reshape(-1,3)[:,d] = val

        self.field_inslipdir = self.field_inslipdir_interior.copy()
        self.field_inslipdir_edges = (
            self.field_inslipdir - self.field_inslipdir_interior
        )

def build_xyz_slip_inputs(model):
    tensile_slip_vec = model.tri_normals
    assert(np.all(np.abs(tensile_slip_vec.dot([0,0,1])) < (1.0 - 1e-7)))
    strike_slip_vec = np.cross(tensile_slip_vec, [0,0,-1.0])
    dip_slip_vec = np.cross(tensile_slip_vec, strike_slip_vec)
    slip = np.zeros((model.n_tris, 3, 3))
    for d in range(3):
        v = np.zeros(3)
        v[d] = 1.0
        slip[:, d, 0] = -strike_slip_vec.dot(v)
        slip[:, d, 1] = -dip_slip_vec.dot(v)
        slip[:, d, 2] = -tensile_slip_vec.dot(v)
    slip = slip.reshape((-1, 3))
    return slip

def tde_stress_matrix(model):
    tri_pts = model.m.pts[model.m.tris]
    tri_pts_3 = np.repeat(tri_pts, 3, axis = 0)
    slip = build_xyz_slip_inputs(model)
    all_strains = cutde.fullspace.clu_strain_all_pairs(
        model.tri_centers, tri_pts_3, slip, model.cfg['pr']
    )
    stress = cutde.fullspace.strain_to_stress(
        all_strains.reshape((-1, 6)),
        model.cfg['sm'], model.cfg['pr']
    ).reshape((model.n_tris, 3 * model.n_tris, 6))
    return stress

def stress_to_traction(normals, stress):
    # map from 6 component symmetric to 9 component full tensor
    components = [
        [0, 3, 4],
        [3, 1, 5],
        [4, 5, 2]
    ]
    traction = np.array([
        np.sum([
            stress[:,:,components[i][j]] * normals[:,j,np.newaxis]
            for j in range(3)
        ], axis = 0) for i in range(3)
    ])
    traction = np.swapaxes(traction, 0, 1)
    rows_cols = int(np.sqrt(traction.size))
    traction = traction.reshape((rows_cols, rows_cols))
    return traction

def tde_matrix(model):
    stress = tde_stress_matrix(model)
    return stress_to_traction(model.tri_normals, stress)

def get_tde_slip_to_traction(tde_matrix, qd_cfg):
    def slip_to_traction(slip):
        out = tde_matrix.dot(slip)
        if qd_cfg.get('only_x', False):
            out.reshape((-1,3))[:,1] = 0.0
            out.reshape((-1,3))[:,2] = 0.0
        return out
    return slip_to_traction

def get_tde_traction_to_slip_iterative(tde_matrix):
    solver_tol = 1e-7
    def traction_to_slip(traction):
        def f(x):
            # print('residual: ' + str(x))
            # print(f.iter)
            f.iter += 1
        f.iter = 0
        return scipy.sparse.linalg.gmres(
            tde_matrix, traction, tol = solver_tol,
            restart = 500, callback = f
        )[0]
    return traction_to_slip

def get_tde_traction_to_slip_direct(tde_matrix):
    print('inverting tde matrix!')
    inverse_tde_matrix = np.linalg.inv(tde_matrix)
    def traction_to_slip(traction):
        return inverse_tde_matrix.dot(traction)
    return traction_to_slip

def get_tde_traction_to_slip(tde_matrix):
    return get_tde_traction_to_slip_direct(tde_matrix)
