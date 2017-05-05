import numpy as np
import matplotlib.pyplot as plt
import okada_wrapper
import itertools

import tectosaur.mesh as mesh
import tectosaur.constraints as constraints
import tectosaur.mass_op as mass_op
from tectosaur.sparse_integral_op import SparseIntegralOp
import tectosaur.geometry as geometry
import solve

def make_free_surface(w, n):
    corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
    return mesh.make_rect(n, n, corners)

def make_fault(L, top_depth, n):
    return mesh.make_rect(n, n, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])

class CombinedMesh:
    def __init__(self, named_pieces):
        self.names = [np[0] for np in named_pieces]
        pieces = [np[1] for np in named_pieces]
        self.pts, self.tris = mesh.concat(*pieces)
        self.sizes = [p[1].shape[0] for p in pieces]
        bounds = [0] + list(itertools.accumulate(self.sizes))
        self.start = bounds[:-1]
        self.past_end = bounds[1:]

    def n_total_tris(self):
        return self.past_end[-1]

    def n_dofs(self):
        return self.n_total_tris() * 9

    def get_name_idx(self, name):
        return self.names.index(name)

    def get_piece_tris(self, name):
        idx = self.get_name_idx(name)
        return self.tris[self.start[idx]:self.past_end[idx]]

    def get_piece_pt_idxs(self, name):
        return np.unique(self.get_piece_tris(name))

    def get_start(self, name):
        return self.start[self.get_name_idx(name)]

    def get_past_end(self, name):
        return self.past_end[self.get_name_idx(name)]

    def extract_pts_vals(self, name, soln):
        idx = self.get_name_idx(name)
        dof_vals = soln.reshape((-1, 3, 3))
        all_pt_vals = np.empty_like(self.pts)
        all_pt_vals[self.get_piece_tris(name)] = dof_vals[self.start[idx]:self.past_end[idx]]

        pt_idxs = self.get_piece_pt_idxs(name)
        piece_pts = self.pts[pt_idxs]
        piece_pt_vals = all_pt_vals[pt_idxs]
        return piece_pts, piece_pt_vals

def build_meshes():
    fault_L = 1.0
    fault_top_depth = -0.5

    w = 6

    basin_center = [0.0, 2.0, -2.1]
    basin_r = 2.0

    # n_flt = 8
    # n_surf = 50
    # basin_refine = 3
    # n_flt = 8
    # n_surf = 30
    # basin_refine = 2
    n_flt = 4
    n_surf = 10
    basin_refine = 1

    surf = make_free_surface(w, n_surf)
    fault = make_fault(fault_L, fault_top_depth, n_flt)
    basin = mesh.make_sphere(basin_center, basin_r, basin_refine)
    # basin = mesh.refine_to_size(mesh.make_ellipse(basin_center, 6.0, 1.0, 1.0), 0.5)

    country_mesh = CombinedMesh([('surf', surf), ('fault', fault), ('basin', mesh.flip_normals(basin))])
    basin_mesh = CombinedMesh([('basin', mesh.flip_normals((country_mesh.pts, country_mesh.get_piece_tris('basin'))))])
    return country_mesh, basin_mesh

def plot_surf_disp(country_mesh, soln):
    obs_pts, vals = country_mesh.extract_pts_vals('surf', soln)

    vmax = np.max(vals)
    for d in range(3):
        plt.figure()
        plt.tripcolor(
            obs_pts[:,0], obs_pts[:, 1], country_mesh.get_piece_tris('surf'),
            vals[:,d], #shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title('u ' + ['x', 'y', 'z'][d])
        plt.colorbar()
    plt.show()

class CompositeOp:
    def __init__(self, *ops_and_starts):
        self.ops = [el[0] for el in ops_and_starts]
        self.row_start = [el[1] for el in ops_and_starts]
        self.col_start = [el[2] for el in ops_and_starts]
        n_rows = max([el[1] + el[0].shape[0] for el in ops_and_starts])
        n_cols = max([el[2] + el[0].shape[1] for el in ops_and_starts])
        self.shape = (n_rows, n_cols)

    def generic_dot(self, v, dot_name):
        out = np.zeros(self.shape[0])
        for i in range(len(self.ops)):
            op = self.ops[i]
            start_row_idx = self.row_start[i]
            end_row_idx = start_row_idx + op.shape[0]
            start_col_idx = self.col_start[i]
            end_col_idx = start_col_idx + op.shape[1]
            input_v = v[start_col_idx:end_col_idx]
            out[start_row_idx:end_row_idx] += getattr(op, dot_name)(input_v)
        return out

    def nearfield_dot(self, v):
        return self.generic_dot(v, "nearfield_dot")

    def nearfield_no_correction_dot(self, v):
        return self.generic_dot(v, "nearfield_no_correction_dot")

    def dot(self, v):
        return self.generic_dot(v, "dot")

    def farfield_dot(self, v):
        return self.generic_dot(v, "farfield_dot")


def main():
    sm = 1.0
    pr = 0.25
    basin_sm = 0.1

    country_mesh, basin_mesh = build_meshes()
    # mesh.plot_mesh3d(country_mesh.pts, country_mesh.tris)

    csU = constraints.continuity_constraints(country_mesh.tris, np.array([]), country_mesh.pts)
    csU.extend(constraints.constant_bc_constraints(
        country_mesh.get_start('fault'), country_mesh.get_past_end('fault'), [1.0, 0.0, 0.0]
    ))
    csU.extend(constraints.free_edge_constraints(country_mesh.get_piece_tris('surf')))
    csT = constraints.constant_bc_constraints(
        0, country_mesh.n_total_tris(), [0.0, 0.0, 0.0]
    )
    cs = constraints.build_composite_constraints((csU, 0), (csT, country_mesh.n_dofs()))


    Hop = SparseIntegralOp(
        [], 0, 0, 6, 3, 6, 4.0, 'H', sm, pr, country_mesh.pts, country_mesh.tris,
        use_tables = True, remove_sing = True
    )
    Aop = SparseIntegralOp(
        [], 0, 0, 6, 3, 6, 4.0, 'A', sm, pr, country_mesh.pts, country_mesh.tris,
        use_tables = True, remove_sing = False
    )
    op = CompositeOp(
        (Hop, 0, 0), (Aop, 0, country_mesh.n_dofs())
    )

    soln = solve.iterative_solve(op, cs)
    plot_surf_disp(country_mesh, soln)

    # soln = solve.iterative_solve(Hop, csU)
    # plot_surf_disp(country_mesh, soln)


if __name__ == '__main__':
    main()

#
#     inner_basin, basin_top, outer_basin, fault, surface = build_meshes()
#     all_basin_mesh = mesh.concat(inner_basin, basin_top)
#     all_country_mesh = mesh.concat_list([surface, outer_basin, fault])
#
#     check_basin_normals(all_basin_mesh, True)
#     check_basin_normals(outer_basin, False)
#
#     # mesh.plot_mesh3d(*mesh.concat_list([fault, surface, basin, basin_top]))
#     # mesh.plot_mesh3d(*all_country_mesh)
#
#     n_all_country = all_country_mesh[1].shape[0]
#     n_surf = surface[1].shape[0]
#     n_outer_basin = outer_basin[1].shape[0]
#     n_fault = fault[1].shape[0]
#     n_all_basin = all_basin_mesh[1].shape[0]
#     n_inner_basin = inner_basin[1].shape[0]
#     n_basin_top = basin_top[1].shape[0]
#
#     surface_tri_idxs = np.arange(n_surf)
#     outer_basin_tri_idxs = n_surf + np.arange(n_outer_basin)
#     fault_tri_idxs = n_surf + n_outer_basin + np.arange(n_fault)
#     inner_basin_tri_idxs = np.arange(n_inner_basin)
#     basin_top_tri_idxs = n_inner_basin + np.arange(n_basin_top)
#
#     pr = 0.25
#     sm_basin = 1.0
#     sm_country = 1.0 #TODO: make different!
#     slip = [1 / np.sqrt(2.0), 0, -1 / np.sqrt(2.0)]
#
#     eps = [0.08, 0.04, 0.02, 0.01]
#
#     nq = 17
#     # build country operators
#     country_op_H = dense_integral_op.DenseIntegralOp(
#         eps, nq, nq, 6, 3, 6, 4.0,
#         'H', sm_country, pr, all_country_mesh[0], all_country_mesh[1]
#     )
#     country_op_A = dense_integral_op.DenseIntegralOp(
#         eps, nq, nq, 6, 3, 6, 4.0,
#         'A', sm_country, pr, all_country_mesh[0], all_country_mesh[1]
#     )
#     country_selfop = mass_op.MassOp(3, all_country_mesh[0], all_country_mesh[1])
#
#     # build basin operators,
#     basin_op_H = dense_integral_op.DenseIntegralOp(
#         eps, nq, nq, 6, 3, 6, 4.0,
#         'H', sm_basin, pr, all_basin_mesh[0], all_basin_mesh[1]
#     )
#     basin_op_A = dense_integral_op.DenseIntegralOp(
#         eps, nq, nq, 6, 3, 6, 4.0,
#         'A', sm_basin, pr, all_basin_mesh[0], all_basin_mesh[1]
#     )
#     basin_selfop = mass_op.MassOp(3, all_basin_mesh[0], all_basin_mesh[1])
#
#     # dof ordering: # u_c, t_c, u_b, t_b
#
#     cs = []
#     # add fault slip constraints
#     # add fault traction = 0 constraints
#     for i in fault_tri_idxs:
#         for b in range(3):
#             for d in range(3):
#                 dof_u = i * 9 + b * 3 + d
#                 dof_t = n_all_country * 9 + dof_u
#                 cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_u)], slip[d]))
#                 cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_t)], 0.0))
#
#     # add surface traction = 0 constraints for both surface and basin-top
#     for i in surface_tri_idxs:
#         for b in range(3):
#             for d in range(3):
#                 dof_t = n_all_country * 9 + i * 9 + b * 3 + d
#                 cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_t)], 0.0))
#     for i in basin_top_tri_idxs:
#         for b in range(3):
#             for d in range(3):
#                 dof_t = n_all_country * 18 + n_all_basin * 9 + i * 9 + b * 3 + d
#                 cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_t)], 0.0))
#
#     # add displacement continuity constraints
#     full_surface = mesh.concat(all_country_mesh, all_basin_mesh)
#     cont_cs = constraints.continuity_constraints(full_surface[1], np.array([]), full_surface[0])
#     def out_disp_dof(in_dof):
#         if in_dof < n_all_country * 9:
#             return in_dof
#         else:
#             return in_dof + n_all_country * 9
#     for c in cont_cs:
#         dof1 = out_disp_dof(c.terms[0].dof)
#         dof2 = out_disp_dof(c.terms[1].dof)
#         cs.append(constraints.ConstraintEQ(
#             [constraints.Term(1.0, dof1), constraints.Term(-1.0, dof2)], 0.0
#         ))
#
#
#     # add basin-country continuity constraints
#     for i in range(n_inner_basin):
#         country_tri = n_surf + i
#         basin_tri = i
#         b_country = [0, 2, 1]
#         for b in range(3):
#             country_pts = all_country_mesh[0][all_country_mesh[1][country_tri,:]]
#             basin_pts = all_basin_mesh[0][all_basin_mesh[1][basin_tri,:]]
#             np.testing.assert_almost_equal(country_pts, basin_pts[b_country,:])
#             for d in range(3):
#                 country_dof_u = country_tri * 9 + b_country[b] * 3 + d
#                 country_dof_t = n_all_country * 9 + country_tri * 9 + b_country[b] * 3 + d
#                 basin_dof_u = n_all_country * 18 + basin_tri * 9 + b * 3 + d
#                 basin_dof_t = n_all_country * 18 + n_all_basin * 9 + basin_tri * 9 + b * 3 + d
#                 # displacement continuity already imposed!
#                 # cs.append(constraints.ConstraintEQ(
#                 #     [constraints.Term(1.0, country_dof_u), constraints.Term(-1.0, basin_dof_u)],
#                 #     0.0
#                 # ))
#                 cs.append(constraints.ConstraintEQ(
#                     [constraints.Term(1.0, country_dof_t), constraints.Term(1.0, basin_dof_t)],
#                     0.0
#                 ))
#
#     # build constraint matrix and condense the matrices
#     n_rows = n_all_country * 18 + n_all_basin * 18
#
#     cm, c_rhs = constraints.build_constraint_matrix(cs, n_rows)
#     cm = cm.tocsr()
#     cmT = cm.T
#
#     op = np.zeros((n_rows, n_rows))
#     op[
#         :(n_all_country * 9),:(n_all_country * 9)
#     ] = country_op_H.mat
#     op[
#         :(n_all_country * 9),(n_all_country * 9):(n_all_country * 18)
#     ] = -country_op_A.mat + country_selfop.mat
#     op[
#         (n_all_country*9):(n_all_country * 18),:(n_all_country * 9)
#     ] = country_op_H.mat
#     op[
#         (n_all_country*9):(n_all_country * 18),(n_all_country * 9):(n_all_country * 18)
#     ] = -country_op_A.mat + country_selfop.mat
#
#     op[
#         (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9),
#         (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9)
#     ] = basin_op_H.mat
#     op[
#         (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9),
#         (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18)
#     ] = -basin_op_A.mat + basin_selfop.mat
#     op[
#         (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18),
#         (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9)
#     ] = basin_op_H.mat
#     op[
#         (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18),
#         (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18)
#     ] = -basin_op_A.mat + basin_selfop.mat
#
#     # plt.imshow(np.log10(np.abs(op)), interpolation = 'none')
#     # plt.colorbar()
#     # plt.show()
#
#     op_constrained = cmT.dot(cmT.dot(op.T).T)
#
#     # plt.imshow(np.log10(np.abs(op_constrained)), interpolation = 'none')
#     # plt.colorbar()
#     # plt.show()
#
#     rhs_constrained = cmT.dot(-op.dot(c_rhs))
#     # solve
#     soln_constrained = np.linalg.solve(op_constrained, rhs_constrained)
#     soln = cm.dot(soln_constrained)
#
#     # extract the fields of interest
#     country_disp_start = 0
#     country_trac_start = n_all_country * 9
#     basin_disp_start = n_all_country * 18
#     basin_trac_start = n_all_country * 18 + n_all_basin * 9
#     country_disp = soln[:country_trac_start]
#     country_trac = soln[country_trac_start:basin_disp_start]
#     basin_disp = soln[basin_disp_start:basin_trac_start]
#     basin_trac = soln[basin_trac_start:]
#
#     surface_disp = []
#     pts = []
#     tris = []
#     for i in np.arange(surface[1].shape[0]):
#         for b in range(3):
#             start_dof = i * 9 + b * 3
#             end_dof = start_dof + 3
#             surface_disp.append(country_disp[start_dof:end_dof])
#             pts.append(surface[0][surface[1][i,b]])
#         tris.append([len(pts) - 3, len(pts) - 2, len(pts) - 1])
#
#     for i in np.arange(basin_top[1].shape[0]):
#         for b in range(3):
#             start_dof = n_inner_basin * 9 + i * 9 + b * 3
#             end_dof = start_dof + 3
#             surface_disp.append(basin_disp[start_dof:end_dof])
#             pts.append(basin_top[0][basin_top[1][i,b]])
#         tris.append([len(pts) - 3, len(pts) - 2, len(pts) - 1])
#
#     surface_disp = np.array(surface_disp)
#     pts = np.array(pts)
#     tris = np.array(tris)
#
#     for d in range(3):
#         vmax = np.max(surface_disp[:,d])
#         plt.figure()
#         plt.tripcolor(
#             pts[:,0], pts[:, 1], tris,
#             surface_disp[:,d], #shading='gouraud',
#             cmap = 'PuOr', vmin = -vmax, vmax = vmax
#         )
#         plt.title('u ' + ['x', 'y', 'z'][d])
#         plt.colorbar()
#
#     okada_vals = okada_exact(pts, sm_country, pr)
#
#     for d in range(3):
#         vmax = np.max(surface_disp[:,d])
#         plt.figure()
#         plt.tripcolor(
#             pts[:,0], pts[:, 1], tris,
#             okada_vals[:,d], #shading='gouraud',
#             cmap = 'PuOr', vmin = -vmax, vmax = vmax
#         )
#         plt.title('u ' + ['x', 'y', 'z'][d])
#         plt.colorbar()
#     plt.show()
#
# def okada_exact(obs_pts, sm, pr):
#     lam = 2 * sm * pr / (1 - 2 * pr)
#     alpha = (lam + sm) / (lam + 2 * sm)
#
#     n_pts = obs_pts.shape[0]
#     u = np.empty((n_pts, 3))
#     for i in range(n_pts):
#         pt = obs_pts[i, :]
#         pt[0] -= 2.0
#         [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
#             alpha, pt, 0.5, 45.0,
#             [-1.0, 1.0], [-1.0, 0.0], [0.0, 1.0, 0.0]
#         )
#         if suc != 0:
#             u[i, :] = 0
#         else:
#             u[i, :] = uv
#     return u
