import numpy as np
import matplotlib.pyplot as plt
import okada_wrapper

import tectosaur.mesh as mesh
import tectosaur.constraints as constraints
import tectosaur.mass_op as mass_op
from tectosaur.sparse_integral_op import SparseIntegralOp
import tectosaur.geometry as geometry
from tectosaur.mass_op import MassOp
from tectosaur.composite_op import CompositeOp
from tectosaur.combined_mesh import CombinedMesh
import solve

def make_free_surface(w, n):
    corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
    return mesh.make_rect(n, n, corners)

def make_fault(L, top_depth, n):
    return mesh.make_rect(n, n, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])


def build_meshes():
    fault_L = 1.0
    fault_top_depth = -0.5

    w = 6

    basin_center = [0.0, 2.0, -2.1]
    basin_r = 2.0

    # n_flt = 8
    # n_surf = 50
    # basin_refine = 3
    n_flt = 8
    n_surf = 30
    basin_refine = 3
    # n_flt = 4
    # n_surf = 10
    # basin_refine = 1

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

def couple_domains(mesh1, mesh2):
    np.testing.assert_almost_equal(mesh1.pts, mesh2.pts)
    all_tris = np.vstack((mesh1.tris, mesh2.tris))
    initial_cs_u = constraints.continuity_constraints(all_tris, np.array([]), mesh1.pts)
    # TODO: Refactor to have a shared find-touching-triangle-vertex-pairs
    # function with continuity_constraints

    cs_u = []
    cs_t = []
    for c in initial_cs_u:
        dof1, dof2 = c.terms[0].dof, c.terms[1].dof
        if dof1 > mesh1.n_dofs():
            dof1 += mesh1.n_dofs()
        cs_u.append(constraints.ConstraintEQ([
                constraints.Term(1.0, dof1),
                constraints.Term(-1.0, dof2)
            ], 0.0
        ))

        second_dof_factor = 1.0
        if (dof1 < mesh1.n_dofs()) == (dof2 < mesh1.n_dofs()):
            second_dof_factor = -1.0

        if dof1 < mesh1.n_dofs():
            dof1 += mesh1.n_dofs()
        else:
            dof1 += mesh2.n_dofs()

        if dof2 < mesh1.n_dofs():
            dof2 += mesh1.n_dofs()
        else:
            dof2 += mesh2.n_dofs()

        cs_t.append(constraints.ConstraintEQ([
                constraints.Term(1.0, dof1),
                constraints.Term(second_dof_factor, dof2)
            ], c.rhs
        ))

    return cs_u + cs_t

def main():
    sm = 1.0
    pr = 0.25
    basin_sm = 1.0

    country_mesh, basin_mesh = build_meshes()
    n_country_dofs = country_mesh.n_dofs() * 2
    n_basin_dofs = basin_mesh.n_dofs() * 2


    country_csU = constraints.continuity_constraints(
        country_mesh.tris, np.array([]), country_mesh.pts
    )
    country_csU.extend(constraints.constant_bc_constraints(
        country_mesh.get_start('fault'), country_mesh.get_past_end('fault'), [1.0, 0.0, 0.0]
    ))
    country_csU.extend(constraints.free_edge_constraints(country_mesh.get_piece_tris('surf')))
    country_csT = constraints.constant_bc_constraints(
        country_mesh.get_start('surf'), country_mesh.get_past_end('surf'), [0.0, 0.0, 0.0]
    )
    country_csT.extend(constraints.constant_bc_constraints(
        country_mesh.get_start('fault'), country_mesh.get_past_end('fault'), [0.0, 0.0, 0.0]
    ))
    country_cs = constraints.build_composite_constraints(
        (country_csU, 0), (country_csT, country_mesh.n_dofs())
    )


    basin_csU = constraints.continuity_constraints(
        basin_mesh.tris, np.array([]), basin_mesh.pts
    )
    # basin_csT = constraints.constant_bc_constraints(
    #     0, basin_mesh.n_total_tris(), [0.0, 0.0, 0.0],
    # )
    basin_csT = []
    basin_cs = constraints.build_composite_constraints(
        (basin_csU, 0), (basin_csT, basin_mesh.n_dofs())
    )

    cs = constraints.build_composite_constraints(
        (country_cs, 0), (basin_cs, n_country_dofs)
    )
    cs.extend(couple_domains(country_mesh, basin_mesh))


    Hop = SparseIntegralOp(
        [], 0, 0, 6, 3, 6, 4.0, 'H', sm, pr, country_mesh.pts, country_mesh.tris,
        use_tables = True, remove_sing = True
    )
    Aop = SparseIntegralOp(
        [], 0, 0, 6, 3, 6, 4.0, 'A', sm, pr, country_mesh.pts, country_mesh.tris,
        use_tables = True, remove_sing = False
    )
    country_mass = MassOp(3, country_mesh.pts, country_mesh.tris)
    country_op = CompositeOp(
        (Hop, 0, 0),
        (Aop, 0, country_mesh.n_dofs()),
        (country_mass, 0, country_mesh.n_dofs()),
        shape = (n_country_dofs, n_country_dofs)
    )

    Uop = SparseIntegralOp(
        [], 0, 0, 6, 3, 6, 4.0, 'U', basin_sm, pr, basin_mesh.pts, basin_mesh.tris,
        use_tables = True, remove_sing = False
    )
    Top = SparseIntegralOp(
        [], 0, 0, 6, 3, 6, 4.0, 'T', basin_sm, pr, basin_mesh.pts, basin_mesh.tris,
        use_tables = True, remove_sing = False
    )
    basin_mass = MassOp(3, basin_mesh.pts, basin_mesh.tris)
    basin_op = CompositeOp(
        (Top, 0, 0),
        (basin_mass, 0, 0),
        (Uop, 0, basin_mesh.n_dofs()),
        shape = (n_basin_dofs, n_basin_dofs)
    )

    op = CompositeOp(
        (country_op, 0, 0),
        (basin_op, n_country_dofs, n_country_dofs)
    )

    soln = solve.iterative_solve(op, cs)
    plot_surf_disp(country_mesh, soln)

    # soln = solve.iterative_solve(Hop, csU)
    # plot_surf_disp(country_mesh, soln)


if __name__ == '__main__':
    main()
