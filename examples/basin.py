import numpy as np
import matplotlib.pyplot as plt
import okada_wrapper

import tectosaur.mesh as mesh
import tectosaur.constraints as constraints
import tectosaur.mass_op as mass_op
import tectosaur.dense_integral_op as dense_integral_op
import tectosaur.geometry as geometry
import solve

#TODO:
# include the mass operators
# understand what i need to put in the zero sections of the matrix
# fix the singular matrix

def build_meshes():
    n_flt = 8
    n_surf = 30
    bw = 1.0
    w = 6
    dx = 2.0 * w / n_surf

    def surf_rect(ca, cb):
        nx = int((cb[0] - ca[0]) / dx)
        ny = int((cb[1] - ca[1]) / dx)
        return mesh.rect_surface(
            ny, nx,
            [[ca[0], ca[1], 0], [ca[0], cb[1], 0], [cb[0], cb[1], 0], [cb[0], ca[1], 0]]
        )
    s00 = surf_rect([-w, -w], [-bw, -bw])
    s01 = surf_rect([-bw, -w], [bw, -bw])
    s02 = surf_rect([bw, -w], [w, -bw])
    s10 = surf_rect([-w, -bw], [-bw, bw])
    s12 = surf_rect([bw, -bw], [w, bw])
    s20 = surf_rect([-w, bw], [-bw, w])
    s21 = surf_rect([-bw, bw], [bw, w])
    s22 = surf_rect([bw, bw], [w, w])
    surf = mesh.flip_normals(mesh.concat_list([s00, s01, s02, s10, s12, s20, s21, s22]))

    n_basin = int(2 * bw / dx)
    cs = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]
    cs = [[bw * c[0],bw * c[1],bw * (c[2] - 1)] for c in cs]
    bottom = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[2], cs[6], cs[4]])
    top = mesh.rect_surface(n_basin, n_basin, [cs[1], cs[5], cs[7], cs[3]])
    back = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[4], cs[5], cs[1]])
    front = mesh.rect_surface(n_basin, n_basin, [cs[7], cs[6], cs[2], cs[3]])
    left = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[1], cs[3], cs[2]])
    right = mesh.rect_surface(n_basin, n_basin, [cs[4], cs[6], cs[7], cs[5]])
    basin = mesh.concat_list([bottom, back, front, left, right])
    basin_top = top

    fault = mesh.rect_surface(n_flt, n_flt, [[1.5,-1,-0.5], [2.5,-1,-1.5],[2.5,1,-1.5],[1.5,1,-0.5]])

    return basin, basin_top, mesh.flip_normals(basin), fault, surf

def check_basin_normals(m, inner):
    for i in range(m[1].shape[0]):
        tri = m[0][m[1][i,:]]
        n = geometry.tri_normal(tri, normalize = True)
        center = np.mean(tri, axis = 0)
        center[2] += 0.5
        v = center + 0.01 * n
        dist0 = np.linalg.norm(center)
        dist1 = np.linalg.norm(v)
        if dist1 > dist0:
            assert(not inner)
        else:
            assert(inner)

def main():
    inner_basin, basin_top, outer_basin, fault, surface = build_meshes()
    all_basin_mesh = mesh.concat(inner_basin, basin_top)
    all_country_mesh = mesh.concat_list([surface, outer_basin, fault])

    check_basin_normals(all_basin_mesh, True)
    check_basin_normals(outer_basin, False)

    # mesh.plot_mesh3d(*mesh.concat_list([fault, surface, basin, basin_top]))
    # mesh.plot_mesh3d(*all_country_mesh)

    n_all_country = all_country_mesh[1].shape[0]
    n_surf = surface[1].shape[0]
    n_outer_basin = outer_basin[1].shape[0]
    n_fault = fault[1].shape[0]
    n_all_basin = all_basin_mesh[1].shape[0]
    n_inner_basin = inner_basin[1].shape[0]
    n_basin_top = basin_top[1].shape[0]

    surface_tri_idxs = np.arange(n_surf)
    outer_basin_tri_idxs = n_surf + np.arange(n_outer_basin)
    fault_tri_idxs = n_surf + n_outer_basin + np.arange(n_fault)
    inner_basin_tri_idxs = np.arange(n_inner_basin)
    basin_top_tri_idxs = n_inner_basin + np.arange(n_basin_top)

    pr = 0.25
    sm_basin = 1.0
    sm_country = 1.0 #TODO: make different!
    slip = [1 / np.sqrt(2.0), 0, -1 / np.sqrt(2.0)]

    eps = [0.08, 0.04, 0.02, 0.01]

    nq = 17
    # build country operators
    country_op_H = dense_integral_op.DenseIntegralOp(
        eps, nq, nq, 6, 3, 6, 4.0,
        'H', sm_country, pr, all_country_mesh[0], all_country_mesh[1]
    )
    country_op_A = dense_integral_op.DenseIntegralOp(
        eps, nq, nq, 6, 3, 6, 4.0,
        'A', sm_country, pr, all_country_mesh[0], all_country_mesh[1]
    )
    country_selfop = mass_op.MassOp(3, all_country_mesh[0], all_country_mesh[1])

    # build basin operators,
    basin_op_H = dense_integral_op.DenseIntegralOp(
        eps, nq, nq, 6, 3, 6, 4.0,
        'H', sm_basin, pr, all_basin_mesh[0], all_basin_mesh[1]
    )
    basin_op_A = dense_integral_op.DenseIntegralOp(
        eps, nq, nq, 6, 3, 6, 4.0,
        'A', sm_basin, pr, all_basin_mesh[0], all_basin_mesh[1]
    )
    basin_selfop = mass_op.MassOp(3, all_basin_mesh[0], all_basin_mesh[1])

    # dof ordering: # u_c, t_c, u_b, t_b

    cs = []
    # add fault slip constraints
    # add fault traction = 0 constraints
    for i in fault_tri_idxs:
        for b in range(3):
            for d in range(3):
                dof_u = i * 9 + b * 3 + d
                dof_t = n_all_country * 9 + dof_u
                cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_u)], slip[d]))
                cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_t)], 0.0))

    # add surface traction = 0 constraints for both surface and basin-top
    for i in surface_tri_idxs:
        for b in range(3):
            for d in range(3):
                dof_t = n_all_country * 9 + i * 9 + b * 3 + d
                cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_t)], 0.0))
    for i in basin_top_tri_idxs:
        for b in range(3):
            for d in range(3):
                dof_t = n_all_country * 18 + n_all_basin * 9 + i * 9 + b * 3 + d
                cs.append(constraints.ConstraintEQ([constraints.Term(1.0, dof_t)], 0.0))

    # add displacement continuity constraints
    full_surface = mesh.concat(all_country_mesh, all_basin_mesh)
    cont_cs = constraints.continuity_constraints(full_surface[1], np.array([]), full_surface[0])
    def out_disp_dof(in_dof):
        if in_dof < n_all_country * 9:
            return in_dof
        else:
            return in_dof + n_all_country * 9
    for c in cont_cs:
        dof1 = out_disp_dof(c.terms[0].dof)
        dof2 = out_disp_dof(c.terms[1].dof)
        cs.append(constraints.ConstraintEQ(
            [constraints.Term(1.0, dof1), constraints.Term(-1.0, dof2)], 0.0
        ))


    # add basin-country continuity constraints
    for i in range(n_inner_basin):
        country_tri = n_surf + i
        basin_tri = i
        b_country = [0, 2, 1]
        for b in range(3):
            country_pts = all_country_mesh[0][all_country_mesh[1][country_tri,:]]
            basin_pts = all_basin_mesh[0][all_basin_mesh[1][basin_tri,:]]
            np.testing.assert_almost_equal(country_pts, basin_pts[b_country,:])
            for d in range(3):
                country_dof_u = country_tri * 9 + b_country[b] * 3 + d
                country_dof_t = n_all_country * 9 + country_tri * 9 + b_country[b] * 3 + d
                basin_dof_u = n_all_country * 18 + basin_tri * 9 + b * 3 + d
                basin_dof_t = n_all_country * 18 + n_all_basin * 9 + basin_tri * 9 + b * 3 + d
                # displacement continuity already imposed!
                # cs.append(constraints.ConstraintEQ(
                #     [constraints.Term(1.0, country_dof_u), constraints.Term(-1.0, basin_dof_u)],
                #     0.0
                # ))
                cs.append(constraints.ConstraintEQ(
                    [constraints.Term(1.0, country_dof_t), constraints.Term(1.0, basin_dof_t)],
                    0.0
                ))

    # build constraint matrix and condense the matrices
    n_rows = n_all_country * 18 + n_all_basin * 18

    cm, c_rhs = constraints.build_constraint_matrix(cs, n_rows)
    cm = cm.tocsr()
    cmT = cm.T

    op = np.zeros((n_rows, n_rows))
    op[
        :(n_all_country * 9),:(n_all_country * 9)
    ] = country_op_H.mat
    op[
        :(n_all_country * 9),(n_all_country * 9):(n_all_country * 18)
    ] = -country_op_A.mat + country_selfop.mat
    op[
        (n_all_country*9):(n_all_country * 18),:(n_all_country * 9)
    ] = country_op_H.mat
    op[
        (n_all_country*9):(n_all_country * 18),(n_all_country * 9):(n_all_country * 18)
    ] = -country_op_A.mat + country_selfop.mat

    op[
        (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9),
        (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9)
    ] = basin_op_H.mat
    op[
        (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9),
        (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18)
    ] = -basin_op_A.mat + basin_selfop.mat
    op[
        (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18),
        (n_all_country * 18):(n_all_country * 18 + n_all_basin * 9)
    ] = basin_op_H.mat
    op[
        (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18),
        (n_all_country * 18 + n_all_basin * 9):(n_all_country * 18 + n_all_basin * 18)
    ] = -basin_op_A.mat + basin_selfop.mat

    # plt.imshow(np.log10(np.abs(op)), interpolation = 'none')
    # plt.colorbar()
    # plt.show()

    op_constrained = cmT.dot(cmT.dot(op.T).T)

    # plt.imshow(np.log10(np.abs(op_constrained)), interpolation = 'none')
    # plt.colorbar()
    # plt.show()

    rhs_constrained = cmT.dot(-op.dot(c_rhs))
    # solve
    soln_constrained = np.linalg.solve(op_constrained, rhs_constrained)
    soln = cm.dot(soln_constrained)

    # extract the fields of interest
    country_disp_start = 0
    country_trac_start = n_all_country * 9
    basin_disp_start = n_all_country * 18
    basin_trac_start = n_all_country * 18 + n_all_basin * 9
    country_disp = soln[:country_trac_start]
    country_trac = soln[country_trac_start:basin_disp_start]
    basin_disp = soln[basin_disp_start:basin_trac_start]
    basin_trac = soln[basin_trac_start:]

    surface_disp = []
    pts = []
    tris = []
    for i in np.arange(surface[1].shape[0]):
        for b in range(3):
            start_dof = i * 9 + b * 3
            end_dof = start_dof + 3
            surface_disp.append(country_disp[start_dof:end_dof])
            pts.append(surface[0][surface[1][i,b]])
        tris.append([len(pts) - 3, len(pts) - 2, len(pts) - 1])

    for i in np.arange(basin_top[1].shape[0]):
        for b in range(3):
            start_dof = n_inner_basin * 9 + i * 9 + b * 3
            end_dof = start_dof + 3
            surface_disp.append(basin_disp[start_dof:end_dof])
            pts.append(basin_top[0][basin_top[1][i,b]])
        tris.append([len(pts) - 3, len(pts) - 2, len(pts) - 1])

    surface_disp = np.array(surface_disp)
    pts = np.array(pts)
    tris = np.array(tris)

    for d in range(3):
        vmax = np.max(surface_disp[:,d])
        plt.figure()
        plt.tripcolor(
            pts[:,0], pts[:, 1], tris,
            surface_disp[:,d], #shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()

    okada_vals = okada_exact(pts, sm_country, pr)

    for d in range(3):
        vmax = np.max(surface_disp[:,d])
        plt.figure()
        plt.tripcolor(
            pts[:,0], pts[:, 1], tris,
            okada_vals[:,d], #shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()
    plt.show()

def okada_exact(obs_pts, sm, pr):
    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)

    n_pts = obs_pts.shape[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = obs_pts[i, :]
        pt[0] -= 2.0
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, 0.5, 45.0,
            [-1.0, 1.0], [-1.0, 0.0], [0.0, 1.0, 0.0]
        )
        if suc != 0:
            u[i, :] = 0
        else:
            u[i, :] = uv
    return u

if __name__ == '__main__':
    main()
