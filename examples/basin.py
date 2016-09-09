import numpy as np
import matplotlib.pyplot as plt
import okada_wrapper

import tectosaur.mesh as mesh
import tectosaur.constraints as constraints
import tectosaur.sparse_integral_op as sparse_integral_op
import tectosaur.geometry as geometry
import solve

#TODO:
# -- use dense operators to make the situation easier for the moment.
# -- assemble operators
# -- continuity constraints for basin
# -- continuity constraints between basin top and surrounding surface
# -- trac = 0 constraints for all surface
# -- run a model with zero traction on basin + surface including the adjoint traction term and constraints
# -- the final basin example!

def build_meshes():
    n_flt = 10
    n_surf = 31
    bw = 1
    w = 6
    dx = 2.0 * w / n_surf

    def surf_rect(ca, cb):
        nx = int((cb[0] - ca[0]) / dx)
        ny = int((cb[1] - ca[1]) / dx)
        return mesh.rect_surface(
            ny, nx,
            [[ca[0], ca[1], 0], [ca[0], cb[1], 0], [cb[0], cb[1], 0], [cb[0], ca[1], 0]]
        )
    s00 = surf_rect([-w, -w], [-1, -1])
    s01 = surf_rect([-1, -w], [1, -1])
    s02 = surf_rect([1, -w], [w, -1])
    s10 = surf_rect([-w, -1], [-1, 1])
    s12 = surf_rect([1, -1], [w, 1])
    s20 = surf_rect([-w, 1], [-1, w])
    s21 = surf_rect([-1, 1], [1, w])
    s22 = surf_rect([1, 1], [w, w])
    surf = mesh.flip_normals(mesh.concat_list([s00, s01, s02, s10, s12, s20, s21, s22]))

    n_basin = int(2.0 / dx)
    cs = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]
    cs = [[bw * c[0],bw * c[1],bw * c[2] - 1] for c in cs]
    bottom = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[2], cs[6], cs[4]])
    top = mesh.rect_surface(n_basin, n_basin, [cs[1], cs[5], cs[7], cs[3]])
    back = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[4], cs[5], cs[1]])
    front = mesh.rect_surface(n_basin, n_basin, [cs[7], cs[6], cs[2], cs[3]])
    left = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[1], cs[3], cs[2]])
    right = mesh.rect_surface(n_basin, n_basin, [cs[4], cs[6], cs[7], cs[5]])
    basin = mesh.concat_list([bottom, back, front, left, right])
    basin_top = top

    fault = mesh.rect_surface(n_flt, n_flt, [[2,-1,-0.5], [3,-1,-1.5],[3,1,-1.5],[2,1,-0.5]])

    return basin, basin_top, mesh.flip_normals(basin), fault, surf

def country_constraints(surface_tris, fault_tris, pts):
    cs = constraints.continuity_constraints(surface_tris, fault_tris, pts)
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]
    slip = [1 / np.sqrt(2.0), 0, -1 / np.sqrt(2.0)]
    cs.extend(constraints.constant_bc_constraints(
        n_surf_tris, n_surf_tris + n_fault_tris, slip
    ))
    cs = sorted(cs, key = lambda x: x[0][0][1])
    return cs

def check_basin_normals(m, inner):
    for i in range(m[1].shape[0]):
        tri = m[0][m[1][i,:]]
        n = geometry.tri_normal(tri, normalize = True)
        center = np.mean(tri, axis = 0)
        center[2] += 1
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

    mesh.plot_mesh3d(*all_country_mesh)
    n_surf = surface[1].shape[0]
    n_basin = outer_basin[1].shape[0]
    n_fault = fault[1].shape[0]
    surface_tris_subset = all_country_mesh[1][:(n_surf + n_basin)]
    fault_tris_subset = all_country_mesh[1][(n_surf + n_basin):]

    pr = 0.25
    sm_basin = 3e9
    sm_country = 1.0

    surface_pt_idxs = np.unique(surface_tris_subset)
    obs_pts = all_country_mesh[0][surface_pt_idxs,:]

    cs = country_constraints(surface_tris_subset, fault_tris_subset, all_country_mesh[0])
    eps = [0.08, 0.04, 0.02, 0.01]
    country_op_H = sparse_integral_op.SparseIntegralOp(
        eps, 18, 16, 6, 3, 6, 4.0,
        'H', sm_country, pr, all_country_mesh[0], all_country_mesh[1]
    )
    country_op_A = sparse_integral_op.SparseIntegralOp(
        eps, 18, 16, 6, 3, 6, 4.0,
        'A', sm_country, pr, all_country_mesh[0], all_country_mesh[1]
    )
    soln = solve.iterative_solve(country_op_H, cs)
    disp = soln[:country_op_H.shape[0]].reshape(
        (int(country_op_H.shape[0] / 9), 3, 3)
    )[:-fault_tris_subset.shape[0]]
    vals = [None] * surface_pt_idxs.shape[0]
    for i in range(surface_tris_subset.shape[0]):
        for b in range(3):
            idx = surface_tris_subset[i, b]
            vals[idx] = disp[i,b,:]
    vals = np.array(vals)
    # okada_vals = okada_exact(obs_pts, sm_country, pr)

    for d in range(3):
        vmax = np.max(np.abs(vals[:,d]))
        plt.figure()
        plt.tripcolor(
            obs_pts[:,0], obs_pts[:, 1], surface_tris_subset,
            vals[:,d], shading='gouraud',
            cmap = 'PuOr', vmin = -vmax, vmax = vmax
        )
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()
    plt.show()

    # soln = iterative_solve(iop, cs)
    # disp = soln.reshape((int(iop.shape[0] / 9), 3, 3))

def okada_exact(obs_pts, sm, pr):
    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)
    print(lam, sm, pr, alpha)

    n_pts = obs_pts.shape[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = obs_pts[i, :]
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
