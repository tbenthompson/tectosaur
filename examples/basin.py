import numpy as np
import matplotlib.pyplot as plt
import okada_wrapper

import tectosaur.mesh as mesh
import tectosaur.constraints as constraints
import tectosaur.sparse_integral_op as sparse_integral_op
import solve

#TODO:
# -- check the normal vectors for basin-top and basin (basin-inner and basin-outer should be two different meshes)
# -- run a model with zero traction on basin + surface including the adjoint traction term and constraints

def build_meshes():
    n_flt = 10
    n_surf = 81
    bw = 1
    w = 6
    dx = 2.0 * w / n_surf

    def surf_rect(ca, cb):
        nx = int((cb[0] - ca[0]) / dx)
        ny = int((cb[1] - ca[1]) / dx)
        print(nx,ny)
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
    surf = mesh.concat_list([s00, s01, s02, s10, s12, s20, s21, s22])

    n_basin = int(2.0 / dx)
    cs = [[-1,-1,-1],[-1,-1,1],[-1,1,-1],[-1,1,1],[1,-1,-1],[1,-1,1],[1,1,-1],[1,1,1]]
    cs = [[bw * c[0],bw * c[1],bw * c[2] - 1] for c in cs]
    bottom = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[2], cs[6], cs[4]])
    top = mesh.rect_surface(n_basin, n_basin, [cs[1], cs[3], cs[7], cs[5]])
    back = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[4], cs[5], cs[1]])
    front = mesh.rect_surface(n_basin, n_basin, [cs[7], cs[3], cs[2], cs[6]])
    left = mesh.rect_surface(n_basin, n_basin, [cs[0], cs[1], cs[3], cs[2]])
    right = mesh.rect_surface(n_basin, n_basin, [cs[4], cs[6], cs[7], cs[5]])
    basin = mesh.concat_list([bottom, back, front, left, right])
    basin_top = top

    fault = mesh.rect_surface(n_flt, n_flt, [[2,-1,-0.5], [3,-1,-1.5],[3,1,-1.5],[2,1,-0.5]])
    # L = 1.0
    # top_depth = -0.5
    # fault = mesh.rect_surface(n_flt, n_flt, [
    #     [-L, 0, top_depth], [-L, 0, top_depth - 1],
    #     [L, 0, top_depth - 1], [L, 0, top_depth]
    # ])
    # surf = mesh.rect_surface(n_surf, n_surf, [[-w,-w,0], [-w,w,0],[w,w,0],[w,-w,0]])

    return basin, basin_top, fault, surf

def country_constraints(surface_tris, fault_tris, pts):
    cs = constraints.continuity_constraints(surface_tris, fault_tris, pts)
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]
    slip = [1 / np.sqrt(2.0), 0, -1 / np.sqrt(2.0)]
    # slip = [1, 0, 0]
    cs.extend(constraints.constant_bc_constraints(
        n_surf_tris, n_surf_tris + n_fault_tris, slip
    ))
    cs = sorted(cs, key = lambda x: x[0][0][1])
    return cs

def main():
    basin, basin_top, fault, surface = build_meshes()
    # mesh.plot_mesh3d(*basin)
    mesh.plot_mesh3d(*mesh.concat_list([fault, surface, basin, basin_top]))

    all_country_mesh = mesh.concat_list([surface, fault])
    import ipdb; ipdb.set_trace()
    surface_tris_subset = all_country_mesh[1][:surface[1].shape[0]]
    fault_tris_subset = all_country_mesh[1][surface[1].shape[0]:]

    pr = 0.25
    sm_basin = 3e9
    sm_country = 1.0

    surface_pt_idxs = np.unique(surface_tris_subset)
    obs_pts = all_country_mesh[0][surface_pt_idxs,:]

    cs = country_constraints(surface_tris_subset, fault_tris_subset, all_country_mesh[0])
    eps = [0.08, 0.04, 0.02, 0.01]
    iop = sparse_integral_op.SparseIntegralOp(
        eps, 18, 16, 6, 3, 6, 4.0,
        'H', sm_country, pr, all_country_mesh[0], all_country_mesh[1]
    )
    soln = solve.iterative_solve(iop, cs)
    disp = soln[:iop.shape[0]].reshape(
        (int(iop.shape[0] / 9), 3, 3)
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
