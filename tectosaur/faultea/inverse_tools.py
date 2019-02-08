import logging
import numpy as np
import scipy.spatial
import scipy.sparse.csgraph as graph
import scipy.sparse
import shapely.geometry
import matplotlib.pyplot as plt
import matplotlib.tri as tri

import cppimport.import_hook
import tectosaur.util.geometry
import tectosaur.nearfield.edge_adj_setup as edge_adj_setup
import tectosaur_topo as tt

import mesh_fncs
import slip_vectors
import collect_dem


def tri_side(tri1, tri2, threshold = 1e-12):
    tri1_normal = tectosaur.util.geometry.tri_normal(tri1, normalize = True)
    tri1_center = np.mean(tri1, axis = 0)
    tri2_center = np.mean(tri2, axis = 0)
    direction = tri2_center - tri1_center
    direction /= np.linalg.norm(direction)
    dot_val = direction.dot(tri1_normal)
    if dot_val > threshold:
        return 0
    elif dot_val < -threshold:
        return 1
    else:
        return 2

def plot_side_of_fault(m, side, view_R = 1.0):
    fC, R = get_fault_centered_view(m)

    C = np.ones(m.pts.shape[0])
    plt.figure(figsize = (10, 10))
    for i in range(3):
        which_tris = m.tris[side == i + 1]
        if which_tris.shape[0] == 0:
            continue
        plt.tripcolor(m.pts[:,0], m.pts[:,1], which_tris, C * i, vmin = 0, vmax = 3, cmap = 'hsv')
    vW = view_R * R
    plt.xlim([fC[0] - view_R * R, fC[0] + view_R * R])
    plt.ylim([fC[1] - view_R * R, fC[1] + view_R * R])
    plt.show()

def get_side_of_fault(m):
    fault_start_idx = m.get_start('fault')
    connectivity = mesh_fncs.tri_connectivity_graph(m.tris)
    fault_touching_pair = np.where(np.logical_and(
        connectivity.row < fault_start_idx,
        connectivity.col >= fault_start_idx
    ))[0]

    side = np.zeros(m.n_tris())
    shared_verts = np.zeros(m.n_tris())

    fault_surf_tris = m.pts[m.tris[connectivity.col[fault_touching_pair]]]
    for i in range(fault_touching_pair.shape[0]):
        surf_tri_idx = connectivity.row[fault_touching_pair[i]]
        surf_tri = m.tris[surf_tri_idx]
        fault_tri = m.tris[connectivity.col[fault_touching_pair[i]]]
        which_side = tri_side(m.pts[fault_tri], m.pts[surf_tri])

        n_shared_verts = 0
        for d in range(3):
            if surf_tri[d] in fault_tri:
                n_shared_verts += 1

        if shared_verts[surf_tri_idx] < 2:
            side[surf_tri_idx] = int(which_side) + 1
            shared_verts[surf_tri_idx] = n_shared_verts
    return side

def plot_fault_trace(m):
    fault_tris = m.get_tris('fault')
    for e in mesh_fncs.get_surf_fault_edges(m.get_tris('surf'), fault_tris):
        i1, d1 = e[0]
        i2, d2 = e[1]
        pts = m.pts[[fault_tris[i1,d1], fault_tris[i2,d2]]]
        plt.plot(pts[:,0], pts[:,1], 'k-', markersize = 10)

def get_fault_centered_view(m):
    fault_pts = m.get_tri_pts('fault').reshape((-1,3))
    fC = np.mean(fault_pts, axis = 0)
    R = np.sqrt(np.max(np.sum((fault_pts - fC) ** 2, axis = 1)))
    return fC, R

def plot_surf_disp(m, side, field, name, vmin = None, vmax = None, filename = None, view_R = 1.0, proj = None, latlon_step = 0.5):
    fC, R = get_fault_centered_view(m)

    if vmin is None:
        vmin = np.min(field)
    if vmax is None:
        vmax = np.max(field)
    cmap = 'PuOr_r'
    levels = np.linspace(vmin, vmax, 17)

    plt.figure(figsize = (10,10))
    fault_start_idx = m.get_start('fault')
    for i in range(2):
        which_tris = np.where(np.logical_or(side[:fault_start_idx] == 0, side[:fault_start_idx] == i + 1))[0]
        reduced_m = mesh_fncs.remove_unused_pts((m.pts, m.tris[which_tris]))
        soln_vals = np.empty(reduced_m[0].shape[0])
        soln_vals[reduced_m[1]] = field[which_tris]

        triang = tri.Triangulation(reduced_m[0][:,0], reduced_m[0][:,1], triangles = reduced_m[1])
        tri_refi, interp_vals = triang, soln_vals
        cntf = plt.tricontourf(tri_refi, interp_vals, cmap = cmap, levels = levels, extend = 'both')
        plt.tricontour(
            tri_refi, interp_vals, levels = levels,
            colors = '#333333', linestyles = 'solid', linewidths = 0.75
        )

    plot_fault_trace(m)


    cbar = plt.colorbar(cntf)
    cbar.set_label('$\\text{displacement (m)}$')
    
    map_axis(fC, R, view_R, proj, latlon_step)
    
    plt.title(name)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

def magnitude(M0):
    return (2.0 / 3.0) * np.log10(M0) - 10.7 + (2.0 / 3.0) * 7

def moment_analysis(m, inverse_soln, proj):
    slip = inverse_soln[0].reshape((-1,2))
    total_slip = np.sqrt(np.sum(slip ** 2, axis = 1))
    total_slip_m = total_slip / 100.0

    from tectosaur.ops.mass_op import MassOp
    fault_mass = MassOp(2, m.pts, m.get_tris('fault'))
    tri_potency = fault_mass.dot(np.concatenate((total_slip_m, np.zeros(total_slip.shape[0] * 2))))[:total_slip.shape[0]]
    potency = np.sum(tri_potency)
    avg_tri_potency = np.sum(tri_potency.reshape((-1,3)), axis = 1)

    mu = 30e9
    M0 = potency * mu
    M = magnitude(M0)
    print('potency: ' + str(potency))
    print('M0: ' + str(M0))
    print('M: ' + str(M))

    tri_centers = np.mean(m.get_tri_pts('fault'), axis = 1)
    tri_lonlat = collect_dem.project(tri_centers[:,0], tri_centers[:,1], tri_centers[:,2], proj, inverse = True)
    surf_elev = collect_dem.get_pt_elevations(tri_lonlat, 5)

    tri_depth = surf_elev - tri_centers[:,2]
    avg_tri_moment = avg_tri_potency * mu
    return tri_depth, avg_tri_moment

def plot_moment_analysis(tri_depth, avg_tri_moment):
    with plt.style.context('ggplot'):
        plt.figure(figsize = (7,7))
        plt.hist(
            -tri_depth / 1000.0, weights = avg_tri_moment,
            bins = 15, rwidth = 0.7,
            orientation = 'horizontal', color = 'k',
            histtype = 'stepfilled'
        )
        plt.ylabel('depth (km)', fontsize = 18, labelpad = 10)
        plt.xlabel('moment (Nm)', fontsize = 18, labelpad = 10)
        plt.gca().set_yticklabels([str(int(-d)) for d in plt.gca().get_yticks()])
        plt.savefig('depth_vs_moment.pdf')
        plt.show()

def latlon_axis(ax, xbounds, ybounds, proj, latlon_step):
    inv_latlon_step = 1.0 / latlon_step
    lon_edges = collect_dem.project(xbounds, [ybounds[0]] * 2, [0,0], proj, inverse = True)
    min_lon = int(np.floor(lon_edges[0][0]))
    max_lon = int(np.ceil(lon_edges[1][0]))
    lon = np.linspace(min_lon, max_lon, inv_latlon_step * (max_lon - min_lon) + 1)
    lon_proj = collect_dem.project(lon, [lon_edges[0][1]] * len(lon), [0] * len(lon), proj)

    ax.set_xticks(lon_proj[:,0])
    ax.set_xticklabels(['$\\mathrm{' + str(x) + '}^{\circ} ~ \mathrm{E}$' for x in lon])
    ax.set_xlabel('$\\mathrm{Longitude}$')

    lat_edges = collect_dem.project([xbounds[0]] * 2, ybounds, [0,0], proj, inverse = True)
    min_lat = int(np.floor(lat_edges[0][1]))
    max_lat = int(np.ceil(lat_edges[1][1]))
    lat = np.linspace(min_lat, max_lat, inv_latlon_step * (max_lat - min_lat) + 1)
    lat_proj = collect_dem.project([lat_edges[0][0]] * len(lat), lat, [0] * len(lat), proj)

    ax.set_yticks(lat_proj[:,1])
    ax.set_yticklabels(['$\\mathrm{' + str(y) + '}^{\circ} ~ \mathrm{N}$' for y in lat])
    ax.set_ylabel('$\\mathrm{Latitude}$')

def map_axis(fC, R, view_R, proj, latlon_step):
    xbounds = [fC[0] - view_R * R, fC[0] + view_R * R]
    ybounds = [fC[1] - view_R * R, fC[1] + view_R * R]
    if proj is not None:
        latlon_axis(plt.gca(), xbounds, ybounds, proj, latlon_step)
    plt.xlim(xbounds)
    plt.ylim(ybounds)
    
def plot_situation(m, data, proj = None, modeled = None, view_R = 2.0, filename = None, min_elevation = None, max_elevation = None, latlon_step = 0.5, figsize = (13,13)):
    fault_pts = m.get_tri_pts('fault').reshape((-1,3))
    fC = np.mean(fault_pts, axis = 0)
    R = np.sqrt(np.max(np.sum((fault_pts - fC) ** 2, axis = 1)))

    surf_pts = m.get_tri_pts('surf')
    if min_elevation is None:
        min_elevation = int(np.floor(np.min(surf_pts[:,:,2]) / 1000.0))
    if max_elevation is None:
        max_elevation = int(np.ceil(np.max(surf_pts[:,:,2]) / 1000.0))
    n_steps = (max_elevation - min_elevation) * 2 + 1
    levels = np.linspace(min_elevation, max_elevation, n_steps)

    plt.figure(figsize = figsize)
    cntf = plt.tricontourf(
        m.pts[:,0], m.pts[:,1], m.get_tris('surf'), m.pts[:,2] / 1000.0,
        levels = levels, extend = 'both'
    )
    plot_fault_trace(m)
    plt.triplot(m.pts[:,0], m.pts[:,1], m.get_tris('fault'), 'w-', linewidth = 0.4)
    plt.quiver(data['X'], data['Y'], data['EW'], data['SN'], color = 'r')
    if modeled is not None:
        plt.quiver(data['X'], data['Y'], modeled[:,0], modeled[:,1], color = 'w')
    cbar = plt.colorbar(cntf)
    cbar.set_label('elevation (km)')
    
    map_axis(fC, R, view_R, proj, latlon_step)

    if filename is not None:
        plt.savefig(filename)
    plt.show()

def find_containing_triangles(m, pts):
    #for each point, find the surface triangle that contains it

    surf_pt_idxs = m.get_pt_idxs('surf')
    surf_pts = m.pts[surf_pt_idxs]
    kd = scipy.spatial.cKDTree(surf_pts[:,:2].copy())
    n_nn = min(10, surf_pt_idxs.shape[0])
    dist, nn_idxs = kd.query(pts.copy(), n_nn)

    pts_inside = []
    for i in range(pts.shape[0]):
        o_pt = [pts[i,0], pts[i,1], 0.0]
        s_pt = shapely.geometry.Point(o_pt)
        nearby_tris = dict()
        for j in range(n_nn):
            touching_tris = np.where(m.tris == surf_pt_idxs[nn_idxs[i,j]])
            for t in touching_tris[0]:
                nearby_tris[t] = nearby_tris.get(t, 0) + 1
        found_tri = False
        for tri_idx,v in nearby_tris.items():
            if v == 3:
                tri_pts = m.pts[m.tris[tri_idx]].copy()
                tri_pts[:,2] = 0
                s_poly = shapely.geometry.Polygon(tri_pts.tolist())
                if not s_poly.contains(s_pt):
                    continue
                assert(not found_tri)
                found_tri = True
                pts_inside.append((i, tri_idx))

    pts_inside = np.array(pts_inside)
    n_outside = pts.shape[0] - pts_inside.shape[0]
    print('number of observation points outside the surface mesh: ' + str(n_outside))
    return pts_inside

def build_interp_matrix(m, obs_pts, containing_tris, which_dims):
    soln_to_obs = scipy.sparse.dok_matrix((
        obs_pts.shape[0] * len(which_dims),
        m.tris.shape[0] * 9
    ))

    # compute the interpolation matrix from triangle vertices to observation points
    for i in range(obs_pts.shape[0]):
        o_pt = [obs_pts[i,0], obs_pts[i,1], 0.0]
        tri_idx = containing_tris[i]
        tri_pts = m.pts[m.tris[tri_idx]].copy()
        tri_pts[:,2] = 0.0
        xyhat = edge_adj_setup.xyhat_from_pt(o_pt, tri_pts.tolist())
        basis_coeffs = tectosaur.util.geometry.linear_basis_tri(*xyhat)
        np.testing.assert_almost_equal(np.sum(basis_coeffs), 1.0)
        
        for b in range(3):
            for d in which_dims:
                soln_to_obs[i * len(which_dims) + d, tri_idx * 9 + b * 3 + d] = basis_coeffs[b]
    return soln_to_obs

def setup_data_matrices(m, data, which_dims):
    obs_pts = np.array([data['X'], data['Y']]).T.copy()
    obs_disp = np.array([data['EW'], data['SN']]).T.copy()
    pts_inside = find_containing_triangles(m, obs_pts)
    obs_pts_inside = obs_pts[pts_inside[:,0],:]
    soln_to_obs = build_interp_matrix(m, obs_pts_inside, pts_inside[:,1], which_dims)
    rhs = obs_disp[pts_inside[:,0],:].flatten()
    return soln_to_obs, rhs, pts_inside

def build_slip_matrix(m, get_vertical = None):
    slip_matrix = scipy.sparse.dok_matrix((m.n_dofs('fault'), m.n_tris('fault') * 3 * 2))
    fault_pts = m.get_tri_pts('fault')
    # slip dofs are an array like (n_fault_tris, 2) with the 1st component being strike-slip and 2nd being dip-slip
    for i in range(fault_pts.shape[0]):
        if get_vertical is None:
            vertical = [0,0,1]
        else:
            vertical = get_vertical(fault_pts[i])
        for vec_idx, vec in enumerate(slip_vectors.get_slip_vectors(fault_pts[i], vertical)):
            for b in range(3):
                for d in range(3):
                    slip_matrix[i * 9 + b * 3 + d, i * 6 + b * 2 + vec_idx] = vec[d]
    return slip_matrix

tt_cfg = dict(log_level = logging.INFO, preconditioner = 'ilu')
def assemble_integral_eqs(m):
    surf = (m.pts, m.get_tris('surf'))
    fault = (m.pts, m.get_tris('fault'))
    sm = 1.0
    pr = 0.25
    forward_system = tt.forward_assemble(surf, fault, sm, pr, **tt_cfg)
    adjoint_system = tt.adjoint_assemble(forward_system, sm, pr, **tt_cfg)
    return forward_system, adjoint_system

def get_vert_vals_linear(m, x):
    vert_n_tris = np.zeros(m[0].shape[0], dtype = np.int)
    for i in range(m[1].shape[0]):
        for b in range(3):
            vert_n_tris[m[1][i,b]] += 1
    vert_vals = np.zeros(m[0].shape[0])
    for i in range(m[1].shape[0]):
        for b in range(3):
            vert_vals[m[1][i,b]] += x[i,b]
    vert_vals[vert_n_tris != 0] /= vert_n_tris[vert_n_tris != 0]
    return vert_vals

def inversion(integral_eqs, soln_to_obs, rhs, slip_matrix, tol):
    def mv(v):
        _,_,_,soln = tt.forward_solve(integral_eqs[0], slip_matrix.dot(v), **tt_cfg)
        return soln_to_obs.dot(soln)

    def rmv(v):
        rhs = soln_to_obs.T.dot(v)
        _,_,_,soln = tt.adjoint_solve(integral_eqs[1], rhs, **tt_cfg)
        return slip_matrix.T.dot(soln)

    A = scipy.sparse.linalg.LinearOperator((rhs.shape[0], slip_matrix.shape[1]), matvec = mv, rmatvec = rmv)
    b = rhs.copy()

    # a low tolerance prevents overfitting without relying on explicit regularization.
    # but, if needed, tikhonov regularization can be added by setting damp = ###.
    inverse_soln = scipy.sparse.linalg.lsmr(A, b, show = True, atol = tol, btol = tol)
    return inverse_soln

def calc_modeled_data(integral_eqs, soln_to_obs, slip_matrix, inverse_soln):
    # from fault strike/dip slip to xyz slip
    cartesian_slip = slip_matrix.dot(inverse_soln[0])
    # get surf displacements from xyz slip
    u_surf = tt.forward_solve(integral_eqs[0], cartesian_slip, **tt_cfg)[3]
    # get displacements at the same location as the data
    data_modeled = soln_to_obs.dot(u_surf)
    return u_surf, data_modeled

def build_proj_rotate(m, from_proj, to_proj):
    all_pts = [m.pts.copy()]
    for d in range(3):
        moved = m.pts.copy()
        moved[:,d] += 1.0
        all_pts.append(moved)

    all_pts = np.concatenate(all_pts)
    lonlatz = collect_dem.project(all_pts[:,0], all_pts[:,1], all_pts[:,2], from_proj, inverse = True)
    to_coords = collect_dem.project(lonlatz[:,0], lonlatz[:,1], lonlatz[:,2], to_proj)

    npts = m.pts.shape[0]
    to_pts = to_coords[:npts]
    from_to_dirs = [to_coords[((d+1)*npts):((d+2)*npts)] - to_pts for d in range(3)]
    all_dirs = np.array(from_to_dirs)
    dir_map = np.swapaxes(np.swapaxes(all_dirs, 0, 1), 1, 2)
    dir_map[m.tris].shape

    proj_rotate = scipy.sparse.dok_matrix((m.n_dofs(), m.n_dofs()))
    dir_map_tris = dir_map[m.tris]
    for i in range(m.tris.shape[0]):
        for b in range(3):
            for d1 in range(3):
                for d2 in range(3):
                    proj_rotate[i * 9 + b * 3 + d1, i * 9 + b * 3 + d2] = dir_map_tris[i,b,d1,d2]
    return proj_rotate

#_pts = np.array([[0.1,0.1]])
#_obs_disp = np.array([[0,1]])
#_m = CombinedMesh.from_named_pieces([('surf', (np.array([[0,0,0],[1,0,0],[0,1,0]]), np.array([[0,1,2]])))])
#_S, _R = soln_to_obs, rhs = inverse_tools.setup_data_matrices(_m, _pts, _obs_disp, [0,1])
#np.testing.assert_almost_equal(_S.dot([1, 0, 0, 0, 0, 0, 0, 0 ,0]), [0.8, 0.0])
#np.testing.assert_almost_equal(_R, [0,1])
