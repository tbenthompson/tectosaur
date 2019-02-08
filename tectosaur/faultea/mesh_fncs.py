import numpy as np
import scipy.interpolate
import scipy.sparse.csgraph
from . import collect_dem
import copy

def remove_unused_pts(m):
    referenced_pts = np.unique(m[1])
    new_pts = m[0][referenced_pts,:]
    new_indices = np.empty(m[0].shape[0], dtype = np.int64)
    new_indices[referenced_pts] = np.arange(referenced_pts.shape[0])
    new_tris = new_indices[m[1]]
    return (new_pts, new_tris)

#TODO: This function is in tectosaur.continuity
def get_surf_fault_edges(surf_tris, fault_tris):
    surf_verts = np.unique(surf_tris)
    surf_fault_edges = []
    for i, t in enumerate(fault_tris):
        in_surf = []
        for d in range(3):
            if t[d] in surf_verts:
                in_surf.append((i, d))
        if len(in_surf) == 2:
            surf_fault_edges.append(in_surf)
    return surf_fault_edges

def get_surf_fault_pts(surf_tris, fault_tris):
    surf_fault_pts = []
    for e in get_surf_fault_edges(surf_tris, fault_tris):
        for j in range(2):
            t_idx, d = e[j]
            surf_fault_pts.append(fault_tris[1][t_idx, d])
    surf_fault_pts = np.unique(surf_fault_pts)
    return surf_fault_pts

def set_surf_elevations(m, n_dem_interp_pts, zoom, proj):
    surf_pt_idxs = m.get_pt_idxs('surf')
    lonlat_pts = collect_dem.project(m.pts[:,0], m.pts[:,1], m.pts[:,2], proj, inverse = True)
    bounds = collect_dem.get_dem_bounds(lonlat_pts)
    proj_dem = collect_dem.project(*collect_dem.get_dem(zoom, bounds, n_dem_interp_pts), proj)
    new_m = copy.copy(m)
    new_m.pts[surf_pt_idxs,2] = scipy.interpolate.griddata(
        (proj_dem[:,0], proj_dem[:,1]), proj_dem[:,2],
        (m.pts[surf_pt_idxs,0], m.pts[surf_pt_idxs,1])
    )
    return new_m


def tri_connectivity_graph(tris):
    n_tris = tris.shape[0]
    touching = [[] for i in range(np.max(tris) + 1)]
    for i in range(n_tris):
        for d in range(3):
            touching[tris[i,d]].append(i)

    rows = []
    cols = []
    for i in range(len(touching)):
        for row in touching[i]:
            for col in touching[i]:
                rows.append(row)
                cols.append(col)
    rows = np.array(rows)
    cols = np.array(cols)
    connectivity = scipy.sparse.coo_matrix((np.ones(rows.shape[0]), (rows, cols)), shape = (n_tris, n_tris))
    return connectivity

def get_connected_components(tris):
    return scipy.sparse.csgraph.connected_components(tri_connectivity_graph(tris))

def find_free_edges(tris):
    edges = dict()
    for i, t in enumerate(tris):
        for d in range(3):
            pt1_idx = t[d]
            pt2_idx = t[(d + 1) % 3]
            if pt1_idx > pt2_idx:
                pt2_idx,pt1_idx = pt1_idx,pt2_idx
            pt_pair = (pt1_idx, pt2_idx)
            edges[pt_pair] = edges.get(pt_pair, []) + [(i, d)]

    free_edges = []
    for k,e in edges.items():
        if len(e) > 1:
            continue
        free_edges.append(e[0])

    return free_edges

def get_boundary_loop(m):
    which_comp = get_connected_components(m[1])[1]
    n_surfaces = np.unique(which_comp).shape[0]
    orderings = []
    for surf_idx in range(n_surfaces):
        tri_subset = m[1][which_comp == surf_idx]
        free_edges = find_free_edges(tri_subset)
        pt_to_pt = [
            (tri_subset[tri_idx, edge_idx], tri_subset[tri_idx, (edge_idx + 1) % 3])
            for tri_idx, edge_idx in free_edges
        ]

        pts_to_edges = dict()
        for i, e in enumerate(pt_to_pt):
            for lr in [0,1]:
                pts_to_edges[e[lr]] = pts_to_edges.get(e[lr], []) + [i]

        for k,v in pts_to_edges.items():
            assert(len(v) == 2)

        ordering = [pt_to_pt[0][0], pt_to_pt[0][1]]
        looped = False
        while not looped:
            pt_idx = ordering[-1]
            prev_pt_idx = ordering[-2]
            for e_idx in pts_to_edges[pt_idx]:
                edge = pt_to_pt[e_idx]
                if edge[0] == prev_pt_idx or edge[1] == prev_pt_idx:
                    continue
                if edge[0] == pt_idx:
                    ordering.append(edge[1])
                else:
                    ordering.append(edge[0])
            if ordering[-1] == ordering[0]:
                looped = True
        orderings.append(ordering)
    return orderings

def to_sphere_xyz(m, proj):
    lonlatz = collect_dem.project(m.pts[:,0], m.pts[:,1], m.pts[:,2], proj, inverse = True)
    xyz_sphere = collect_dem.project(lonlatz[:,0], lonlatz[:,1], lonlatz[:,2], 'ellps')
    m_xyz = copy.copy(m)
    m_xyz.pts = xyz_sphere
    return m_xyz
