import numpy as np
import scipy.sparse.csgraph
from tectosaur.util.geometry import tri_normal
from tectosaur.constraints import ConstraintEQ, Term

def find_touching_pts(tris):
    max_pt_idx = np.max(tris)
    out = [[] for i in range(max_pt_idx + 1)]
    for i, t in enumerate(tris):
        for d in range(3):
            out[t[d]].append((i, d))
    return out

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

def tri_side(tri1, tri2, threshold = 1e-12):
    tri1_normal = tri_normal(tri1, normalize = True)
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

def get_surf_fault_tips(surf_tris, fault_tris):
    surf_fault_edges = np.array(get_surf_fault_edges(
        surf_tris, fault_tris
    ))
    if surf_fault_edges.shape[0] == 0:
        return np.array([])
    surf_fault_edge_vertices = fault_tris[surf_fault_edges[:,:,0], surf_fault_edges[:,:,1]]
    verts, counts = np.unique(surf_fault_edge_vertices, return_counts = True)
    return verts[counts < 2]

def get_side_of_fault(pts, tris, fault_start_idx):
    connectivity = tri_connectivity_graph(tris)
    fault_touching_pair = np.where(np.logical_and(
        connectivity.row < fault_start_idx,
        connectivity.col >= fault_start_idx
    ))[0]

    side = np.zeros(tris.shape[0])
    shared_verts = np.zeros(tris.shape[0])
    for i in range(fault_touching_pair.shape[0]):
        surf_tri_idx = connectivity.row[fault_touching_pair[i]]
        surf_tri = tris[surf_tri_idx]
        fault_tri_idx = connectivity.col[fault_touching_pair[i]]
        fault_tri = tris[fault_tri_idx]
        which_side = tri_side(pts[fault_tri], pts[surf_tri])

        n_shared_verts = 0
        for d in range(3):
            if surf_tri[d] in fault_tri:
                n_shared_verts += 1

        if shared_verts[surf_tri_idx] < 2:
            side[surf_tri_idx] = int(which_side) + 1
            shared_verts[surf_tri_idx] = n_shared_verts
    return side

#TODO: this function needs to know the idxs of the surface_tris and fault_tris, so use
# idx lists and pass the full tris array, currently using the (n_surf_tris * 9) hack!
def continuity_constraints(pts, tris, fault_start_idx, tensor_dim = 3):
    surface_tris = tris[:fault_start_idx]
    fault_tris = tris[fault_start_idx:]
    touching_pt = find_touching_pts(surface_tris)
    side = get_side_of_fault(pts, tris, fault_start_idx)
    fault_tips = get_surf_fault_tips(tris[:fault_start_idx], tris[fault_start_idx:])

    constraints = []
    for i, tpt in enumerate(touching_pt):
        if len(tpt) == 0:
            continue

        for independent_idx in range(len(tpt)):
            independent = tpt[independent_idx]
            independent_tri_idx = independent[0]
            independent_corner_idx = independent[1]
            independent_tri = surface_tris[independent_tri_idx]

            for dependent_idx in range(independent_idx + 1, len(tpt)):
                dependent = tpt[dependent_idx]
                dependent_tri_idx = dependent[0]
                dependent_corner_idx = dependent[1]
                dependent_tri = surface_tris[dependent_tri_idx]

                # Check for anything that touches across the fault.
                side1 = side[independent_tri_idx]
                side2 = side[dependent_tri_idx]
                crosses = (
                    (side1 != side2) and (side1 != 0) and (side2 != 0)
                    and dependent_tri[dependent_corner_idx] not in fault_tips
                )
                fault_tri_idx = None
                if crosses:
                    fault_tri_idxs, fault_corner_idxs = np.where(
                        fault_tris == dependent_tri[dependent_corner_idx]
                    )
                    if fault_tri_idxs.shape[0] != 0:
                        fault_tri_idx = fault_tri_idxs[0]
                        fault_corner_idx = fault_corner_idxs[0]

                        # plt_pts = np.vstack((
                        #     pts[independent_tri],
                        #     pts[dependent_tri],
                        #     pts[fault_tris[fault_tri_idx]]
                        # ))
                        # import matplotlib.pyplot as plt
                        # plt.tripcolor(pts[:,0], pts[:,1], tris[:surface_tris.shape[0]], side[:surface_tris.shape[0]])
                        # plt.triplot(plt_pts[:,0], plt_pts[:,1], np.array([[0,1,2]]), 'b-')
                        # plt.triplot(plt_pts[:,0], plt_pts[:,1], np.array([[3,4,5]]), 'k-')
                        # plt.triplot(pts[:,0], pts[:,1], tris[fault_start_idx:], 'r-')
                        # plt.show()

                for d in range(tensor_dim):
                    independent_dof = (independent_tri_idx * 3 + independent_corner_idx) * tensor_dim + d
                    dependent_dof = (dependent_tri_idx * 3 + dependent_corner_idx) * tensor_dim + d
                    if dependent_dof <= independent_dof:
                        continue
                    diff = 0.0
                    terms = [Term(1.0, dependent_dof), Term(-1.0, independent_dof)]
                    if fault_tri_idx is not None:
                        fault_dof = (
                            fault_start_idx * 9 +
                            fault_tri_idx * 9 + fault_corner_idx * 3 + d
                        )
                        if side1 < side2:
                            terms.append(Term(-1.0, fault_dof))
                        else:
                            terms.append(Term(1.0, fault_dof))
                    constraints.append(ConstraintEQ(terms, 0.0))
    return constraints

