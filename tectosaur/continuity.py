import numpy as np
import scipy.sparse.csgraph
from tectosaur.util.geometry import tri_normal, unscaled_normals, normalize
from tectosaur.constraints import ConstraintEQ, Term
from tectosaur.stress_constraints import stress_constraints, stress_constraints2, \
    equilibrium_constraint, constant_stress_constraint

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

def get_side_of_fault(pts, tris, fault_start_idx):
    connectivity = tri_connectivity_graph(tris)
    fault_touching_pair = np.where(np.logical_and(
        connectivity.row < fault_start_idx,
        connectivity.col >= fault_start_idx
    ))[0]

    side = np.zeros(tris.shape[0])
    shared_verts = np.zeros(tris.shape[0])

    fault_surf_tris = pts[tris[connectivity.col[fault_touching_pair]]]
    for i in range(fault_touching_pair.shape[0]):
        surf_tri_idx = connectivity.row[fault_touching_pair[i]]
        surf_tri = tris[surf_tri_idx]
        fault_tri = tris[connectivity.col[fault_touching_pair[i]]]
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
#TODO: refactor and merge this with the traction continuity constraints
def continuity_constraints(pts, tris, fault_start_idx, tensor_dim = 3):
    surface_tris = tris[:fault_start_idx]
    fault_tris = tris[fault_start_idx:]
    touching_pt = find_touching_pts(surface_tris)
    side = get_side_of_fault(pts, tris, fault_start_idx)
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
                crosses = (side1 != side2) and (side1 != 0) and (side2 != 0)
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

def traction_admissibility_constraints(pts, tris, fault_start_idx):
    # At each vertex, there should be three remaining degrees of freedom.
    # Initially, there are n_tris*3 degrees of freedom.
    # So, we need (n_tris-1)*3 constraints.

    touching_pt = find_touching_pts(tris)
    ns = normalize(unscaled_normals(pts[tris]))
    side = get_side_of_fault(pts, tris, fault_start_idx)

    continuity_cs = []
    admissibility_cs = []
    for tpt in touching_pt:
        if len(tpt) == 0:
            continue

        # Separate the triangles touching at the vertex into a groups
        # by the normal vectors for each triangle.
        normal_groups = []
        for i in range(len(tpt)):
            tri_idx = tpt[i][0]
            n = ns[tri_idx]
            joined = False
            for j in range(len(normal_groups)):
                if np.allclose(normal_groups[j][0], n):
                    tri_idx2 = tpt[normal_groups[j][1][0]][0]
                    side1 = side[tri_idx]
                    side2 = side[tri_idx2]
                    crosses = (side1 != side2) and (side1 != 0) and (side2 != 0)
                    fault_tri_idx = None
                    # if crosses:
                    #     continue
                    normal_groups[j][1].append(i)
                    joined = True
                    break
            if not joined:
                normal_groups.append((n, [i]))

        # Continuity within normal group
        for i in range(len(normal_groups)):
            group = normal_groups[i][1]
            independent_idx = group[0]
            independent = tpt[independent_idx]
            independent_tri_idx = independent[0]
            independent_corner_idx = independent[1]
            independent_dof_start = independent_tri_idx * 9 + independent_corner_idx * 3
            for j in range(1, len(group)):
                dependent_idx = group[j]
                dependent = tpt[dependent_idx]
                dependent_tri_idx = dependent[0]
                dependent_corner_idx = dependent[1]
                dependent_dof_start = dependent_tri_idx * 9 + dependent_corner_idx * 3
                for d in range(3):
                    terms = [
                        Term(1.0, dependent_dof_start + d),
                        Term(-1.0, independent_dof_start + d)
                    ]
                    continuity_cs.append(ConstraintEQ(terms, 0.0))

        if len(normal_groups) == 1:
            # Only continuity needed!
            continue

        # assert(len(normal_groups) == 2)

        # Add constant stress constraints
        for i in range(len(normal_groups)):
            tpt_idx1 = normal_groups[i][1][0]
            tri_idx1 = tpt[tpt_idx1][0]
            corner_idx1 = tpt[tpt_idx1][1]
            tri1 = pts[tris[tri_idx1]]
            tri_data1 = (tri1, tri_idx1, corner_idx1)

            for j in range(i + 1, len(normal_groups)):
                tpt_idx2 = normal_groups[j][1][0]
                tri_idx2 = tpt[tpt_idx2][0]
                # print(tri_idx1, tri_idx2)
                corner_idx2 = tpt[tpt_idx2][1]
                tri2 = pts[tris[tri_idx2]]
                tri_data2 = (tri2, tri_idx2, corner_idx2)

                # for c in new_cs:
                #     print(', '.join(['(' + str(t.val) + ',' + str(t.dof) + ')' for t in c.terms]) + ' rhs: ' + str(c.rhs))
                admissibility_cs.append(constant_stress_constraint(tri_data1, tri_data2))
                admissibility_cs.append(equilibrium_constraint(tri_data1))
                admissibility_cs.append(equilibrium_constraint(tri_data2))
    return continuity_cs, admissibility_cs
