import numpy as np
import scipy.sparse.csgraph
from tectosaur.util.geometry import tri_normal, unscaled_normals, normalize
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

basis_gradient = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T
def inv_jacobian(tri):
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    n = np.cross(v1, v2)
    jacobian = [v1, v2, n]
    inv_jacobian = np.linalg.inv(jacobian)
    return inv_jacobian[:,:2]

def calc_gradient(tri, field):
    return (inv_jacobian(tri).dot(basis_gradient.dot(field))).T

def calc_shear_stress(tri, disp, t1, t2, sm):
    G = calc_gradient(tri, disp)
    return (sm / 2.0) * (G.dot(t1).dot(t2) + G.dot(t2).dot(t1))

def traction_admissibility_constraints(pts, tris, disp, sm):
    touching_pt = find_touching_pts(tris)
    constraints = []
    ns = normalize(unscaled_normals(pts[tris]))

    # for i in range(tris.shape[0]):
    #     Jinv = inv_jacobian(pts[tris[i]])
    #     basis_gradient
    #     terms = []
    #     for d in range(3):
    #         for basis_component in range(3):
    #             dof = i * 9 + basis_component * 3 + d
    #             val = 0
    #             for k in range(2):
    #                 val += Jinv[d,k] * basis_gradient[k,basis_component]
    #             terms.append(Term(val, dof))
    #     constraints.append(ConstraintEQ(terms, 0.0))

    for i, tpt in enumerate(touching_pt):
        if len(tpt) == 0:
            continue

        for independent_idx in range(len(tpt)):
            independent = tpt[independent_idx]
            independent_tri_idx = independent[0]
            independent_corner_idx = independent[1]
            independent_tri = tris[independent_tri_idx]
            independent_n = ns[independent_tri_idx]
            independent_dof_start = independent_tri_idx * 9 + independent_corner_idx * 3

            for dependent_idx in range(independent_idx + 1, len(tpt)):
                dependent = tpt[dependent_idx]
                dependent_tri_idx = dependent[0]
                dependent_corner_idx = dependent[1]
                dependent_tri = tris[dependent_tri_idx]
                dependent_n = ns[dependent_tri_idx]
                dependent_dof_start = dependent_tri_idx * 9 + dependent_corner_idx * 3

                if dependent_tri_idx <= independent_tri_idx:
                    continue

                if np.allclose(independent_n, dependent_n):
                    for d in range(3):
                        terms = [
                            Term(1.0, dependent_dof_start + d),
                            Term(-1.0, independent_dof_start + d)
                        ]
                        constraints.append(ConstraintEQ(terms, 0.0))
                    continue

                terms = []
                for d in range(3):
                    terms.append(Term(-independent_n[d], dependent_dof_start + d))
                    terms.append(Term(dependent_n[d], independent_dof_start + d))
                constraints.append(ConstraintEQ(terms, 0.0))

                # #calculate r vectors
                # r2 = np.cross(independent_n, dependent_n)
                # r2 /= np.linalg.norm(r2)
                # r1 = np.cross(independent_n, r2)
                # r3 = np.cross(dependent_n, r2)

                # # calculate decomposition factors
                # a = independent_n.dot(dependent_n)
                # b = dependent_n.dot(r1)
                # c = a
                # d = independent_n.dot(r3)

                # np.testing.assert_almost_equal(independent_n, c * dependent_n + d * r3)
                # np.testing.assert_almost_equal(dependent_n, a * independent_n + b * r1)

                # # calculate shear stress from slip
                # independent_disp = disp.reshape((-1,3,3))[independent_tri_idx]
                # S1 = calc_shear_stress(pts[independent_tri], independent_disp, r1, r2, sm)
                # dependent_disp = disp.reshape((-1,3,3))[dependent_tri_idx]
                # S2 = calc_shear_stress(pts[dependent_tri], dependent_disp, r3, r2, sm)

                # terms = []
                # for d in range(3):
                #     terms.append(Term(r2[d], independent_dof_start + d))
                #     terms.append(Term(-c * r2[d], dependent_dof_start + d))
                # constraints.append(ConstraintEQ(terms, -d * S2))

                # terms = []
                # for d in range(3):
                #     terms.append(Term(r2[d], dependent_dof_start + d))
                #     terms.append(Term(-a * r2[d], independent_dof_start + d))
                # constraints.append(ConstraintEQ(terms, -b * S1))
    return constraints
