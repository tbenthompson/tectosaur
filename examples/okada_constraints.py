from tectosaur.adjacency import find_touching_pts
import tectosaur.geometry as geom

def check_if_crosses_fault(tri1, tri2, fault_touching_pts, fault_tris, pts):
    for fault_tri_idx,_ in fault_touching_pts:
        fault_t = fault_tris[fault_tri_idx]
        plane = pts[fault_t]
        tri1_sides = [geom.which_side_point(plane, pts[tri1[d]]) for d in range(3)]
        tri2_sides = [geom.which_side_point(plane, pts[tri2[d]]) for d in range(3)]
        side1 = geom.tri_side(tri1_sides)
        side2 = geom.tri_side(tri2_sides)
        if side1 != side2:
            return True
    return False

def constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    touching_pt = find_touching_pts(surface_tris)
    fault_touching_pt = find_touching_pts(fault_tris)
    constraints = []
    for i, tpt in enumerate(touching_pt):

        tri1_idx = tpt[0][0]
        tri1 = surface_tris[tri1_idx]
        for dependent in tpt[1:]:
            tri2_idx = dependent[0]
            tri2 = surface_tris[tri2_idx]

            # Check for anything that touches across the fault.
            if check_if_crosses_fault(tri1, tri2, fault_touching_pt[i], fault_tris, pts):
                print("HI")
                continue

            for d in range(3):
                indepedent_dof = tri1_idx * 9 + tpt[0][1] * 3 + d
                dependent_dof = tri2_idx * 9 + dependent[1] * 3 + d
                if dependent_dof <= indepedent_dof:
                    continue
                constraints.append((
                    [(1.0, dependent_dof), (-1.0, indepedent_dof)], 0.0
                ))
    # assert(len(touching_pt) == surface_tris.size - len(constraints) / 3)

    # X component = 1
    # Y comp = Z comp = 0
    slip = [-1, 0, 0]
    for i in range(n_surf_tris, n_surf_tris + n_fault_tris):
        for d in range(3):
            for b in range(3):
                dof = i * 9 + b * 3 + d
                constraints.append(([(1.0, dof)], slip[d]))
    constraints = sorted(constraints, key = lambda x: x[0][0][1])
    return constraints

def insert_constraints(lhs, rhs, cs):
    c_start = lhs.shape[0] - len(cs)
    for i, c in enumerate(cs):
        idx1 = c_start + i
        rhs[idx1] = c[1]
        for dw in c[0]:
            coeff = dw[0]
            idx2 = dw[1]
            lhs[idx1, idx2] = coeff
            lhs[idx2, idx1] = coeff
