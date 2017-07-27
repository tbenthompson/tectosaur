from tectosaur.mesh.adjacency import find_touching_pts, find_free_edges
import tectosaur.util.geometry as geom
from tectosaur.constraints import ConstraintEQ, Term

def build_composite_constraints(*cs_and_starts):
    all_cs = []
    for cs, start in cs_and_starts:
        for c in cs:
            all_cs.append(ConstraintEQ(
                [Term(t.val, t.dof + start) for t in c.terms], c.rhs
            ))
    return all_cs

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

def elastic_rigid_body_constraints(pts, tris, basis_idxs):
    fixed_pt_idx = basis_idxs[0]
    fixed_pt = pts[tris[fixed_pt_idx[0], fixed_pt_idx[1]]]
    lengthening_pt_idx = basis_idxs[1]
    lengthening_pt = pts[tris[lengthening_pt_idx[0], lengthening_pt_idx[1]]]
    in_plane_pt_idx = basis_idxs[2]
    in_plane_pt = pts[tris[in_plane_pt_idx[0], in_plane_pt_idx[1]]]

    # Fix the location of the first point.
    cs = []
    for d in range(3):
        dof = fixed_pt_idx[0] * 9 + fixed_pt_idx[1] * 3 + d
        cs.append(ConstraintEQ([Term(1.0, dof)], 0.0))

    # Remove rotations between the two points.
    sep_vec = lengthening_pt - fixed_pt

    # Guaranteed to be orthogonal
    if sep_vec[2] != 0.0:
        orthogonal_vec1 = np.array([1, 1, (-sep_vec[0] - sep_vec[1]) / sep_vec[2]])
    elif sep_vec[1] != 0.0:
        orthogonal_vec1 = np.array([1, (-sep_vec[0] - sep_vec[2]) / sep_vec[1], 1])
    else:
        orthogonal_vec1 = np.array([(-sep_vec[1] - sep_vec[2]) / sep_vec[0], 1, 1])
    orthogonal_vec1 /= np.linalg.norm(orthogonal_vec1)
    orthogonal_vec2 = np.cross(sep_vec, orthogonal_vec1)

    for v in [orthogonal_vec1, orthogonal_vec2]:
        ts = []
        for d in range(3):
            dof = lengthening_pt_idx[0] * 9 + lengthening_pt_idx[1] * 3 + d
            ts.append(Term(v[d], dof))
        cs.append(ConstraintEQ(ts, 0.0))

    # Keep the third point in the same plane as the first two points.
    sep_vec2 = in_plane_pt - fixed_pt
    plane_normal = np.cross(sep_vec, sep_vec2)
    plane_normal /= np.linalg.norm(plane_normal)

    ts = []
    for d in range(3):
        dof = in_plane_pt_idx[0] * 9 + in_plane_pt_idx[1] * 3 + d
        ts.append(Term(plane_normal[d], dof))
    cs.append(ConstraintEQ(ts, 0.0))

    return cs



def continuity_constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    touching_pt = find_touching_pts(surface_tris)
    if fault_tris.shape[0] > 0:
        fault_touching_pt = find_touching_pts(fault_tris)
    constraints = []
    for i, tpt in enumerate(touching_pt):
        if len(tpt) == 0:
            continue

        tri1_idx = tpt[0][0]
        tri1 = surface_tris[tri1_idx]
        for dependent in tpt[1:]:
            tri2_idx = dependent[0]
            tri2 = surface_tris[tri2_idx]

            # Check for anything that touches across the fault.
            crosses = (
                fault_tris.shape[0] > 0
                and check_if_crosses_fault(
                    tri1, tri2, fault_touching_pt[i], fault_tris, pts
                )
            )
            if crosses:
                continue

            for d in range(3):
                indepedent_dof = tri1_idx * 9 + tpt[0][1] * 3 + d
                dependent_dof = tri2_idx * 9 + dependent[1] * 3 + d
                if dependent_dof <= indepedent_dof:
                    continue
                constraints.append(ConstraintEQ(
                    [Term(1.0, dependent_dof), Term(-1.0, indepedent_dof)], 0.0
                ))

    return constraints

def free_edge_constraints(tris):
    free_edges = find_free_edges(tris)
    cs = []
    for tri_idx, edge_idx in free_edges:
        for v in range(2):
            for d in range(3):
                dof = tri_idx * 9 + ((edge_idx + v) % 3) * 3 + d
                cs.append(ConstraintEQ([Term(1.0, dof)], 0.0))
    return cs

def all_bc_constraints(start_tri, end_tri, vs):
    cs = []
    for i in range(start_tri * 9, end_tri * 9):
        cs.append(ConstraintEQ([Term(1.0, i)], vs[i - start_tri * 9]))
    return cs

def constant_bc_constraints(start_tri, end_tri, value):
    cs = []
    for i in range(start_tri, end_tri):
        for b in range(3):
            for d in range(3):
                dof = i * 9 + b * 3 + d
                cs.append(ConstraintEQ([Term(1.0, dof)], value[d]))
    return cs
