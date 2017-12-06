import numpy as np

from tectosaur.constraints import ConstraintEQ, Term

def find_touching_pts(tris):
    max_pt_idx = np.max(tris)
    out = [[] for i in range(max_pt_idx + 1)]
    for i, t in enumerate(tris):
        for d in range(3):
            out[t[d]].append((i, d))
    return out

def build_composite_constraints(*cs_and_starts):
    all_cs = []
    for cs, start in cs_and_starts:
        for c in cs:
            all_cs.append(ConstraintEQ(
                [Term(t.val, t.dof + start) for t in c.terms], c.rhs
            ))
    return all_cs

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


# Crossing a fault can be defined in an entirely topological sense.
# If a pair of triangles shares an edge that is not a fault edge, then
# those triangle are not across the fault from each other.
def check_if_crosses_fault(tri1, tri2, fault_touching_pts, fault_tris):
    shared_pts = []
    for d in range(3):
        idx = np.where(tri2 == tri1[d])[0]
        if idx.shape[0] == 0:
            continue
        shared_pts.append(d)
    assert(0 < len(shared_pts) < 3)

    if len(shared_pts) == 1:
        if len(fault_touching_pts[tri1[shared_pts[0]]]) > 0:
            return True

    if len(shared_pts) == 2:
        p1_touches = len(fault_touching_pts[tri1[shared_pts[0]]]) > 0
        p2_touches = len(fault_touching_pts[tri1[shared_pts[1]]]) > 0
        if p1_touches and p2_touches:
            return True

    return False

def continuity_constraints(surface_tris, fault_tris, tensor_dim = 3):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    touching_pt = find_touching_pts(surface_tris)
    if fault_tris.shape[0] > 0:
        fault_touching_pt = find_touching_pts(fault_tris)
    else:
        fault_touching_pt = []
    fault_touching_pt.extend(
        [[] for i in range(len(touching_pt) - len(fault_touching_pt))]
    )
    constraints = []
    for i, tpt in enumerate(touching_pt):
        if len(tpt) == 0:
            continue

        for independent_idx in range(len(tpt)):
            independent = tpt[independent_idx]
            independent_tri_idx = independent[0]
            independent_tri = surface_tris[independent_tri_idx]

            for dependent_idx in range(independent_idx + 1, len(tpt)):
                dependent = tpt[dependent_idx]
                dependent_tri_idx = dependent[0]
                dependent_tri = surface_tris[dependent_tri_idx]

                # Check for anything that touches across the fault.
                crosses = (
                    fault_tris.shape[0] > 0
                    and check_if_crosses_fault(
                        independent_tri, dependent_tri, fault_touching_pt, fault_tris
                    )
                )

                if crosses:
                    continue

                for d in range(tensor_dim):
                    independent_dof = (independent_tri_idx * 3 + independent[1]) * tensor_dim + d
                    dependent_dof = (dependent_tri_idx * 3 + dependent[1]) * tensor_dim + d
                    if dependent_dof <= independent_dof:
                        continue
                    diff = 0.0
                    constraints.append(ConstraintEQ(
                        [Term(1.0, dependent_dof), Term(-1.0, independent_dof)], diff
                    ))
    return constraints

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
