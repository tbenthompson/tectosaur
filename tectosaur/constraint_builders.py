import numpy as np

from tectosaur.constraints import ConstraintEQ, Term

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

def check_continuity(tris, field):
    n_pts = np.max(tris) + 1
    discontinuity_pts = []
    for i in range(n_pts):
        tri_idxs, corner_idxs = np.where(tris == i)
        # print(i, tri_idxs, corner_idxs)
        vals = field[tri_idxs * 3 + corner_idxs]
        # print(vals, vals[0])
        if not np.all(vals == vals[0]):
            discontinuity_pts.append(i)
    return discontinuity_pts

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

# TODO: normally dofs refer to each element of a 3d vector (9 * i + 3 * v + d)
# but here dofs refers to each element of a 1d vector (3 * i + v).
# It would be good to have different words for these two concepts.
def free_edge_dofs(tris, free_edges):
    pt_idxs = set()
    for tri_idx, edge_idx in free_edges:
        for v in range(2):
            pt_idxs.add(tris[tri_idx][(edge_idx + v) % 3])
    dofs = []
    for i, t in enumerate(tris):
        for v in range(3):
            if t[v] not in pt_idxs:
                continue
            dofs.append(i * 3 + v * 1)
    return dofs

def plot_free_edges(pts, tris, free_edges, dims = [0,1]):
    # import matplotlib here so it doesn't slow down importing the main module
    # when it isn't needed
    import matplotlib.pyplot as plt
    for tri_idx, edge_idx in free_edges:
        edge_pts = np.array([
            pts[tris[tri_idx][(edge_idx + v) % 3]] for v in range(2)
        ])
        plt.plot(edge_pts[:,0], edge_pts[:,1], '*-')
    plt.show()

def free_edge_constraints(tris):
    free_edges = find_free_edges(tris)
    cs = []
    for dof in free_edge_dofs(tris, free_edges):
        for d in range(3):
            vec_dof = dof * 3 + d
            cs.append(ConstraintEQ([Term(1.0, vec_dof)], 0.0))
    return cs

def jump_constraints(jump, negative):
    n_dofs_per_side = jump.shape[0]
    cs = []
    coeff_2 = 1.0 if negative else -1.0
    for i in range(n_dofs_per_side):
        dof_1 = i
        dof_2 = i + n_dofs_per_side
        ts = []
        ts.append(Term(1.0, dof_1))
        ts.append(Term(coeff_2, dof_2))
        cs.append(ConstraintEQ(ts, jump[i]))
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
