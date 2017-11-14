import numpy as np
import tectosaur.mesh.find_near_adj as find_near_adj
from tectosaur.nearfield.table_params import table_min_internal_angle, min_intersect_angle

from tectosaur.util.cpp import imp
edge_adj_setup = imp('tectosaur.nearfield.edge_adj_setup')
standardize = imp('tectosaur.nearfield.standardize')
tri_tri_intersect = imp('tectosaur.util.tri_tri_intersect')

def check_min_adj_angle(m, ea = None):
    pts, tris = m
    close_or_touch_pairs = find_near_adj.find_close_or_touching(pts, tris, 2.0)
    nearfield_pairs, va, ea = find_near_adj.split_adjacent_close(close_or_touch_pairs, tris)
    bad_pairs = []
    lower_lim = min_intersect_angle
    upper_lim = (2 * np.pi - min_intersect_angle)
    for pair in ea:
        obs_tri = pts[tris[pair[0]]]
        src_tri = pts[tris[pair[1]]]
        phi = edge_adj_setup.calc_adjacent_phi(obs_tri.tolist(), src_tri.tolist())
        if lower_lim <= phi <= upper_lim:
            continue
        bad_pairs.append(pair)
    return np.array(bad_pairs)

def check_for_slivers(m):
    pts, tris = m
    bad_tris = []
    for i, t in enumerate(tris):
        t_list = pts[t].tolist()
        try:
            out = standardize.standardize(t_list, table_min_internal_angle, True)
        except standardize.BadTriangleException as e:
            bad_tris.append(i)
    return bad_tris

def check_tris_tall_enough(m):
    pts, tris = m
    bad_tris = []
    for i, t in enumerate(tris):
        t_list = pts[t].tolist()
        split_pt = edge_adj_setup.get_split_pt(t_list);
        xyhat = edge_adj_setup.xyhat_from_pt(split_pt, t_list)
        if not edge_adj_setup.check_xyhat(xyhat):
            bad_tris.append(i)
    return bad_tris

def check_for_intersections_nearfield(pts, tris, nearfield_pairs):
    if nearfield_pairs.shape[0] == 0:
        return []
    unique_near_pairs = np.unique(np.sort(nearfield_pairs, axis = 1), axis = 0)
    bad_pairs = []
    for pair in unique_near_pairs:
        tri1 = pts[tris[pair[0]]].tolist()
        tri2 = pts[tris[pair[1]]].tolist()
        I = tri_tri_intersect.tri_tri_intersect(tri1, tri2)
        if I:
            bad_pairs.append(pair)
    return bad_pairs

def check_for_intersections_va(pts, tris, va):
    if va.shape[0] == 0:
        return []
    unique_va = np.unique(np.hstack((np.sort(va[:,:2], axis = 1), va[:,2:])), axis = 0)
    bad_pairs = []
    for pair in unique_va:
        tri1 = pts[tris[pair[0]]]

        for edge in [1,2]:
            shared_pt_tri1 = pair[2]
            tri1[0] += (tri1[edge] - tri1[0]) * 0.01

            tri2 = pts[tris[pair[1]]].tolist()
            I = tri_tri_intersect.tri_tri_intersect(tri1.tolist(), tri2)
            if I:
                bad_pairs.append(pair[:2])
                break
    return bad_pairs

def check_for_intersections_ea(pts, tris, ea):
    if ea.shape[0] == 0:
        return []
    unique_ea = np.unique(np.hstack((np.sort(ea[:,:2], axis = 1), ea[:,2:])), axis = 0)
    bad_pairs = []
    for pair in unique_ea:
        for d in range(3):
            if tris[pair[0],d] in tris[pair[1]]:
                continue
            dist = np.sqrt(np.sum((pts[tris[pair[1]]] - pts[tris[pair[0],d]]) ** 2, axis = 1))
            if np.any(dist == 0):
                bad_pairs.append(pair)
    return bad_pairs

def check_for_intersections(m):
    pts, tris = m
    close_or_touch_pairs = find_near_adj.find_close_or_touching(pts, tris, 2.0)
    nearfield_pairs, va, ea = find_near_adj.split_adjacent_close(close_or_touch_pairs, tris)

    bad_pairs = []

    # Three situations:
    # 1) Nearfield pair is intersection
    bad_pairs.extend(check_for_intersections_nearfield(pts, tris, nearfield_pairs))

    # 2) Vertex adjacent pair actually intersects beyond just the vertex
    # We can test for this by moving the shared vertex short distance
    # along one of the edges of the first triangle. If there is still
    # an intersection, then the triangles intersect at locations besides
    # just the shared vertex.
    bad_pairs.extend(check_for_intersections_va(pts, tris, va))

    # 3) Edge adjacent pair is actually coincident. <-- Easy!
    bad_pairs.extend(check_for_intersections_ea(pts, tris, ea))

    return np.array(bad_pairs)

def check_for_problems(m):
    intersections = check_for_intersections(m)
    slivers = check_for_slivers(m)
    short_tris = check_tris_tall_enough(m)
    sharp_angles = check_min_adj_angle(m)
    return intersections, sliver, short_tris, sharp_angles
