import numpy as np
import tectosaur.mesh.find_near_adj as find_near_adj
from tectosaur.nearfield.table_params import table_min_internal_angle, min_intersect_angle

from tectosaur.util.cpp import imp
edge_adj_setup = imp('tectosaur.nearfield.edge_adj_setup')
standardize = imp('tectosaur.nearfield.standardize')

def check_edge_adj_sizes(pts, tris, ea):
    pass

def check_min_adj_angle(m):
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
