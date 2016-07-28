from tectosaur.adjacency import find_adjacents
# parameters:
# nearfield offset (ratio of element length)
# nearfield obs gauss order
# nearfield coincident gauss order
# nearfield edge adjacent gauss order
# nearfield vertex adjacent gauss order
# nearfield non-touching gauss order
# nearfield non-touching threshold
# farfield gauss order
class DenseIntegralOp:
    def __init__(self, near_offset, near_obs_order, near_co_order, near_edge_adj_order,
        near_vert_adj_order, near_no_touch_order, near_no_touch_threshold, far_order,
        sm, pr, pts, tris):

        va, ea = find_adjacents(tris)

    def dot(self, v):
        return v
