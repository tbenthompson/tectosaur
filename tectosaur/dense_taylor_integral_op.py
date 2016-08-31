import numpy as np

from tectosaur.quadrature import gauss2d_tri
from tectosaur.find_nearfield import find_nearfield
from tectosaur.adjacency import find_adjacents, vert_adj_prep, edge_adj_prep, rotate_tri
from tectosaur.geometry import tri_normal, linear_basis_tri, tri_pt
from tectosaur.nearfield_op import pairs_quad, vert_adj
from tectosaur.dense_integral_op import farfield, set_co_entries, set_adj_entries, set_near_entries
from tectosaur.quadrature import gauss4d_tri
from tectosaur.util.timer import Timer
from tectosaur.triangle_rules import coincident_quad

import cppimport
taylor_integrals = cppimport.imp("tectosaur.taylor_integrals").taylor_integrals

floatt = np.float64

def pairs_taylor_quad(quad, offset, pts, tris, obs_tris, src_tris, kernel, sm, pr):
    return getattr(taylor_integrals, 'taylor_integrals' + kernel)(
        quad[0].astype(floatt),
        quad[1].astype(floatt),
        offset,
        pts.astype(floatt),
        tris.astype(np.int32),
        obs_tris.astype(np.int32),
        src_tris.astype(np.int32),
        sm, pr
    )

# parameters:
# nearfield offset (ratio of element length)
# nearfield obs gauss order
# nearfield coincident gauss order
# nearfield edge adjacent gauss order
# nearfield vertex adjacent gauss order
# nearfield non-touching gauss order
# nearfield non-touching threshold
# farfield gauss order
class DenseTaylorIntegralOp:
    def __init__(self, near_offset, near_obs_order, near_co_order, near_edge_adj_order,
        near_vert_adj_order, near_order, near_threshold, far_order,
        kernel, sm, pr, pts, tris):

        t = Timer(silent = True)

        # co_q = gauss4d_tri(near_obs_order, near_co_order)
        co_q = coincident_quad(near_offset, near_obs_order, near_obs_order, near_co_order, near_co_order)
        e_adj_q = gauss4d_tri(near_obs_order, near_edge_adj_order)
        near_q = gauss4d_tri(near_obs_order, near_order)
        t.report('Make quadrature')

        va, ea = find_adjacents(tris)
        t.report("Find adjacent")

        ea_tri_indices, ea_obs_clicks, ea_src_clicks, ea_obs_tris, ea_src_tris =\
            edge_adj_prep(tris, ea)
        t.report('Edge adjacency prep')

        va_tri_indices, va_obs_clicks, va_src_clicks, va_obs_tris, va_src_tris =\
            vert_adj_prep(tris, va)
        t.report("Vert adjacency prep")

        co_indices = np.arange(tris.shape[0])
        co_mat = pairs_taylor_quad(
            co_q, near_offset, pts, tris, co_indices, co_indices, kernel, sm, pr
        ).reshape((co_indices.shape[0], 3, 3, 3, 3))
        print(co_mat[0,0,0,0,0])
        t.report('Coincident')

        # ea_mat_rot = pairs_taylor_quad(
        #     e_adj_q, near_offset, pts, tris,
        #     ea_tri_indices[:,0], ea_tri_indices[:,1], kernel, sm, pr
        # ).reshape((ea_tri_indices.shape[0], 3, 3, 3, 3))
        # t.report('Edge adjacent')

        va_mat_rot = vert_adj(near_vert_adj_order, kernel, sm, pr, pts, va_obs_tris, va_src_tris)
        t.report("Vert adjacent")

        nearfield_pairs = np.array(find_nearfield(pts, tris, va, ea, near_threshold))
        if nearfield_pairs.size == 0:
            nearfield_pairs = np.array([], dtype = np.int).reshape(0,2)
        nearfield_mat = pairs_quad(
            kernel, sm, pr, pts, tris[nearfield_pairs[:,0]], tris[nearfield_pairs[:, 1]],
            near_q, False
        )
        t.report("Nearfield")

        out = farfield(kernel, sm, pr, pts, tris, tris, far_order)
        out = out.astype(floatt)
        t.report("Farfield")

        out = set_co_entries(out, co_mat, co_indices)
        # out = set_adj_entries(
        #     out, ea_mat_rot, ea_tri_indices, ea_obs_clicks, ea_src_clicks
        # )
        # out = set_adj_entries(
        #     out, va_mat_rot, va_tri_indices, va_obs_clicks, va_src_clicks
        # )
        # out = set_near_entries(out, nearfield_mat, nearfield_pairs)
        t.report("Insert coincident nearfield")

        out.shape = (
            out.shape[0] * out.shape[1] * out.shape[2],
            out.shape[3] * out.shape[4] * out.shape[5]
        )

        self.mat = out
        self.shape = self.mat.shape
        self.gpu_mat = None

        #TODO:
        # for taylor series stuff:
        # 3) see how it works out on the gpu!

        # for edge adjacent and coincident use the taylor series + observation expansion stuff
        # for vertex adjacent use the existing tri-tri gpu routines
        # for non-touching nearfield use the existing tri-tri gpu routines
        # for the farfield use the existing tri-tri routines

    def dot(self, v):
        raise Exception("BAD")
