import numpy as np

from tectosaur.quadrature import gauss2d_tri
from tectosaur.adjacency import find_adjacents
from tectosaur.geometry import tri_normal, linear_basis_tri, tri_pt
from tectosaur.util.timer import Timer

import cppimport
taylor_integrals = cppimport.imp("tectosaur.taylor_integrals").taylor_integrals

def calc_near_obs_dir(pts, tris, va, ea, q):
    # find offset direction for each quadrature point
    # TODO: similar to some stuff in interp_galerkin_mat
    # TODO: adjust offset direction depending on adjacent elements
    # --- the quadrature order should also adjust depending on how
    # far i can get the expansion point away from an element
    tri_pts = pts[tris]
    unscaled_normals = np.cross(
        tri_pts[:,2,:] - tri_pts[:,0,:],
        tri_pts[:,2,:] - tri_pts[:,1,:]
    )
    jacobians = np.linalg.norm(unscaled_normals, axis = 1)
    offset_dir = unscaled_normals / np.sqrt(jacobians)[:,np.newaxis]
    offset_dir = np.tile(offset_dir[:,np.newaxis,:], (1, q[0].shape[0], 1))
    return offset_dir




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
        near_vert_adj_order, near_no_touch_order, near_no_touch_threshold, far_order,
        sm, pr, pts, tris):

        t = Timer()

        near_obs_q = gauss2d_tri(near_obs_order)

        v_adj, e_adj = find_adjacents(tris)
        t.report("Find adjacent")

        obs_dir = calc_near_obs_dir(pts, tris, v_adj, e_adj, near_obs_q)
        t.report("Nearfield offset points")

        # singular_tris = np.array(
        #     [(ea[0], ea[1]) for ea in e_adj] +
        #     [(va[0], va[1]) for va in v_adj]
        # )

        # singular_tris = singular_tris[singular_tris[:,0].argsort()]

        co_q = gauss2d_tri(near_co_order)
        co_pairs = np.array([(i, i) for i in range(tris.shape[0])])
        t = Timer()
        result = taylor_integrals.taylor_integralsH(
            near_obs_q[0].astype(np.float32),
            near_obs_q[1].astype(np.float32),
            obs_dir.astype(np.float32),
            near_offset,
            co_q[0].astype(np.float32),
            co_q[1].astype(np.float32),
            pts.astype(np.float32),
            tris.astype(np.int32),
            co_pairs[:, 0].astype(np.int32),
            co_pairs[:, 1].astype(np.int32),
            sm, pr
        )
        t.report('taylor')
        result = result.reshape((2,3,3,3,3))
        from tectosaur.nearfield_op import coincident
        result2 = coincident(25, [0.16,0.08,0.04,0.02,0.01], 'U', 1.0, 0.25, pts, tris)
        t.report('richardson')
        import ipdb; ipdb.set_trace()

        #TODO:
        # for taylor series stuff:
        # 3) see how it works out on the gpu!

        # for edge adjacent and coincident use the taylor series + observation expansion stuff
        # for vertex adjacent use the existing tri-tri gpu routines
        # for non-touching nearfield use the existing tri-tri gpu routines
        # for the farfield use the existing tri-tri routines

    def dot(self, v):
        return v
