import numpy as np

from tectosaur.quadrature import gauss2d_tri
from tectosaur.adjacency import find_adjacents
from tectosaur.geometry import tri_normal, linear_basis_tri, tri_pt
from tectosaur.util.timer import Timer

import cppimport
taylor_integrals = cppimport.imp("tectosaur.taylor_integrals").taylor_integrals

def calc_near_obs_pts(pts, tris, va, ea, q, offset):
    # find real coordinates for each quadrature point
    qx = q[0]
    basis = np.array([1 - qx[:, 0] - qx[:, 1], qx[:, 0], qx[:, 1]])
    tri_pts = pts[tris]

    t = Timer()
    # TODO: Some of this crazy vectorization is identical to stuff in interp_galerkin_mat
    obs_pts = np.sum(
        np.tile(tri_pts[:,:,np.newaxis,:],(1,1,qx.shape[0],1)) *
            np.tile(basis[np.newaxis,:,:,np.newaxis], (tri_pts.shape[0],1,1,3)),
        axis = 1
    )
    t.report('obs_pts')

    # find offset direction for each quadrature point
    # TODO: adjust offset direction depending on adjacent elements
    # --- the quadrature order should also adjust depending on how far i can get the expansion point away from an element
    unscaled_normals = np.cross(
        tri_pts[:,2,:] - tri_pts[:,0,:],
        tri_pts[:,2,:] - tri_pts[:,1,:]
    )
    jacobians = np.linalg.norm(unscaled_normals, axis = 1)
    offset_dir = unscaled_normals / np.sqrt(jacobians)[:,np.newaxis]

    # calculate offset points (surf_pt + near_offset * offset_dir)
    offset_pts = obs_pts + offset * offset_dir[:,np.newaxis,:]
    return offset_pts




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

        near_obs_pts = calc_near_obs_pts(pts, tris, v_adj, e_adj, near_obs_q, near_offset)
        t.report("Nearfield offset points")

        # singular_tris = np.array(
        #     [(ea[0], ea[1]) for ea in e_adj] +
        #     [(va[0], va[1]) for va in v_adj]
        # )

        # singular_tris = singular_tris[singular_tris[:,0].argsort()]

        co_q = gauss2d_tri(near_co_order)
        co_pairs = np.array([(i, i) for i in range(tris.shape[0])])
        result = taylor_integrals.taylor_integralsH(
            near_obs_q[0], near_obs_q[1], near_obs_q[1],
            co_q[0], co_q[1],
            pts, tris, co_pairs[:, 0], co_pairs[:, 1],
            sm, pr
        )

        #TODO:
        # for taylor series stuff:
        # 1) rewrite a simple version of the elastic kernels (use the work i did for the optimized gpu pt-pt version)
        # 2) operator overloading/C++ taylor series implementation. extract it from the old 3bem stuff.
        # 3) see how it works out on the gpu!

        # for edge adjacent and coincident use the taylor series + observation expansion stuff
        # for vertex adjacent use the existing tri-tri gpu routines
        # for non-touching nearfield use the existing tri-tri gpu routines
        # for the farfield use the existing tri-tri routines

    def dot(self, v):
        return v
