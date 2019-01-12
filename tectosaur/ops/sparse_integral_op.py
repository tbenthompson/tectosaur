import numpy as np

from tectosaur.nearfield.nearfield_op import NearfieldIntegralOp, RegularizedNearfieldIntegralOp

from tectosaur.util.timer import Timer

import logging
logger = logging.getLogger(__name__)

# Handy wrapper for creating an integral op
def make_integral_op(pts, tris, k_name, k_params, cfg, obs_subset, src_subset):
    if cfg['use_fmm']:
        farfield = FMMFarfieldBuilder(
            cfg['fmm_order'], cfg['fmm_mac'], cfg['pts_per_cell']
        )
    else:
        farfield = None
    return SparseIntegralOp(
        cfg['quad_vertadj_order'], cfg['quad_far_order'],
        cfg['quad_near_order'], cfg['quad_near_threshold'],
        k_name, k_params, pts, tris, cfg['float_type'],
        farfield_op_type = farfield,
        obs_subset = obs_subset,
        src_subset = src_subset
    )

class RegularizedSparseIntegralOp:
    def __init__(self, nq_coincident, nq_edge_adj, nq_vert_adjacent,
            nq_far, nq_near, near_threshold,
            K_near_name, K_far_name, params, pts, tris, float_type, farfield_op_type,
            obs_subset = None, src_subset = None):

        if obs_subset is None:
            obs_subset = np.arange(tris.shape[0])
        if src_subset is None:
            src_subset = np.arange(tris.shape[0])

        self.nearfield = RegularizedNearfieldIntegralOp(
            pts, tris, obs_subset, src_subset,
            nq_coincident, nq_edge_adj, nq_vert_adjacent, nq_far, nq_near,
            near_threshold, K_near_name, K_far_name,
            params, float_type
        )

        self.farfield = farfield_op_type(
            nq_far, K_far_name, params, pts, tris,
            float_type, obs_subset, src_subset
        )

        self.shape = self.nearfield.shape

    async def nearfield_dot(self, v):
        t = Timer(output_fnc = logger.debug)
        logger.debug("start nearfield_dot")
        out = self.nearfield.dot(v)
        t.report("nearfield_dot")
        return out

    def nearfield_no_correction_dot(self, v):
        return self.nearfield.nearfield_no_correction_dot(v)

    def dot(self, v):
        import asyncio
        loop = asyncio.new_event_loop()
        async def dot_helper():
            yfar = asyncio.ensure_future(self.farfield_dot(v))
            ynear = asyncio.ensure_future(self.nearfield_dot(v))
            return (await yfar) + (await ynear)
        out = loop.run_until_complete(dot_helper())
        loop.close()

        return out

    async def farfield_dot(self, v):
        t = Timer(output_fnc = logger.debug)
        logger.debug("start farfield_dot")
        out = await self.farfield.async_dot(v)
        t.report('farfield_dot')
        return out
