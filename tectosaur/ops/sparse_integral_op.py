import numpy as np

import taskloaf as tsk

from tectosaur.nearfield.nearfield_op import NearfieldIntegralOp, RegularizedNearfieldIntegralOp
from tectosaur.nearfield.table_lookup import coincident_table, adjacent_table

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

class SparseIntegralOp:
    def __init__(self, nq_vert_adjacent, nq_far, nq_near, near_threshold,
            K_name, params, pts, tris, float_type, farfield_op_type,
            obs_subset = None, src_subset = None):

        if obs_subset is None:
            obs_subset = np.arange(tris.shape[0])
        if src_subset is None:
            src_subset = np.arange(tris.shape[0])

        self.nearfield = NearfieldIntegralOp(
            pts, tris, obs_subset, src_subset,
            nq_vert_adjacent, nq_far, nq_near,
            near_threshold, K_name, params, float_type
        )

        self.farfield = farfield_op_type(
            nq_far, K_name, params, pts, tris,
            float_type, obs_subset, src_subset
        )

        self.shape = self.nearfield.shape
        self.gpu_nearfield = None

    async def nearfield_dot(self, v):
        t = Timer(output_fnc = logger.debug)
        logger.debug("start nearfield_dot")
        out = self.nearfield.dot(v)
        t.report("nearfield_dot")
        return out

    async def nearfield_no_correction_dot(self, v):
        return self.nearfield.nearfield_no_correction_dot(v)

    async def async_dot(self, tsk_w, v):
        async def call_farfield(tsk_w):
            return (await self.farfield_dot(tsk_w, v))
        async def call_nearfield(tsk_w):
            return (await self.nearfield_dot(v))
        near_t = tsk.task(tsk_w, call_nearfield)
        far_t = tsk.task(tsk_w, call_farfield)
        far_out = await far_t
        near_out = await near_t
        return near_out + far_out

    def dot(self, v):
        async def wrapper(tsk_w):
            return await self.async_dot(tsk_w, v)
        return tsk.run(wrapper)

    async def farfield_dot(self, tsk_w, v):
        return (await self.farfield.async_dot(tsk_w, v))

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
        self.gpu_nearfield = None

    async def nearfield_dot(self, v):
        t = Timer(output_fnc = logger.debug)
        logger.debug("start nearfield_dot")
        out = self.nearfield.dot(v)
        t.report("nearfield_dot")
        return out

    async def nearfield_no_correction_dot(self, v):
        return self.nearfield.nearfield_no_correction_dot(v)

    async def async_dot(self, tsk_w, v):
        async def call_farfield(tsk_w):
            return (await self.farfield_dot(tsk_w, v))
        async def call_nearfield(tsk_w):
            return (await self.nearfield_dot(v))
        near_t = tsk.task(tsk_w, call_nearfield)
        far_t = tsk.task(tsk_w, call_farfield)
        far_out = await far_t
        near_out = await near_t
        return near_out + far_out

    def dot(self, v):
        async def wrapper(tsk_w):
            return await self.async_dot(tsk_w, v)
        return tsk.run(wrapper)

    async def farfield_dot(self, tsk_w, v):
        return (await self.farfield.async_dot(tsk_w, v))
