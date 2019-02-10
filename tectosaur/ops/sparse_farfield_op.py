import attr
import numpy as np
import scipy.sparse

from tectosaur.fmm.tsfmm import TSFMM
import tectosaur.util.geometry as geometry
import tectosaur.util.gpu as gpu
from tectosaur.farfield import farfield_pts_direct
from tectosaur.util.quadrature import gauss2d_tri, gauss4d_tri
from tectosaur.util.timer import Timer
from tectosaur.kernels import kernels

import logging
logger = logging.getLogger(__name__)

class TriToTriDirectFarfieldOp:
    def __init__(self, nq_far, K_name, params, pts, tris,
            float_type, obs_subset, src_subset):
        self.shape = (obs_subset.shape[0] * 9, src_subset.shape[0] * 9)
        self.dim = pts.shape[1]
        self.tensor_dim = kernels[K_name].tensor_dim
        self.n_obs = obs_subset.shape[0]
        self.n_src = src_subset.shape[0]

        in_size = self.n_src * self.dim * self.tensor_dim
        out_size = self.n_obs * self.dim * self.tensor_dim
        self.gpu_in = gpu.empty_gpu(in_size, float_type)
        self.gpu_out = gpu.empty_gpu(out_size, float_type)

        self.q = gauss2d_tri(nq_far)

        self.gpu_pts = gpu.to_gpu(pts, float_type)
        self.gpu_obs_tris = gpu.to_gpu(tris[obs_subset], np.int32)
        self.gpu_src_tris = gpu.to_gpu(tris[src_subset], np.int32)
        self.gpu_params = gpu.to_gpu(np.array(params), float_type)
        self.block_size = 128
        self.n_blocks = int(np.ceil(self.n_obs / self.block_size))

        self.module = gpu.load_gpu(
            'farfield_tris.cl',
            tmpl_args = dict(
                block_size = self.block_size,
                float_type = gpu.np_to_c_type(float_type),
                quad_pts = self.q[0],
                quad_wts = self.q[1]
            )
        )
        self.fnc = getattr(self.module, "farfield_tris_to_tris" + K_name)

    def dot(self, v):
        self.gpu_in[:] = v[:].astype(self.gpu_in.dtype)
        self.fnc(
            self.gpu_out, self.gpu_in,
            self.gpu_pts, self.gpu_obs_tris, self.gpu_src_tris,
            self.gpu_params,
            np.int32(self.n_obs), np.int32(self.n_src),
            grid = (self.n_blocks, 1, 1), block = (self.block_size, 1, 1)
        )
        return self.gpu_out.get()

    async def async_dot(self, v):
        return self.dot(v)

    def nearfield_dot(self, v):
        return self.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.dot(v)

    def farfield_dot(self, v):
        return self.dot(v)

@attr.s()
class FMMFarfieldOp:
    mac = attr.ib()
    pts_per_cell = attr.ib()
    order = attr.ib()
    def __call__(self, nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset):
        return FMMFarfieldOpImpl(
            nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset, self.mac, self.pts_per_cell, self.order
        )

class FMMFarfieldOpImpl:
    def __init__(self, nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset, mac, pts_per_cell, order):

        L_scale = np.max(pts)
        scaled_pts = pts / L_scale
        m_obs = (scaled_pts, tris[obs_subset].copy())
        m_src = (scaled_pts, tris[src_subset].copy())

        self.L_factor = L_scale ** (-kernels[K_name].scale_type)

        self.fmm = TSFMM(
            m_obs, m_src, params = params, order = order,
            quad_order = nq_far, float_type = float_type,
            K_name = K_name,
            mac = mac, max_pts_per_cell = pts_per_cell,
            n_workers_per_block = 128
        )

    def dot(self, v):
        t = Timer(output_fnc = logger.debug)
        out = self.fmm.dot(v)
        t.report('fmm eval')
        return self.L_factor * out

    async def async_dot(self, v):
        return self.L_factor * (await self.fmm.async_dot(v))

    def nearfield_dot(self, v):
        return self.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.dot(v)

    def farfield_dot(self, v):
        return self.dot(v)
