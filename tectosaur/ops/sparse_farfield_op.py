import attr
import numpy as np
import scipy.sparse

import tectosaur.fmm.fmm as fmm
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

        q = gauss4d_tri(nq_far, nq_far)
        self.gpu_qx = gpu.to_gpu(q[0], float_type)
        self.gpu_qw = gpu.to_gpu(q[1], float_type)
        self.nq = self.gpu_qx.shape[0]

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
                float_type = gpu.np_to_c_type(float_type)
            )
        )
        self.fnc = getattr(self.module, "farfield_tris" + K_name)

    def dot(self, v):
        self.gpu_in[:] = v[:].astype(self.gpu_in.dtype)
        self.fnc(
            self.gpu_out, self.gpu_in,
            self.gpu_pts, self.gpu_obs_tris, self.gpu_src_tris,
            self.gpu_qx, self.gpu_qw, self.gpu_params,
            np.int32(self.n_obs), np.int32(self.n_src), np.int32(self.nq),
            grid = (self.n_blocks, 1, 1), block = (self.block_size, 1, 1)
        )
        return self.gpu_out.get()

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
    alpha = attr.ib(default = 1e-5)
    def __call__(self, nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset):
        return FMMFarfieldOpImpl(
            nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset, self.mac, self.pts_per_cell, self.alpha
        )

class FMMFarfieldOpImpl:
    def __init__(self, nq_far, K_name, params, pts, tris, float_type,
            obs_subset, src_subset, mac, pts_per_cell, alpha):

        cfg = fmm.make_config(
            K_name, params, 1.1, mac, 2, nq_far, float_type,
            alpha = alpha
        )

        m_obs = (pts, tris[obs_subset])
        m_src = (pts, tris[src_subset])
        self.obs_tree = fmm.make_tree(m_obs, cfg, pts_per_cell)
        self.src_tree = fmm.make_tree(m_src, cfg, pts_per_cell)
        self.fmm_obj = fmm.FMM(
            self.obs_tree, m_obs,
            self.src_tree, m_src,
            cfg
        )
        self.evaluator = fmm.FMMEvaluator(self.fmm_obj)

    def dot(self, v):
        t = Timer(output_fnc = logger.debug)
        v_tree = self.fmm_obj.to_tree(v)
        t.report('to tree space')

        fmm_out = self.evaluator.eval(tsk_w, v_tree)
        t.report('fmm eval')

        out = self.fmm_obj.to_orig(fmm_out)
        t.report('to output space')
        return out

    def nearfield_dot(self, v):
        return self.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.dot(v)

    def farfield_dot(self, v):
        return self.dot(v)
