import attr
import numpy as np
from multiprocessing import Pool
import cloudpickle

import tectosaur.util.gpu as gpu
from tectosaur.mesh.modify import concat
from tectosaur.ops.dense_integral_op import FarfieldTriMatrix
from tectosaur.constraint_builders import continuity_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.util.timer import Timer
from tectosaur.fmm.cfg import get_gpu_module
from tectosaur.util.quadrature import gauss4d_tri

# A tikhonov regularization least squares solution via the SVD eigenvalue
# relation.
def reg_lstsq_inverse(M, alpha):
    U, eig, VT = np.linalg.svd(M)
    inv_eig = eig / (eig ** 2 + alpha ** 2)
    return (VT.T * inv_eig).dot(U.T)

def caller(data):
    f = cloudpickle.loads(data)
    return f()

def build_c2e(gpu_params, gpu_centers, gpu_R, check_r, equiv_r, cfg):
    t = Timer()
    n_surf_dofs = cfg.surf[1].shape[0] * 9

    n_nodes = gpu_R.shape[0]
    quad = gauss4d_tri(4, 4)
    gpu_result = gpu.empty_gpu((n_nodes, n_surf_dofs, n_surf_dofs), cfg.float_type)
    gpu_module = get_gpu_module(
        cfg.surf, quad, cfg.K, cfg.float_type, cfg.n_workers_per_block
    )

    block_size = 128
    n_obs = cfg.surf[1].shape[0]
    n_obs_blocks = int(np.ceil(n_obs / block_size))
    gpu_module.build_c2e(
        gpu_result, np.int32(n_nodes), np.int32(n_obs),
        gpu_centers, gpu_R,
        cfg.float_type(check_r), cfg.float_type(equiv_r),
        gpu_params,
        grid = (n_nodes, n_obs_blocks, 1),
        block = (1, block_size, 1)
    )
    result = gpu_result.get()
    t.report('build e2cs')

    data = []
    for i in range(n_nodes):
        def task(e2c = result[i], alpha = cfg.alpha):
            return reg_lstsq_inverse(e2c, alpha)
        data.append(cloudpickle.dumps(task))
    p = Pool()
    out = p.map(caller, data)
    t.report('invert for c2e')
    return out
