import numpy as np
import ndadapt
import tectosaur.util.gpu as gpu
import pycuda.driver as drv

float_type = np.float64

def get_gpu_module(rho_qx, rho_qw):
    cuda_float_type = 'float'
    if float_type == np.float64:
        cuda_float_type = 'double'
    return gpu.load_gpu('kernels.cu', tmpl_args = dict(
        rho_qx = rho_qx,
        rho_qw = rho_qw,
        float_type = cuda_float_type
    ))#, print_code = True)


def gpu_integrator(type, p, K, obs_tri, src_tri, tol, eps,
        sm, pr, rho_qx, rho_qw):
    ps = [p] * 3
    q_unmapped = ndadapt.tensor_gauss(ps)
    main_block = (32, 1, 1)

    func = get_gpu_module(rho_qx, rho_qw).get_function(type + "_integrals" + K)

    def integrator(mins, maxs):
        remaining = mins.shape[0] % main_block[0]

        out = np.zeros((mins.shape[0],81)).astype(float_type)

        def call_integrator(block, grid, start_idx, end_idx):
            if grid[0] == 0:
                return
            for chunk in range(3):
                temp_result = np.empty((end_idx-start_idx, 81)).astype(float_type)
                func(
                    drv.Out(temp_result),
                    np.int32(chunk),
                    np.int32(q_unmapped[0].shape[0]),
                    drv.In(q_unmapped[0].flatten().astype(float_type)),
                    drv.In(q_unmapped[1].astype(float_type)),
                    drv.In(mins[start_idx:end_idx].astype(float_type)),
                    drv.In(maxs[start_idx:end_idx].astype(float_type)),
                    drv.In(np.array(obs_tri).astype(float_type)),
                    drv.In(np.array(src_tri).astype(float_type)),
                    float_type(eps), float_type(sm), float_type(pr),
                    block = block, grid = grid
                )
                out[start_idx:end_idx] += temp_result

        last_large_block_idx = mins.shape[0] - remaining
        n_blocks_per_call = 512
        call_size = main_block[0] * n_blocks_per_call
        next_call_start = 0
        next_call_end = call_size
        while next_call_end < last_large_block_idx + call_size:
            this_call_end = min(next_call_end, last_large_block_idx)
            call_integrator(
                main_block,
                ((this_call_end - next_call_start) // main_block[0], 1, 1),
                next_call_start,
                this_call_end
            )
            next_call_start += call_size
            next_call_end += call_size
        call_integrator((1,1,1), (remaining, 1, 1), mins.shape[0] - remaining, mins.shape[0])
        return out[np.newaxis,:,0]
    return integrator

def new_integrate(type, K, obs_tri, src_tri, tol, eps, sm, pr, rho_qx, rho_qw):
    p = 3
    integrator = gpu_integrator(
        type, p, K, obs_tri, src_tri, tol, eps,
        sm, pr, rho_qx, rho_qw
    )
    result, count = ndadapt.hadapt_nd_iterative(
        integrator, (0,0,0), (1,1,1), tol,
        quiet = False,
        # max_refinements = 7
    )
    return result
