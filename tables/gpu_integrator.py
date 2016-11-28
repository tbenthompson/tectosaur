import numpy as np
import ndadapt
import tectosaur.util.gpu as gpu
import pycuda.driver as drv

def get_gpu_module(n_rho, taylor_order, use_taylor):
    return gpu.load_gpu('kernels.cu', tmpl_args = dict(
            n_rho = n_rho, taylor_order = taylor_order, use_taylor = use_taylor
        ))#, print_code = True)


float_type = np.float32
def gpu_integrator(type, p, K, obs_tri, src_tri, tol, eps,
        sm, pr, rho_qx, rho_qw, taylor_order):
    ps = [p] * 3
    q_unmapped = ndadapt.tensor_gauss(ps)
    main_block = (32, 1, 1)

    if taylor_order > 0:
        use_taylor = True
    else:
        use_taylor = False

    module = get_gpu_module(rho_qx.shape[0], taylor_order, use_taylor)

    funcs = [
        module.get_function(type + '_integrals' + K + str(chunk))
        for chunk in range(3)
    ]

    def integrator(mins, maxs):
        remaining = mins.shape[0] % main_block[0]
        grid_main = (mins.shape[0] // main_block[0], 1, 1)
        grid_rem = (remaining, 1, 1)

        out = np.empty((mins.shape[0],81)).astype(float_type)
        out_rem = np.empty((grid_rem[0],81)).astype(float_type)


        def call_integrator(block, grid, result_buf, start_idx, end_idx):
            if grid[0] == 0:
                return
            result_buf[:] = 0
            for chunk in range(3):
                temp_result = np.empty_like(result_buf).astype(float_type)
                funcs[chunk](
                    drv.Out(temp_result),
                    np.int32(q_unmapped[0].shape[0]),
                    drv.In(q_unmapped[0].flatten().astype(float_type)),
                    drv.In(q_unmapped[1].astype(float_type)),
                    drv.In(mins[start_idx:end_idx].astype(float_type)),
                    drv.In(maxs[start_idx:end_idx].astype(float_type)),
                    drv.In(np.array(obs_tri).astype(float_type)),
                    drv.In(np.array(src_tri).astype(float_type)),
                    float_type(eps), float_type(sm), float_type(pr),
                    drv.In(rho_qx.astype(float_type)),
                    drv.In(rho_qw.astype(float_type)),
                    np.int32(block[0]),
                    block = block, grid = grid
                )
                result_buf += temp_result

        call_integrator(main_block, grid_main, out, 0, mins.shape[0] - remaining)
        call_integrator((1,1,1), grid_rem, out_rem, mins.shape[0] - remaining, mins.shape[0])
        out[(mins.shape[0] - remaining):] = out_rem
        return out
    return integrator

def new_integrate(type, K, obs_tri, src_tri, tol, eps, sm, pr, rho_qx, rho_qw, taylor_order):
    p = 7
    integrator = gpu_integrator(
        type, p, K, obs_tri, src_tri, tol, eps,
        sm, pr, rho_qx, rho_qw, taylor_order
    )
    result, count = ndadapt.hadapt_nd_iterative(
        integrator, (0,0,0), (1,1,1), tol,
        quiet = False,
        # max_refinements = 7
    )
    return result
