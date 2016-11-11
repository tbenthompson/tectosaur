import numpy as np
import ndadapt
import pycuda.driver as drv

gpu_module = gpu.load_gpu('kernels.cu', print_code = True)

def get_gpu_integrator(p, K, tri, tol, eps, sm, pr, rho_qx, rho_qw):
    ps = [p] * 3
    q_unmapped = ndadapt.tensor_gauss(ps)
    def integrator(mins, maxs):
        block = (32, 1, 1)
        remaining = mins.shape[0] % block[0]
        grid_main = (mins.shape[0] // block[0], 1, 1)
        grid_rem = (remaining, 1, 1)

        out = np.empty(mins.shape[0]).astype(np.float64)
        out_rem = np.empty(grid_rem[0]).astype(np.float64)

        def call_integrator(block, grid, result_buf, start_idx, end_idx):
            if grid[0] == 0:
                return
            result_buf = 0
            for chunk in range(3):
                temp_result = np.empty_like(result_buf)
                gpu_module.get_function('compute_integrals' + str(chunk))(
                    drv.Out(temp_result),
                    np.int32(q_unmapped[0].shape[0]),
                    drv.In(q_unmapped[0].astype(np.float64)),
                    drv.In(q_unmapped[1].astype(np.float64)),
                    drv.In(mins[start_idx:end_idx].astype(np.float64)),
                    drv.In(maxs[start_idx:end_idx].astype(np.float64)),
                    block = block, grid = grid
                )
                result_buf += temp_result

        call_integrator(block, grid_main, out, 0, mins.shape[0] - remaining)
        call_integrator(
            (1,1,1), grid_rem, out_rem, mins.shape[0] - remaining, mins.shape[0]
        )
        out[(mins.shape[0] - remaining):] = out_rem
        # out2 = []
        # for i in range(mins.shape[0]):
        #     q = map_to(q_unmapped, mins[i,:], maxs[i,:])
        #     out2.append(np.sum(f(q[0]) * q[1]))
        # # np.testing.assert_almost_equal(out2, out, 2)
        # return out2
        return out
    return integrator

def new_integrate_coincident(K, tri, tol, eps, sm, pr, rho_qx, rho_qw):
    p = 7
    integrator = get_gpu_integrator(p, K, tri, tol, eps, sm, pr, rho_qx, rho_qw)
    result = ndadapt.hadapt_nd_iterative(integrator, (0,0,0), (1,1,1), tol)
