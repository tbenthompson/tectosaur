import numpy as np
import tectosaur.util.gpu as gpu

#TODO: Replicated in math_tools.hpp
def to_interval(a, b, x):
    return a + (b - a) * (x + 1.0) / 2.0

#TODO: Replicated in math_tools.hpp
def from_interval(a, b, x):
    return ((x - a) / (b - a)) * 2.0 - 1.0

"""Chebyshev nodes in the [a,b] interval. Good for interpolation. Avoids Runge phenomenon."""
def cheb(a, b, n):
    out = []
    for i in range(n):
        out.append(to_interval(a, b, np.cos(((2 * i + 1) * np.pi) / (2 * n))))
    return out

def cheb_wts(a, b, n):
    j = np.arange(n)
    return ((-1) ** j) * np.sin(((2 * j + 1) * np.pi) / (2 * n))

def cheblob(a, b, n):
    return to_interval(a, b, np.cos((np.arange(n) * np.pi) / (n - 1)))

def cheblob_wts(a, b, n):
    wts = ((-1) ** np.arange(n)) * np.ones(n)
    wts[0] *= 0.5
    wts[-1] *= 0.5
    return wts

# pts should be shaped (n,p)
# wts should be shaped (n,)
# vals should be shaped (n,q)
# xhat should be shaped (m,p)
# output is shaped (m,q)
# n is the number of interpolation pts
# m is the number of evaluation pts
# p is the number of input dimensions
# q is the number of output dimensions
def barycentric_evalnd(pts, wts, vals, xhat, float_type):
    n_interp_pts, n_input_dims = pts.shape
    n_eval_pts = xhat.shape[0]
    n_output_dims = vals.shape[1]

    block_size = 128
    gpu_cfg = dict(
        np_float_type = float_type,
        float_type = gpu.np_to_c_type(float_type),
        block_size = block_size,
        n_input_dims = n_input_dims
    )
    module = gpu.load_gpu('nearfield/interpolate_kernel.cl', tmpl_args = gpu_cfg)
    fnc = module.interpolate

    gpu_pts = gpu.to_gpu(pts, float_type)
    gpu_wts = gpu.to_gpu(wts, float_type)
    gpu_vals = gpu.to_gpu(vals, float_type)
    gpu_xhat = gpu.to_gpu(xhat, float_type)

    gpu_result = gpu.empty_gpu((n_eval_pts, n_output_dims), float_type)

    n_threads = int(np.ceil(n_eval_pts / block_size))

    fnc(
        gpu_result,
        np.int32(n_interp_pts), np.int32(n_eval_pts), np.int32(n_output_dims),
        gpu_pts, gpu_wts, gpu_vals, gpu_xhat,
        grid = (n_threads, 1, 1),
        block = (block_size, 1, 1)
    )

    return gpu_result.get()

if __name__ == '__main__':
    # test_barycentric_interp()
    # test_barycentric_interp2d()
    test_barycentric_interp3d()
