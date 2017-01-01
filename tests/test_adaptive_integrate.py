import numpy as np

import tectosaur.quadrature as quad
import cppimport
adaptive_integrate = cppimport.imp('tectosaur.adaptive_integrate').adaptive_integrate
adaptive_integrate2 = cppimport.imp('tectosaur.adaptive_integrate2').adaptive_integrate2

from test_decorators import golden_master, slow
import tectosaur.util.gpu as gpu

# Move this stuff to gpu.py
float_type = np.float64
def to_gpu(arr):
    return gpu.cl.array.to_device(gpu.ocl_gpu_queue, arr.astype(float_type))

def empty_gpu(shape):
    return gpu.cl.array.empty(gpu.ocl_gpu_queue, shape, float_type)

def quad_to_gpu(quad_rule):
    gpu_qx = to_gpu(quad_rule[0].flatten())
    gpu_qw = to_gpu(quad_rule[1])
    return gpu_qx, gpu_qw
#####

def make_gpu_integrator(type, K, obs_tri, src_tri, tol, eps, sm, pr, rho_q, theta_q, chunk):
    module = gpu.ocl_load_gpu('tectosaur/kernels.cl')
    fnc = getattr(module, type + '_integrals' + K)
    gpu_rqx, gpu_rqw = quad_to_gpu(rho_q)
    gpu_tqx, gpu_tqw = quad_to_gpu(theta_q)
    gpu_obs_tri = to_gpu(np.array(obs_tri).flatten())
    gpu_src_tri = to_gpu(np.array(src_tri).flatten())

    def integrand(x):
        n_x = x.shape[0]
        integrand.total_n_x += n_x
        print(integrand.total_n_x)
        out = np.zeros((n_x,81)).astype(float_type)

        def call_integrator(start_idx, end_idx):
            n_items = end_idx - start_idx
            gpu_pts = to_gpu(x[start_idx:end_idx,:])
            gpu_result = empty_gpu((n_items, 81))
            fnc(
                gpu.ocl_gpu_queue, (n_items,), None,
                gpu_result.data, np.int32(chunk), gpu_pts.data,
                np.int32(rho_q[0].shape[0]), gpu_rqx.data, gpu_rqw.data,
                np.int32(theta_q[0].shape[0]), gpu_tqx.data, gpu_tqw.data,
                gpu_obs_tri.data, gpu_src_tri.data,
                float_type(eps), float_type(sm), float_type(pr)
            );
            out[start_idx:end_idx] = gpu_result.get()

        call_size = 2 ** 12
        next_call_start = 0
        next_call_end = call_size
        while next_call_end < n_x + call_size:
            this_call_end = min(next_call_end, n_x)
            call_integrator(next_call_start, this_call_end)
            next_call_start += call_size
            next_call_end += call_size
        return out
    integrand.total_n_x = 0

    return integrand

@slow
@golden_master
def test_coincident_integral():
    tri = [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]]
    eps = 0.01
    tol = 0.001
    K = 'H'
    rho_gauss = quad.gaussxw(50)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    theta_q = quad.gaussxw(50)
    res = np.zeros(81)
    for chunk in range(3):
        chunk_res = adaptive_integrate2.integrate(
            make_gpu_integrator(
                'coincident', K, tri, tri, tol, eps, 1.0, 0.25, rho_q, theta_q, chunk
            ),
            [0,0], [1,1], tol
        )
        res += chunk_res[0]
    return res

@slow
@golden_master
def test_adjacent_integral():
    obs_tri = [[0, 0, 0], [1.2, 0, 0], [0.3, 1.1, 0]]
    src_tri = [[1.2, 0, 0], [0, 0, 0], [0.3, -1.1, 0]]
    eps = 0.01
    tol = 0.001
    K = 'H'
    rho_gauss = quad.gaussxw(50)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    theta_q = quad.gaussxw(50)
    res = np.zeros(81)
    for chunk in range(2):
        chunk_res = adaptive_integrate2.integrate(
            make_gpu_integrator(
                'adjacent', K, obs_tri, src_tri, tol, eps, 1.0, 0.25, rho_q, theta_q, chunk
            ),
            [0,0], [1,1], tol
        )
        res += chunk_res[0]
    return np.array(res)

def f1d(x):
    return np.sin(x)

def test_adaptive_1d():
    res = adaptive_integrate2.integrate(f1d, [0], [1], 0.01)
    np.testing.assert_almost_equal(res[0], 1 - np.cos(1))

def f2d(x):
    return np.array([np.sin(x[:,0])*x[:,1]]).T

def test_adaptive_2d():
    res = adaptive_integrate2.integrate(f2d, [0,0], [1,1], 0.01)
    np.testing.assert_almost_equal(res[0], np.sin(0.5) ** 2)

def vec_out(x):
    return np.array([x[:,0] + 1, x[:,0] - 1]).T

def test_adaptive_vector_out():
    res = adaptive_integrate2.integrate(vec_out, [0], [1], 0.01)
    np.testing.assert_almost_equal(res[0], [1.5, -0.5])
