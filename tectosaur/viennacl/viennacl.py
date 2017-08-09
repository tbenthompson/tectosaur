import numpy as np
import tectosaur.util.gpu as gpu

from cppimport import cppimport
wrapper = cppimport("tectosaur.viennacl.vienna_wrapper")

setup = wrapper.setup
check_platform = wrapper.check_platform

def prod(A, x, float_type):
    float_char = 'd' if float_type == np.float64 else 's'
    assert(x.shape[0] == A.shape[1])
    A_gpu = gpu.to_gpu(A, float_type)
    x_gpu = gpu.to_gpu(x, float_type)

    A_ip = A_gpu.data.int_ptr
    x_ip = x_gpu.data.int_ptr
    if len(x_gpu.shape) > 1:
        C_gpu = gpu.zeros_gpu((A.shape[0], x.shape[1]), float_type)
        C_ip = C_gpu.data.int_ptr
        fnc = getattr(wrapper, 'mat_mat_prod_' + float_char)
        fnc(A_ip, x_ip, C_ip, A.shape[0], A.shape[1], x.shape[1])
        return C_gpu
    else:
        y_gpu = gpu.zeros_gpu(A.shape[0], float_type)
        y_ip = y_gpu.data.int_ptr
        fnc = getattr(wrapper, 'mat_vec_prod_' + float_char)
        out = fnc(A_ip, x_ip, y_ip, A.shape[0], A.shape[1])
        return y_gpu

def qr_solve(A, B, float_type):
    float_char = 'd' if float_type == np.float64 else 's'
    m = A.shape[0]
    r = 1 if len(B.shape) == 1 else B.shape[1]
    assert(B.shape[0] == A.shape[1])

    # grow A to be a multiple of 16 in size (viennacl bug?)
    m_grown = 16 * int(np.ceil(m / 16))
    A_grown = np.zeros((m_grown, m_grown), float_type)
    A_grown[:m,:m] = A

    A_gpu = gpu.to_gpu(A_grown, float_type)
    B_gpu = gpu.to_gpu(B, float_type)
    C_gpu = gpu.zeros_gpu((m_grown, r), float_type)

    A_ip = A_gpu.data.int_ptr
    B_ip = B_gpu.data.int_ptr
    C_ip = C_gpu.data.int_ptr

    fnc = getattr(wrapper, 'qr_solve_' + float_char)
    out = fnc(A_ip, B_ip, C_ip, m_grown, r)
    return out, A_grown
    return C_gpu
