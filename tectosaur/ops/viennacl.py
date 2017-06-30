import tectosaur.util.gpu as gpu

import cppimport
vmv = cppimport.imp("tectosaur.vienna_mat_vec").vienna_mat_vec

setup = vmv.setup
check_platform = vmv.check_platform

def prod(A, x, float_type):
    A_gpu = gpu.to_gpu(A, float_type)
    x_gpu = gpu.to_gpu(x, float_type)

    A_ip = A_gpu.data.int_ptr
    x_ip = x_gpu.data.int_ptr
    out = vmv.mat_vec(A_ip, x_ip, A.shape[0], A.shape[1])
    return out
