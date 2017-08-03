import tectosaur.util.gpu as gpu

from cppimport import cppimport
vmv = cppimport("tectosaur.viennacl.vienna_mat_vec")

setup = vmv.setup
check_platform = vmv.check_platform

def prod(A, x, float_type):
    assert(x.shape[0] == A.shape[1])
    A_gpu = gpu.to_gpu(A, float_type)
    x_gpu = gpu.to_gpu(x, float_type)

    A_ip = A_gpu.data.int_ptr
    x_ip = x_gpu.data.int_ptr
    if len(x_gpu.shape) > 1:
        C_gpu = gpu.zeros_gpu((A.shape[0], x.shape[1]), float_type)
        C_ip = C_gpu.data.int_ptr
        vmv.mat_mat_prod(A_ip, x_ip, C_ip, A.shape[0], A.shape[1], x.shape[1])
        return C_gpu
    else:
        y_gpu = gpu.zeros_gpu(A.shape[0], float_type)
        y_ip = y_gpu.data.int_ptr
        out = vmv.mat_vec_prod(A_ip, x_ip, y_ip, A.shape[0], A.shape[1])
        return y_gpu
