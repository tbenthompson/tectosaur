import tectosaur.util.gpu as gpu

import os, sys
here_dir = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(here_dir)

import cppimport
vmv = cppimport.imp("vienna_mat_vec")

setup = vmv.setup
check_platform = vmv.check_platform

def prod(A, x, float_type):
    A_gpu = gpu.to_gpu(A, float_type)
    x_gpu = gpu.to_gpu(x, float_type)

    A_ip = A_gpu.data.int_ptr
    x_ip = x_gpu.data.int_ptr
    out = vmv.mat_vec(A_ip, x_ip, A.shape[0], A.shape[1])
    return out
