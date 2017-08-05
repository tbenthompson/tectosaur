import os
import pytest
import numpy as np
import tectosaur.util.gpu as gpu

class FakePlatform:
    def __init__(self, name):
        self.name = name

class FakeContextBuilder:
    def __init__(self, platforms):
        self.env_var = False
        self.platforms = [FakePlatform(name) for name in platforms]

    def get_platforms(self):
        return self.platforms

    def from_platform(self, platform, device_idx):
        return platform.name + str(device_idx)

    def check_for_env_variable(self):
        return self.env_var

    def make_any_context(self):
        return 'ANY'

def test_no_platforms():
    with pytest.raises(Exception):
        gpu.make_default_ctx(FakeContextBuilder([]))

def test_env_var():
    ctxb = FakeContextBuilder([]);
    ctxb.env_var = True
    assert(gpu.make_default_ctx(ctxb) == 'ANY')

def test_cuda_platform():
    assert(gpu.make_default_ctx(FakeContextBuilder(['CUDA'])) == 'CUDA0')

def test_cuda_platform2():
    assert(gpu.make_default_ctx(FakeContextBuilder(['CPU', 'CUDA'])) == 'CUDA0')

def test_no_cuda():
    assert(gpu.make_default_ctx(FakeContextBuilder(['CPU'])) == 'CPU0')

def test_gpu_compare_simple():
    assert(gpu.compare(1, 1))
    assert(not gpu.compare(1, 2))

def test_gpu_compare_list():
    A = list(range(10))
    B = list(range(11))
    assert(gpu.compare(A, A))
    assert(not gpu.compare(A, B))

def test_gpu_compare_numpy():
    A = np.arange(10)
    B = np.arange(11)
    assert(gpu.compare(A, A))
    assert(not gpu.compare(A, B))

def test_simple_module():
    n = 10
    in_arr = np.random.rand(n)
    arg = 1.0;
    this_dir = os.path.dirname(os.path.realpath(__file__))
    module = gpu.load_gpu('ocl_kernel.cl', tmpl_dir = this_dir, tmpl_args = dict(arg = arg))
    fnc = module.add

    in_gpu = gpu.to_gpu(in_arr, np.float32)
    out_gpu = gpu.empty_gpu(n, np.float32)
    ev = fnc(gpu.gpu_queue, (n,), None, out_gpu.data, in_gpu.data)
    ev.wait()
    output = out_gpu.get()

    correct = in_arr + arg
    np.testing.assert_almost_equal(correct, output)
