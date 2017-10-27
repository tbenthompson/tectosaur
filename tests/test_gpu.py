import os
import pytest
import numpy as np
from threading import Thread
import tectosaur.util.gpu as gpu
import tectosaur.util.opencl as opencl

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
    module = gpu.load_gpu('kernel.cl', tmpl_dir = this_dir, tmpl_args = dict(arg = arg))
    fnc = module.add

    in_gpu = gpu.to_gpu(in_arr, np.float32)
    out_gpu = gpu.empty_gpu(n, np.float32)
    fnc(out_gpu, in_gpu, grid = (n,1,1), block = (1,1,1))
    output = out_gpu.get()

    correct = in_arr + arg
    np.testing.assert_almost_equal(correct, output)

def test_threaded_get():
    R = np.random.rand(10)
    gpu_R = gpu.to_gpu(R, np.float32)
    def f():
        f.R2 = gpu.threaded_get(gpu_R)
    f.R2 = None
    thread = Thread(target = f)
    thread.start()
    thread.join()
    np.testing.assert_almost_equal(R, f.R2)
