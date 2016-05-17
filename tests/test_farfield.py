import cppimport
import numpy as np
import tectosaur.farfield

def normalize(vs):
    return vs / np.linalg.norm(vs, axis = 1).reshape((vs.shape[0], 1))

def test_farfield():
    n = 8000
    obs_pts = np.random.rand(n, 3).astype(np.float32)
    obs_ns = normalize(np.random.rand(n, 3).astype(np.float32))
    src_pts = np.random.rand(n, 3).astype(np.float32)
    src_ns = obs_ns

    import pycuda.autoinit
    import pycuda.driver as drv
    import numpy

    from pycuda.compiler import SourceModule
    code = open('tectosaur/farfield.cu', 'r').read()
    mod = SourceModule(code)
    farfield = mod.get_function("farfield")

    result = np.zeros((n, n, 3, 3)).astype(np.float32)

    block = (8, 4, 1)
    grid = (int(n / block[0]), int(n / block[1]))
    print(grid)
    print(block)
    print(farfield(
        drv.Out(result), drv.In(obs_pts), drv.In(obs_ns),
        drv.In(src_pts), drv.In(src_ns), np.float32(1.0),
        np.float32(0.25), np.int32(n),
        block = block,
        grid = grid,
        time_kernel = True
    ))

    # print(result)

    # import time
    # start = time.time()
    # result2 = tectosaur.farfield.farfield(obs_pts, obs_ns, src_pts, src_ns, 1.0, 0.25)
    # print(time.time() - start)
    # print(np.sum((result2 - result) ** 2) / (n * n))
    # print(xdiff)
    # print(obs_pts[:, 0])
