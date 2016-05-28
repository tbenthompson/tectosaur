import cppimport
import numpy as np
import tectosaur.farfield
from tectosaur.gpu import load_gpu
import tectosaur.quadrature as quad
import tectosaur.mesh as mesh
from slow_test import slow

from pycuda import driver as drv

def normalize(vs):
    return vs / np.linalg.norm(vs, axis = 1).reshape((vs.shape[0], 1))

@slow
def test_farfield():
    n = 8000
    obs_pts = np.random.rand(n, 3).astype(np.float32)
    obs_ns = normalize(np.random.rand(n, 3).astype(np.float32))
    src_pts = np.random.rand(n, 3).astype(np.float32)
    src_ns = obs_ns

    farfield = load_gpu('tectosaur/integrals.cu', print_code = True)\
        .get_function("farfield_ptsH")

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

@slow
def test_farfield_tris():
    farfield = load_gpu('tectosaur/integrals.cu').get_function("farfield_trisH")

    N = 1024
    pts, tris = mesh.make_strip(N)
    pts = np.array(pts).astype(np.float32)
    tris = np.array(tris).astype(np.int32)

    q = quad.gauss4d_tri(2)
    qx = q[0].astype(np.float32)
    qw = q[1].astype(np.float32)

    result = np.empty((tris.shape[0], tris.shape[0], 3, 3, 3, 3)).astype(np.float32)

    block = (8, 8, 1)
    grid = (tris.shape[0] // block[0], tris.shape[0] // block[1])
    print(block, grid)
    print(farfield(
        drv.Out(result),
        np.int32(qx.shape[0]),
        drv.In(qx),
        drv.In(qw),
        drv.In(pts),
        np.int32(tris.shape[0]),
        drv.In(tris),
        np.int32(tris.shape[0]),
        drv.In(tris),
        np.float32(1.0),
        np.float32(0.25),
        block = block,
        grid = grid,
        time_kernel = True
    ))
    import ipdb; ipdb.set_trace()

