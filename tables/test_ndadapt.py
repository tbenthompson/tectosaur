import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from ndadapt import *

def make_integrator(p, f):
    ps = [p] * 3
    q_unmapped = tensor_gauss(ps)

    def integrator(mins, maxs):
        out = []
        for i in range(mins.shape[0]):
            q = map_to(q_unmapped, mins[i,:], maxs[i,:])
            out.append(np.sum(f(q[0]) * q[1]))
        return out
    return integrator

def test_tensor_gauss():
    nqs = (4, 3, 6)
    q = tensor_gauss(nqs)
    for i in range(2 * nqs[0] - 1):
        for j in range(2 * nqs[1] - 1):
            for k in range(2 * nqs[2] - 1):
                f = lambda pts: (pts[:,0] ** i) * (pts[:,1] ** j) * (pts[:,2] ** k)
                est = sum(f(q[0]) * q[1])
                correct = np.prod([
                    (1.0 ** (idx + 1) - (-1.0) ** (idx + 1)) / (float(idx) + 1)
                    for idx in (i, j, k)
                ])
                np.testing.assert_almost_equal(est, correct)

def test_map_to():
    nqs = (4, 3, 6)
    q = map_to(tensor_gauss(nqs), np.array([0, 0, 0.]), np.array([1, 1, 1.]))
    i, j, k = 6, 4, 8
    f = lambda pts: (pts[:,0] ** i) * (pts[:,1] ** j) * (pts[:,2] ** k)
    est = sum(f(q[0]) * q[1])
    correct = np.prod([1.0 ** (idx + 1) / (float(idx) + 1) for idx in (i, j, k)])
    np.testing.assert_almost_equal(est, correct)

def test_2d_adapt():
    res, count = hadapt_nd(
        lambda pts: np.cos(pts[:,0] * 100) * np.cos(pts[:,1] * 100), (0,0), (1,1), 1e-15
    )
    print(res, count)
    res, count = hadapt_nd(
        lambda pts: np.cos(pts[:,0] * 100), (0,), (1,), 1e-6
    )
    print(res ** 2, count)

def make_integrator(d, p, f):
    ps = [p] * d
    q_unmapped = tensor_gauss(ps)

    def integrator(mins, maxs):
        out = []
        for i in range(mins.shape[0]):
            q = map_to(q_unmapped, mins[i,:], maxs[i,:])
            fvs = f(q[0])
            if len(fvs.shape) == 1:
                fvs = fvs[:,np.newaxis]
            assert(fvs.shape[0] == q[1].shape[0])
            out.append(np.sum(fvs * q[1][:,np.newaxis], axis = 0))
        return np.array(out)
    return integrator

def make_gpu_integrator(p, f):
    ps = [p] * 3
    q_unmapped = tensor_gauss(ps)
    code = open('toy_kernel.cu', 'r').read()
    module = SourceModule(code, options = ['--use_fast_math', '--restrict'])
    def integrator(mins, maxs):
        block = (32, 1, 1)
        remaining = mins.shape[0] % block[0]
        grid_main = (mins.shape[0] // block[0], 1, 1)
        grid_rem = (remaining, 1, 1)

        out = np.empty(mins.shape[0]).astype(np.float64)
        out_rem = np.empty(grid_rem[0]).astype(np.float64)

        def call_integrator(block, grid, result_buf, start_idx, end_idx):
            if grid[0] == 0:
                return
            module.get_function('compute_integrals')(
                drv.Out(result_buf),
                np.int32(q_unmapped[0].shape[0]),
                drv.In(q_unmapped[0].astype(np.float64)),
                drv.In(q_unmapped[1].astype(np.float64)),
                drv.In(mins[start_idx:end_idx].astype(np.float64)),
                drv.In(maxs[start_idx:end_idx].astype(np.float64)),
                block = block, grid = grid
            )

        call_integrator(block, grid_main, out, 0, mins.shape[0] - remaining)
        call_integrator(
            (1,1,1), grid_rem, out_rem, mins.shape[0] - remaining, mins.shape[0]
        )
        out[(mins.shape[0] - remaining):] = out_rem
        # out2 = []
        # for i in range(mins.shape[0]):
        #     q = map_to(q_unmapped, mins[i,:], maxs[i,:])
        #     out2.append(np.sum(f(q[0]) * q[1]))
        # # np.testing.assert_almost_equal(out2, out, 2)
        # return out2
        return out
    return integrator

def adapt_tester(f, correct):
    p = 5
    integrator = make_integrator(3, p, f)
    res, count = hadapt_nd_iterative(integrator, (0,0,0), (1,1,1), 1e-14)
    np.testing.assert_almost_equal(res, correct)

def test_simple():
    adapt_tester(lambda pts: pts[:,0], 0.5)

def test_vector_simple():
    adapt_tester(lambda pts: np.array([pts[:,0], pts[:,1]]).T, [0.5, 0.5])

def test_harder_vector_integrals():
    adapt_tester(
        lambda pts: np.array([np.cos(pts[:,0]), np.sin(pts[:,1])]).T,
        [np.sin(1), 1 - np.cos(1)]
    )

def test_1dintegral():
    integrator = make_integrator(1, 5, lambda pts: np.sin(100 * pts[:,0]))
    res, count = hadapt_nd_iterative(integrator, (0,), (1,), 1e-14)
    np.testing.assert_almost_equal(res, (1 - np.cos(100)) / 100.0)

def test_2dintegral():
    integrator = make_integrator(2, 5, lambda pts: np.sin(20 * pts[:,0] * pts[:,1]))
    res, count = hadapt_nd_iterative(integrator, (0,0), (1,1), 1e-14)
    np.testing.assert_almost_equal(res, 0.1764264058805085268)

def test_3d_adapt_nearly_sing():
    eps = 0.001
    A = 1.0
    f = lambda pts: 1.0 / (
        (pts[:,0] - pts[:,2]) ** 2 +
        (pts[:, 1] - 0.3) ** 2 +
        eps ** 2
    ) ** A

    p = 7
    res, count = hadapt_nd_iterative(make_gpu_integrator(p, f), (0,0,0), (1,1,1), 1e-14)
    print(res, count * p ** 3)


if __name__ == '__main__':
    # test_tensor_gauss()
    # test_map_to()
    # test_kbnsum()
    # test_2d_adapt()
    test_2d_adapt_nearly_sing()
