from tectosaur.coincident_rule import *
from tectosaur.quadrature import *
from tectosaur.basis import *
import cppimport
cppimport.install()
from tectosaur.coincident import *
from slow_test import slow

def test_simple():
    eps = 0.01

    q = coincident_quad(0.01, 10, 10, 10, 10)

    result = quadrature(lambda x: x[:, 2], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 2)

    result = quadrature(lambda x: x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 12.0, 2)

    result = quadrature(lambda x: x[:, 2] * x[:, 3], q)
    np.testing.assert_almost_equal(result, 1.0 / 48.0, 2)

@slow
def test_kernel():
    tri = [
        [0, 0, 0],
        [0, 0, 1],
        [1, 0, 0]
    ]

    def laplace(i, j, eps, pts):
        obsxhat = pts[:, 0]
        obsyhat = pts[:, 1]
        srcxhat = pts[:, 2]
        srcyhat = pts[:, 3]

        obsbasis = linear_basis_tri(obsxhat, obsyhat)
        srcbasis = linear_basis_tri(srcxhat, srcyhat)

        obsn = tri_normal(tri)

        obspt = (tri_pt(obsbasis, tri).T - eps * obsn).T
        srcpt = tri_pt(srcbasis, tri)

        R = np.sqrt(
            (obspt[0,:] - srcpt[0,:]) ** 2 +
            (obspt[1,:] - srcpt[1,:]) ** 2 +
            (obspt[2,:] - srcpt[2,:]) ** 2
        )
        K = (1.0 / 4.0 / np.pi) * ((1 / R ** 3) - (3.0 * eps ** 2 / R ** 5))
        return obsbasis[i, :] * srcbasis[j, :] * K

    eps = 0.01
    for i in range(3):
        for j in range(i):
            K = lambda pts: laplace(i, j, eps, pts)
            exact = quadrature(K, coincident_quad(eps, 18, 15, 15, 18))
            def tryit(n1,n2,n3,n4):
                q = coincident_quad(eps, n1, n2, n3, n4)
                result = quadrature(K, q)
                return np.abs((result - exact) / exact)
            assert(tryit(7,6,4,10) < 0.005)
            assert(tryit(14,9,7,10) < 0.0005)
            assert(tryit(13,12,11,13) < 0.00005)

def test_coincident_gpu():
    import pycuda.autoinit
    import pycuda.driver as drv
    from pycuda.compiler import SourceModule

    import mako.template
    import mako.runtime
    import mako.exceptions
    import mako.lookup
    import os

    N = 16 * 10000 * 6
    print(N)
    xs = np.linspace(0, 100, N + 1)
    tris = []
    pts = []
    for i in range(N):
        npts = len(pts)
        tris.append([npts, npts + 1, npts + 2])
        tris.append([npts + 2, npts + 1, npts + 3])
        pts.append((xs[i], 0, 0))
        pts.append((xs[i + 1], 0, 0))
        pts.append((xs[i], 1, 0))
        pts.append((xs[i + 1], 1, 0))

    pts = np.array(pts).astype(np.float32)
    tris = np.array(tris).astype(np.int32)
    q = richardson_quad([0.01], lambda e: coincident_quad(e, 8, 8, 5, 10))
    print(q[0].shape)
    qx = q[0].astype(np.float32)
    qw = q[1].astype(np.float32)

    result = np.zeros((tris.shape[0], 3, 3, 3, 3)).astype(np.float32)

    block = (128, 1, 1)
    filepath = 'tectosaur/coincident.cu'
    lookup = mako.lookup.TemplateLookup(directories=[os.path.dirname(filepath)])
    tmpl = mako.template.Template(filename = filepath, lookup = lookup)
    code = tmpl.render(block = block)
    # print('\n'.join([str(i) + '   ' + line for i,line in enumerate(code.split('\n'))]))
    mod = SourceModule(code, options = ['-std=c++11'], no_extern_c = True)
    coincident = mod.get_function("coincident")

    grid = (int(tris.shape[0]/block[0]),1)
    print(coincident(
        drv.Out(result),
        np.int32(q[0].shape[0]),
        drv.In(qx),
        drv.In(qw),
        drv.In(pts),
        drv.In(tris),
        np.float32(1.0),
        np.float32(0.25),
        block = block,
        grid = grid,
        time_kernel = True
    ))
    # result2 = coincidentH(q[0], q[1], pts, tris, 1.0, 0.25)
    # print(np.max(np.abs(result - result2)))
    # import ipdb; ipdb.set_trace()

def func_star(args):
    return coincidentH(*args)


def test_coincident2():
    pts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.5, 0]
    ])
    tris = [[0,1,2]] * 6 * 167

    q = richardson_quad([0.01], lambda e: coincident_quad(e, 8, 8, 5, 10))
    import time
    # import multiprocessing
    # n_cores = 12
    # pool = multiprocessing.Pool(n_cores)
    # start = time.time()
    # step = len(tris) / n_cores
    # pool.map(func_star, [
    #     (q[0], q[1], pts, tris[(step*i):(step*(i+1))], 1.0, 0.25)
    #     for i in range(n_cores)
    # ])
    # print(time.time() - start)
    start = time.time()
    result = coincidentH(q[0], q[1], pts, tris, 1.0, 0.25)
    print(time.time() - start)


def test_cpp_module():
    # q = coincident_quad(0.01, 14, 9, 7, 10)
    # qhigh = coincident_quad(0.01, 20, 20, 20, 20)
    pts = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0.5, 0.5, 0]
    ])
    tris = [[0,1,2]] * 1

    import time
    # a1 = coincidentH(q[0], q[1], [0.01], pts, tris, 1.0, 0.25)
    # a2 = coincidentH(qhigh[0], qhigh[1], [0.01], pts, tris, 1.0, 0.25)
    # print(np.max(np.abs(a2 - a1)))

    eps = 10 ** np.linspace(-2, -2, 1)

    start = time.time()

    # int(5 - 3 * np.log(e))
    # qq = richardson_quad(eps, lambda e: coincident_quad(e, 7, 6, 4, 10))
    # qq2 = richardson_quad(eps, lambda e: coincident_quad(e, 15, 15, 15, 15))
    # relerr = []
    # for i in range(1000):
    #     logscale = np.random.uniform(0, 20)
    #     scale = 2 ** logscale
    #     x = np.random.uniform(0, 1)
    #     y = np.random.uniform(0, 1)
    #     leg1 = np.sqrt((x - 1) ** 2 + y ** 2)
    #     leg2 = np.sqrt(x ** 2 + y ** 2)
    #     if leg1 > 1.0 or leg2 > 1.0 or leg1 < 0.5 or leg2 < 0.5:
    #         continue
    #     pts = [
    #         [0, 0, 0],
    #         [scale, 0, 0],
    #         [x * scale, y * scale, 0]
    #     ]

    #     a1 = coincidentH(qq[0], qq[1], pts, tris, 1.0, 0.25)[0][0][0][0][0]
    #     a2 = coincidentH(qq2[0], qq2[1], pts, tris, 1.0, 0.25)[0][0][0][0][0]
    #     print("GO")
    #     print("points: " + str(pts))
    #     print("abserr: " + str(np.abs(a1 - a2)))
    #     relerr.append(np.abs(a1 - a2) / np.abs(a2))
    #     print("relerr: " + str(relerr[-1]))
    #     print("correcter: " + str(a2))
    #     print('')
    # import ipdb; ipdb.set_trace()

    # print(len(qq[0]))
    # print(time.time() - start)
    # start = time.time()
    # a1 = coincidentH(qq[0], qq[1], pts, tris, 1.0, 0.25)
    # print(time.time() - start)
    # print("result: " + str(a1[0][0][0][0][0]))
