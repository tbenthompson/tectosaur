import pytest
import numpy as np

import tectosaur.util.geometry as geometry
import tectosaur.nearfield.nearfield_op as nearfield_op
from tectosaur.nearfield.interpolate import to_interval
from tectosaur.ops.dense_integral_op import DenseIntegralOp

from tectosaur.nearfield.table_lookup import *
from tectosaur.nearfield.table_params import min_angle_isoceles_height

from tectosaur.util.test_decorators import golden_master, slow, kernel

float_type = np.float64

def test_find_va_rotations():
    res = find_va_rotations([1,3,5],[2,3,4])
    np.testing.assert_equal(res, [[1,2,0],[1,2,0]])

    res = find_va_rotations([3,1,5],[2,3,4])
    np.testing.assert_equal(res, [[0,1,2],[1,2,0]])

    res = find_va_rotations([1,5,3],[2,4,3])
    np.testing.assert_equal(res, [[2,0,1],[2,0,1]])

def test_sub_basis():
    xs = np.linspace(0.0, 1.0, 11)
    for x in xs:
        pts = np.array([[0,0],[1,0],[x,1-x],[0,1]])
        tri1 = pts[[0,1,2]].tolist()
        tri2 = pts[[0,2,3]].tolist()
        full_tri = pts[[0,1,3]].tolist()
        input = np.ones(81).tolist()
        tri1area = np.linalg.norm(geometry.tri_normal(np.hstack((tri1, np.zeros((3,1))))))
        tri2area = np.linalg.norm(geometry.tri_normal(np.hstack((tri2, np.zeros((3,1))))))
        I1 = np.array(sub_basis(input, tri1, full_tri))
        I2 = np.array(sub_basis(input, tri2, full_tri))
        result = tri1area * I1 + tri2area * I2
        np.testing.assert_almost_equal(result, 1.0)

def test_sub_basis_identity():
    A = np.random.rand(81).tolist()
    B = sub_basis(
        A, [[0,0],[1,0],[0,1]], [[0,0],[1,0],[0,1]]
    )
    np.testing.assert_almost_equal(A, B)

def test_sub_basis_rotation():
    A = np.random.rand(81).reshape((3,3,3,3))
    B = sub_basis(A.flatten().tolist(), [[0,0],[1,0],[0,1]], [[0,1],[0,0],[1,0]])
    np.testing.assert_almost_equal(A[:,:,[1,2,0],:], np.array(B).reshape((3,3,3,3)))

def coincident_lookup_helper(K, correct_digits, n_tests = 10):
    np.random.seed(113)


    results = []
    for i in range(n_tests):
        try:
            A = np.random.rand(1)[0] * 0.5
            B = np.random.rand(1)[0]
            pr = np.random.rand(1)[0] * 0.5
            scale = np.random.rand(1)[0]
            flip = np.random.rand(1) > 0.5

            params = [1.0, pr]

            R = geometry.random_rotation()
            # print(R)

            pts = scale * np.array([[0,0,0],[1,0,0],[A,B,0]], dtype = np.float64)
            pts = R.dot(pts)

            if flip:
                tris = np.array([[0,2,1]])
            else:
                tris = np.array([[0,1,2]])

            op = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris, float_type)
            results.append(op.mat)
        except standardize.BadTriangleException as e:
            print("Bad tri: " + str(e))
    return np.array(results)

@golden_master(4)
def test_coincident_fast_lookup(request, kernel):
    return coincident_lookup_helper(kernel, 5, 3)

@slow
@golden_master(4)
def test_coincident_lookup(request, kernel):
    return coincident_lookup_helper(kernel, 5)

def adjacent_lookup_helper(K, correct_digits, n_tests = 10):
    np.random.seed(973)


    results = []
    for i in range(n_tests):
        # We want phi in [0, 3*pi/2] because the old method doesn't work for
        # phi >= 3*np.pi/2
        phi = to_interval(min_intersect_angle, 1.4 * np.pi, np.random.rand(1)[0])
        pr = np.random.rand(1)[0] * 0.5

        params = [1.0, pr]
        alpha = np.random.rand(1)[0] * 3 + 1
        beta = np.random.rand(1)[0] * 3 + 1

        scale = np.random.rand(1)[0] * 3
        translation = np.random.rand(3)
        R = geometry.random_rotation()

        # print(alpha, beta, phi, pr, scale, translation.tolist(), R.tolist())

        H = min_angle_isoceles_height
        pts = np.array([
            [0,0,0],[1,0,0],
            [0.5,alpha*H*np.cos(phi),alpha*H*np.sin(phi)],
            [0.5,beta*H,0]
        ])

        pts = (translation + R.dot(pts.T).T * scale).copy()

        tris = np.array([[0,1,3],[1,0,2]])

        op = DenseIntegralOp(
            10, 3, 10, 3.0, K, params, pts, tris, float_type
        )
        results.append(op.mat[:9,9:])
    return np.array(results)

@golden_master()
def test_adjacent_fast_lookup(request, kernel):
    return adjacent_lookup_helper(kernel, 5, 1)

@slow
@golden_master()
def test_adjacent_lookup(request, kernel):
    return adjacent_lookup_helper(kernel, 5)

def fault_surface_experiment():
    pts = np.array([[-1, 0, 0], [0, -1, 0], [0, 1, 0], [1, 0, 0], [0, 0, -1]])
    tris = np.array([[0, 1, 2], [2, 1, 3], [1, 2, 4]])
    K = 'elasticT3'
    params = [1.0, 0.25]
    op = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris, float_type)
    A = op.mat[:9,18:]
    B = op.mat[9:18,18:]
    import matplotlib.pyplot as plt
    plt.imshow(np.sign(A) * np.log10(np.abs(A)), interpolation = 'none')
    plt.colorbar()
    plt.figure()
    plt.imshow(np.sign(B) * np.log10(np.abs(B)), interpolation = 'none')
    plt.title('B')
    plt.show()
    import ipdb
    ipdb.set_trace()

def adj_theta_dependence_experiment():
    ts = np.linspace(0.2 * np.pi, 1.8 * np.pi, 20)
    K = 'elasticT3'
    params = [1.0, 0.25]
    v1 = []
    v2 = []
    for theta in ts:
        pts = np.array([
            [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0],
            [0.5, 0.5 * np.cos(theta), 0.5 * np.sin(theta)]
        ])
        tris = np.array([[0,1,2],[1,0,3]])
        op = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris, float_type)
        v1.append(op.mat[:9,9:].reshape((3,3,3,3)))
        v2.append(op.mat[9:,:9].reshape((3,3,3,3)))
    v1 = np.array(v1)
    v2 = np.array(v2)
    vv1,vv2 = np.load('adj_theta_dependence.npy')
    import matplotlib.pyplot as plt
    for b1 in range(3):
        for d1 in range(3):
            for b2 in range(3):
                for d2 in range(3):
                    plt.plot(ts, v1[:,b1,d1,b2,d2], 'r-')
                    plt.plot(ts, v2[:,b1,d1,b2,d2], 'b-')
                    plt.plot(ts, vv1[:,b1,d1,b2,d2], 'r-.')
                    plt.plot(ts, vv2[:,b1,d1,b2,d2], 'b-.')
                    plt.title(' '.join([str(x) for x in [b1,d1,b2,d2]]))
                    plt.show()

def adj_flipping_experiment():
    theta = np.pi / 2.0
    K = 'elasticT3'
    params = [1.0, 0.25]
    pts = np.array([
        [0, 0, 0], [1, 0, 0], [0.5, 0.5, 0],
        [0.5, 0.5 * np.cos(theta), 0.5 * np.sin(theta)]
    ])
    print(pts)
    tris = np.array([[0,1,2],[1,0,3]])
    tris2 = np.array([[0,1,2],[0,1,3]])
    op = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris, float_type)
    op2 = DenseIntegralOp(10, 3, 10, 3.0, K, params, pts, tris2, float_type)
    A = op.mat[:9,9:].reshape((3,3,3,3))
    B = op2.mat[:9,9:].reshape((3,3,3,3))

    AG,BG = np.load('flipped_experiment_90.npy')
    AG = AG.reshape((3,3,3,3))
    BG = BG.reshape((3,3,3,3))
    CG = BG.copy()
    CG[:,:,0,:] = BG[:,:,1,:]
    CG[:,:,1,:] = BG[:,:,0,:]
    CG *= -1
    import ipdb
    ipdb.set_trace()

@profile
def benchmark_co_lookup_pts():
    import taskloaf as tsk
    from tectosaur.mesh.mesh_gen import make_rect
    from tectosaur.nearfield.table_lookup import coincident_lookup_pts
    n = 600
    corners = [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]
    m = make_rect(n, n, corners)
    tri_pts = m[0][m[1]]

    # import pickle
    # t = Timer(just_print = True)
    # np.save('abc.npy', tri_pts)
    # t.report('save')
    # tp = np.load('abc.npy')
    # t.report('load')
    # d = pickle.dumps(tri_pts)
    # t.report('pickle')
    # pickle.loads(d)
    # t.report('unpickle')
    # import ctypes
    # c_double_p = ctypes.POINTER(ctypes.c_double)
    # d2 = tri_pts.ctypes.data_as(c_double_p)
    # t.report('access ptr')
    # new_array = np.ctypeslib.as_array(d2, shape=tri_pts.shape)
    # t.report('new array from ptr')

    n_chunks = 2
    chunk_bounds = np.linspace(0, tri_pts.shape[0], n_chunks + 1)
    async def submit(w):

        r = w.memory.put(tri_pts)
        t = Timer(just_print = True)
        async with tsk.Profiler(w, range(n_chunks)):
            for j in range(5):
                results = []
                for i in range(n_chunks):
                    s, e = np.ceil(chunk_bounds[i:(i+2)]).astype(np.int)
                    async def T(w, s = s, e = e):
                        print('start ', s,e)
                        tri_pts = await tsk.remote_get(r)
                        # tri_pts = np.load('abc.npy')
                        out = coincident_lookup_pts(tri_pts[s:e], 0.25)
                        print('finish ', s,e)
                        return None
                        # return out
                    results.append(tsk.task(w, T, to = i))
                for i in range(n_chunks)[::-1]:
                    await results[i]
        t.report('done')

    tsk.cluster(n_chunks, submit, tsk.mpirun)#lambda n,f: tsk.localrun(n,f,pin = True))

def benchmark_numpy_pickle():
    #TODO: cloudpickle dumps numpy arrays very slowly, should be solvable!
    # from cloudpickle import dumps, loads
    from pickle import dumps, loads
    n = 10000000
    x = np.random.rand(n)
    t = Timer(just_print = True)
    a = dumps(x)
    t.report('')
    b = loads(a)
    t.report('')

if __name__ == "__main__":
    # benchmark_numpy_pickle()
    benchmark_co_lookup_pts()
    # adj_flipping_experiment()
    # adj_theta_dependence_experiment()
