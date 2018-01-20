import pytest
import numpy as np

import tectosaur.util.geometry as geometry
import tectosaur.nearfield.nearfield_op as nearfield_op
from tectosaur.nearfield.interpolate import to_interval
from tectosaur.ops.dense_integral_op import DenseIntegralOp

import tectosaur.nearfield.standardize as standardize
from tectosaur.nearfield._table_lookup import find_va_rotations, sub_basis
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


import time
import taskloaf as tsk
def master_print(w, *args):
    if w.addr == 0:
        print(*args)

import taskloaf as tsk
import taskloaf.shmem
import taskloaf.profile
import time
from tectosaur.mesh.mesh_gen import make_rect
from tectosaur.nearfield.table_lookup import coincident_lookup_pts
import mpi4py
import struct
import capnp
import msg_capnp

import asyncio
import uvloop
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

async def T(w, args):
    m = msg_capnp.Msg.from_bytes(args)
    with taskloaf.shmem.Shmem(m.smfilename) as sm:
        sb = m.sb
        eb = m.eb
        st = m.st
        master_print(w, 'reset st', w.st - st)
        w.st = st
        # print(time.time() - w.st, w.addr, sb, eb)
        start_time = time.time() - st
        master_print(w, 'start', w.addr, time.time() - st)
        # This allows zero-copy reads from the incoming data.

        tri_pts2 = np.frombuffer(memoryview(sm.mem)[sb:eb]).reshape((-1,3,3))
        master_print(w, 'startfnc', w.addr, time.time() - st)

        out_inner_chunks = []
        # This inner_chunking is done so that the taskloaf worker can respond to
        # messages while work is being done. (Cooperative multitasking)
        inner_chunk_size = 5000
        master_print(
            w, 'n_inner_chunks: ',
            int(np.ceil(tri_pts2.shape[0] / inner_chunk_size))
        )
        for i in range(0, tri_pts2.shape[0], inner_chunk_size):
            inner_chunk = tri_pts2[i:(i + inner_chunk_size)]
            def f(w, inner_chunk = inner_chunk):
                return coincident_lookup_pts(inner_chunk, 0.25)
            out_inner_chunks.append(await tsk.task(w, f))
        master_print(w, 'finish', w.addr, time.time() - st)
        return start_time

# A first attempt at doing a really good parallelization of a simple
# embarassingly function using taskloaf. As a baseline (with n = 2000):
# single threaded time = 4.5 seconds
# OpenMP with OMP_NUM_THREADS=80
# 19/12/17 @ 5:15PM, current best including data transfer = 0.714s
# 19/12/17 @ 5:15PM, current best without data transfer = 0.13s
@profile
def benchmark_co_lookup_pts():
    n_cores = mpi4py.MPI.COMM_WORLD.Get_size()

    async def submit(w):
        n_chunks = n_cores

        # async with tsk.Profiler(w, range(n_chunks)):
        # n = 4000
        # corners = [[-1, -1, 0], [-1, 1, 0], [1, 1, 0], [1, -1, 0]]
        # m = make_rect(n, n, corners)
        # tri_pts = m[0][m[1]]
        # np.save('tripts.npy', tri_pts)
        tri_pts = np.load('tripts.npy')
        # time.sleep(2.0)
        chunk_bounds = np.linspace(0, tri_pts.shape[0], n_chunks + 1)

        t2 = Timer(just_print = True)
        with taskloaf.shmem.alloc_shmem(tri_pts) as sm_filepath:
            ratio = tri_pts.nbytes / tri_pts.shape[0]
            for k in range(10):
                T_dref = w.memory.put(serialized = taskloaf.serialize.dumps(T))
                st = time.time()
                w.st = time.time()
                super_chunks = 8
                super_chunk_size = n_chunks // super_chunks
                super_results = []
                async def super_chunk_fnc(w, super_chunk_start_b):
                    super_chunk_start = taskloaf.serialize.loads(w, super_chunk_start_b)
                    m = msg_capnp.Msg.new_message()
                    m.smfilename = sm_filepath
                    m.st = float(st)

                    results = []
                    for sub_i in range(super_chunk_size):
                        i = super_chunk_start + sub_i
                        s, e = np.ceil(chunk_bounds[i:(i+2)]).astype(np.int)
                        sb = s * ratio
                        eb = e * ratio
                        m.sb = int(sb)
                        m.eb = int(eb)
                        results.append(tsk.task(w, T_dref, m.to_bytes(), to = i % n_cores))
                    m.clear_write_flag()
                    to_start = time.time() - st
                    master_print(w, 'waiting', to_start)
                    out = []
                    for res in results:
                        out.append(await res)
                    return out
                super_dref = w.memory.put(
                    serialized = taskloaf.serialize.dumps(super_chunk_fnc)
                )

                async with taskloaf.profile.Profiler(w, range(1)):
                    t = Timer(just_print = True)
                    assert(n_chunks % super_chunks == 0)
                    for super_idx in range(super_chunks):
                        super_chunk_start = super_idx * super_chunk_size
                        dref = w.memory.put(value = taskloaf.serialize.dumps(super_chunk_start))
                        super_results.append(tsk.task(
                            w, super_dref, dref, to = super_idx * n_cores // super_chunks
                        ))
                    # print(
                    #     'longest start:',
                    #     np.max([await super_results[i] for i in range(super_chunks)])
                    # )
                    t.report('done')
                    print('')
                    print('')
                    print('')
                    print('')
                    print('')
        t2.report('done full')

    tsk.cluster(n_cores, submit, tsk.mpiexisting)#lambda n,f: tsk.localrun(n,f,pin = False))

if __name__ == "__main__":
    benchmark_co_lookup_pts()
    # adj_flipping_experiment()
    # adj_theta_dependence_experiment()
