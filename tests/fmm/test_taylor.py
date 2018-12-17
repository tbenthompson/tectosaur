import time
from math import factorial
import scipy.special
import scipy.spatial
import numpy as np

import tectosaur as tct
from tectosaur.mesh.modify import concat
from tectosaur.fmm.tsfmm import *
from tectosaur.fmm.ts_terms import *
import tectosaur.util.gpu as gpu

def test_R():
    y = (2.0,3.0,4.0)
    n_max = 1

    reala, imaga = Rdirect(n_max, y)
    realb, imagb = R(n_max, y)
    np.testing.assert_almost_equal(reala, realb)
    np.testing.assert_almost_equal(imaga, imagb)

def test_R_storagefree():
    order = 4
    y = (1.0,2.0,3.0)

    reala, imaga = R(order, y)
    realb, imagb = R_storagefree(order, y)
    np.testing.assert_almost_equal(reala, realb)
    np.testing.assert_almost_equal(imaga, imagb)

def test_S():
    y = (2.0,3.0,4.0)
    n_max = 4

    reala, imaga = Sdirect(n_max, y)
    realb, imagb = S(n_max, y)
    np.testing.assert_almost_equal(reala, realb)
    np.testing.assert_almost_equal(imaga, imagb)

def test_S_storagefree():
    order = 4
    y = (1.0,2.0,3.0)

    reala, imaga = S(order, y)
    realb, imagb = S_storagefree(order, y)
    np.testing.assert_almost_equal(reala, realb)
    np.testing.assert_almost_equal(imaga, imagb)

def test_transfer_invr():
    n_src = 100
    n_obs = 100
    order = 5
    src_pts = np.random.rand(n_src, 3) - 0.5
    str = np.random.rand(n_src)
    obs_pts = np.random.rand(n_obs, 3)
    obs_pts[:,0] += 3

    r = scipy.spatial.distance.cdist(src_pts, obs_pts)
    correct = np.sum((1.0 / r).T * str, axis = 1)

    for order in range(1, 10):
        exp_pt = np.array((0,0,0))
        src_sep = src_pts - exp_pt[np.newaxis,:]
        Rsumreal = np.zeros((order + 1, 2 * order + 1))
        Rsumimag = np.zeros((order + 1, 2 * order + 1))
        for i in range(src_sep.shape[0]):
            Rvr, Rvi = R(order, src_sep[i,:])
            Rsumreal += Rvr * str[i]
            Rsumimag += Rvi * str[i]

        obs_sep = obs_pts - exp_pt[np.newaxis,:]
        result = np.zeros(n_obs)
        for i in range(obs_sep.shape[0]):
            Svr, Svi = S(order, obs_sep[i,:])
            # for mi in range(order + 1):
            #     result[i] += np.sum(
            #         Svr[:,order + mi] * Rsumreal[:,order + mi]
            #         + Svi[:,order + mi] * Rsumimag[:,order + mi]
            #     )
            #     if mi > 0:
            #         result[i] += np.sum(
            #             ((-1) ** (2 * mi)) * Svr[:,order + mi] * Rsumreal[:,order + mi]
            #             + ((-1) ** (2 * mi + 2)) * Svi[:,order + mi] * Rsumimag[:,order + mi]
            #         )
            result[i] += np.sum(Svr * Rsumreal + Svi * Rsumimag)
        n_multipole = (order + 1) * (2 * order + 1)
        print(n_src * n_obs, n_src * n_multipole + n_multipole * n_obs)
        print(order, n_multipole, (result - correct)[-1], result[-1], correct[-1])

def test_ror_invr3():
    n_src = 100
    n_obs = 2
    order = 5
    src_pts = np.random.rand(n_src, 3) - 0.5
    str = np.ones((n_src, 3))#np.random.rand(n_src, 3)
    obs_pts = np.random.rand(n_obs, 3)
    obs_pts[:,0] += 3

    r = scipy.spatial.distance.cdist(obs_pts, src_pts)
    rd = [np.subtract.outer(obs_pts[:,d], src_pts[:,d]) for d in range(3)]
    d1 = 0
    d2 = 0
    correct = np.sum((rd[d1] * rd[d2] / (r ** 3)) * str[:,d2], axis = 1)
    # correct = np.sum((rd[d1] / (r ** 3)) * str[:,d2], axis = 1)
    print(correct)

    for order in range(1, 10):
        exp_pt = np.array((0,0,0))
        src_sep = exp_pt[np.newaxis,:] - src_pts
        Rsumreal1 = np.zeros((order + 1, 2 * order + 1))
        Rsumimag1 = np.zeros((order + 1, 2 * order + 1))
        Rsumreal2 = np.zeros((order + 1, 2 * order + 1))
        Rsumimag2 = np.zeros((order + 1, 2 * order + 1))
        for i in range(src_sep.shape[0]):
            Rvr, Rvi = R(order, src_sep[i,:])
            Rsumreal1 += Rvr * str[i,d2]
            Rsumimag1 += Rvi * str[i,d2]
            Rsumreal2 += src_sep[i,d2] * Rvr * str[i,d2]
            Rsumimag2 += src_sep[i,d2] * Rvi * str[i,d2]

        obs_sep = exp_pt[np.newaxis,:] - obs_pts
        result = np.zeros(n_obs)
        for i in range(obs_sep.shape[0]):
            Svr, Svi = S(order, obs_sep[i,:])
            Sdvr, Sdvi = Sderivs(order, obs_sep[i,:], d1)
            t1 = Sdvr * Rsumreal2 + Sdvi * Rsumimag2
            t2 = -obs_sep[i,d2] * (Sdvr * Rsumreal1 + Sdvi * Rsumimag1)
            print(np.sum(t1), np.sum(t2))
            result[i] += np.sum(t1 + t2)
        # n_multipole = (order + 1) * (2 * order + 1)
        # print(n_src * n_obs, n_src * n_multipole + n_multipole * n_obs)
        print(order, (result - correct)[-1], result[-1], correct[-1])


def test_elasticU():
    n_src = 100
    n_obs = 2
    order = 5
    src_pts = np.random.rand(n_src, 3) - 0.5
    str = np.ones((n_src, 3))#np.random.rand(n_src, 3)
    obs_pts = np.random.rand(n_obs, 3)
    obs_pts[:,0] += 3

    r = scipy.spatial.distance.cdist(obs_pts, src_pts)
    rd = [np.subtract.outer(obs_pts[:,d], src_pts[:,d]) for d in range(3)]
    d1 = 0
    d2 = 0
    nu = 0.25
    K = (rd[d1] * rd[d2]) / (r ** 2)
    if d1 == d2:
        K += (3 - 4 * nu)
    K /= r
    correct = np.sum(K * str[:,d2], axis = 1)
    # correct = np.sum((rd[d1] / (r ** 3)) * str[:,d2], axis = 1)
    print(correct)

    for order in range(1, 20):
        exp_pt = np.array((0,0,0))
        src_sep = exp_pt[np.newaxis,:] - src_pts
        Rsumreal1 = np.zeros((order + 1, 2 * order + 1))
        Rsumimag1 = np.zeros((order + 1, 2 * order + 1))
        Rsumreal2 = np.zeros((order + 1, 2 * order + 1))
        Rsumimag2 = np.zeros((order + 1, 2 * order + 1))
        for i in range(src_sep.shape[0]):
            Rvr, Rvi = R(order, src_sep[i,:])
            Rsumreal1 += Rvr * str[i,d2]
            Rsumimag1 += Rvi * str[i,d2]
            Rsumreal2 += src_sep[i,d2] * Rvr * str[i,d2]
            Rsumimag2 += src_sep[i,d2] * Rvi * str[i,d2]

        obs_sep = exp_pt[np.newaxis,:] - obs_pts
        result = np.zeros(n_obs)
        for i in range(obs_sep.shape[0]):
            Svr, Svi = S(order, obs_sep[i,:])
            Sdvr, Sdvi = Sderivs(order, obs_sep[i,:], d1)
            t1 = Sdvr * Rsumreal2 + Sdvi * Rsumimag2
            t2 = -obs_sep[i,d2] * (Sdvr * Rsumreal1 + Sdvi * Rsumimag1)
            result[i] += np.sum(t1 + t2)
            if d1 == d2:
                result[i] += (3 - 4 * nu) * np.sum(Svr * Rsumreal1 + Svi * Rsumimag1)
        # n_multipole = (order + 1) * (2 * order + 1)
        # print(n_src * n_obs, n_src * n_multipole + n_multipole * n_obs)
        print(order, (result - correct)[-1], result[-1], correct[-1])
    print(result)

def test_cudaR():
    order = 8
    float_type = np.float64
    quad_order = 4
    K_params = np.array([1.0, 0.25])
    m_src = (
        np.array([[0,0,0],[0,1.3,0],[0.1,0,1]]),
        np.array([[0,1,2]])
    )
    offset = 4
    m_obs = (
        np.array([[offset,0,0],[1 + offset,0,0],[offset,1,0]]),
        np.array([[0,1,2]])
    )
    v = np.ones(m_src[1].shape[0] * 9)

    gpu_src_pts = gpu.to_gpu(m_src[0], float_type)
    gpu_src_tris = gpu.to_gpu(m_src[1], np.int32)
    gpu_obs_pts = gpu.to_gpu(m_obs[0], float_type)
    gpu_obs_tris = gpu.to_gpu(m_obs[1], np.int32)
    gpu_v = gpu.to_gpu(v, float_type)

    gpu_multipoles = gpu.empty_gpu(4 * 2 * (order + 1) * (2 * order + 1), float_type)
    cfg = make_config(
        params = K_params, order = order, quad_order = quad_order,
        float_type = float_type
    )
    cfg['gpu_module'].p2m(
        gpu_multipoles,
        gpu_src_pts,
        gpu_src_tris,
        gpu_v,
        np.int32(gpu_src_tris.shape[0]),
        grid = (1,1,1),
        block = (1,1,1)
    )
    gpu_out = gpu.empty_gpu(m_obs[1].shape[0] * 9, float_type)
    cfg['gpu_module'].m2p_R(
        gpu_out,
        gpu_obs_pts,
        gpu_obs_tris,
        gpu_multipoles,
        np.int32(gpu_obs_tris.shape[0]),
        grid = (1,1,1),
        block = (1,1,1)
    )


    tri_pts = m_src[0][m_src[1]]
    import tectosaur.util.geometry as geo
    un = geo.unscaled_normals(tri_pts)
    js = geo.jacobians(un)
    exp_pt = np.array((0,0,0))
    Rsumreal = np.zeros((order + 1, 2 * order + 1, 4))
    Rsumimag = np.zeros((order + 1, 2 * order + 1, 4))
    for qi in range(cfg['quad'][0].shape[0]):
        xhat = cfg['quad'][0][qi, 0]
        yhat = cfg['quad'][0][qi, 1]
        quadw = cfg['quad'][1][qi]
        basis = geo.linear_basis_tri(xhat, yhat)
        pt = geo.tri_pt(basis, tri_pts[0])
        src_sep = pt - exp_pt
        VV = np.array([0,0,0.0])
        for bi in range(3):
            for d in range(3):
                VV[d] += v[bi * 3 + d] * basis[bi]

        Rvr, Rvi = R(order, src_sep)
        for d in range(3):
            Rsumreal[:,:,d] += quadw * js[0] * Rvr * VV[d]
            Rsumimag[:,:,d] += quadw * js[0] * Rvi * VV[d]
        Rsumreal[:,:,3] += quadw * js[0] * Rvr * VV.dot(src_sep)
        Rsumimag[:,:,3] += quadw * js[0] * Rvi * VV.dot(src_sep)

    multipoles = gpu_multipoles.get().reshape((order + 1, 2 * order + 1, 4, 2))
    for d in range(4):
        Rsumreal2 = multipoles[:,:,d,0]
        Rsumimag2 = multipoles[:,:,d,1]
        np.testing.assert_almost_equal(Rsumreal[:,:,d], Rsumreal2)
        np.testing.assert_almost_equal(Rsumimag[:,:,d], Rsumimag2)


    tri_pts = m_obs[0][m_obs[1]]
    import tectosaur.util.geometry as geo
    un = geo.unscaled_normals(tri_pts)
    js = geo.jacobians(un)
    result = np.zeros(m_obs[1].shape[0] * 9)
    for qi in range(cfg['quad'][0].shape[0]):
        xhat = cfg['quad'][0][qi, 0]
        yhat = cfg['quad'][0][qi, 1]
        quadw = cfg['quad'][1][qi]
        basis = geo.linear_basis_tri(xhat, yhat)
        pt = geo.tri_pt(basis, tri_pts[0])
        obs_sep = pt - exp_pt

        Svr, Svi = S(order, obs_sep)
        for bi in range(3):
            result[bi * 3 + 0] += quadw * js[0] * basis[bi] * np.sum(
                Svr * Rsumreal[:,:,0] + Svi * Rsumimag[:,:,0]
            )
    result2 = gpu_out.get()
    full_m = concat(m_src, m_obs)
    src_subset = np.arange(0, m_src[1].shape[0])
    obs_subset = np.arange(0, m_obs[1].shape[0]) + m_src[1].shape[0]
    op = tct.TriToTriDirectFarfieldOp(
        quad_order, 'invr3', [], full_m[0], full_m[1],
        float_type, obs_subset, src_subset
    )
    result3 = op.dot(v)
    np.testing.assert_almost_equal(result, result3)
    np.testing.assert_almost_equal(result, result2)

def test_cudaU():
    order = 8
    float_type = np.float64
    quad_order = 2
    K_params = np.array([1.0, 0.25])
    K_name = 'elasticU3'

    n = 20
    offset = 5
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    m_src = tct.make_rect(n, n, corners)
    m_obs = tct.make_rect(n, n, corners)
    m_obs[0][:,0] += offset

    v = np.random.rand(m_src[1].shape[0] * 9)


    full_m = concat(m_src, m_obs)
    src_subset = np.arange(0, m_src[1].shape[0])
    obs_subset = np.arange(0, m_obs[1].shape[0]) + m_src[1].shape[0]
    op = tct.TriToTriDirectFarfieldOp(
        quad_order, K_name, K_params, full_m[0], full_m[1],
        float_type, obs_subset, src_subset
    )
    y1 = op.dot(v)

    for order in range(1, 8):
        cfg = make_config(
            params = K_params, order = order, quad_order = quad_order,
            float_type = float_type
        )
        gpu_src_pts = gpu.to_gpu(m_src[0], float_type)
        gpu_src_tris = gpu.to_gpu(m_src[1], np.int32)
        gpu_obs_pts = gpu.to_gpu(m_obs[0], float_type)
        gpu_obs_tris = gpu.to_gpu(m_obs[1], np.int32)
        gpu_params = gpu.to_gpu(K_params, float_type)
        gpu_v = gpu.to_gpu(v, float_type)

        gpu_multipoles = gpu.empty_gpu(4 * 2 * (order + 1) * (2 * order + 1), float_type)
        cfg['gpu_module'].p2m(
            gpu_multipoles,
            gpu_src_pts,
            gpu_src_tris,
            gpu_v,
            np.int32(gpu_src_tris.shape[0]),
            grid = (1,1,1),
            block = (1,1,1)
        )
        # print(gpu_multipoles.get().reshape((order + 1, order + 1, 4, 2)))

        gpu_out = gpu.empty_gpu(m_obs[1].shape[0] * 9, float_type)
        cfg['gpu_module'].m2p_U(
            gpu_out,
            gpu_obs_pts,
            gpu_obs_tris,
            gpu_multipoles,
            gpu_params,
            np.int32(gpu_obs_tris.shape[0]),
            grid = (1,1,1),
            block = (1,1,1)
        )
        y2 = gpu_out.get()

        print(np.linalg.norm(y1 - y2))
    np.testing.assert_almost_equal(y1, y2, 5)

def test_fmmU():
    order = 4
    float_type = np.float64
    quad_order = 2
    K_params = np.array([1.0, 0.25])
    K_name = 'elasticU3'

    n = 20
    offset = 0.0
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    m_src = tct.make_rect(n, n, corners)
    v = np.random.rand(m_src[1].shape[0] * 9)

    m_obs = tct.make_rect(n, n, corners)
    m_obs[0][:,0] += offset

    full_m = concat(m_src, m_obs)
    src_subset = np.arange(0, m_src[1].shape[0])
    obs_subset = np.arange(0, m_obs[1].shape[0]) + m_src[1].shape[0]
    op = tct.TriToTriDirectFarfieldOp(
        quad_order, K_name, K_params, full_m[0], full_m[1],
        float_type, obs_subset, src_subset
    )
    y1 = op.dot(v)

    fmm = TSFMM(
        m_obs, m_src, params = K_params, order = order,
        quad_order = quad_order, float_type = float_type,
        mac = 2.5, max_pts_per_cell = 20, n_workers_per_block = 128
    )
    report_interactions(fmm)

    y2 = fmm.dot(v)
    print(order, np.linalg.norm((y1 - y2)) / np.linalg.norm(y1))
    np.testing.assert_almost_equal(y1, y2)

def benchmark():
    np.random.seed(123456)
    float_type = np.float32
    n = 100
    corners = [[-1.0, -1.0, 0], [-1.0, 1.0, 0], [1.0, 1.0, 0], [1.0, -1.0, 0]]
    m = tct.make_rect(n, n, corners)
    v = (100000 * np.random.rand(m[1].shape[0] * 9)).astype(float_type)
    t = tct.Timer()

    # all_tris = np.arange(m[1].shape[0])
    # op = tct.TriToTriDirectFarfieldOp(
    #     2, 'elasticU3', [1.0, 0.25], m[0], m[1],
    #     float_type, all_tris, all_tris
    # )
    # t.report('build direct')
    # y1 = op.dot(v)
    # t.report('op.dot direct')

    # TODO: block the p2m,m2m...
    # TODO: still maybe some room in p2p compared to direct
    # TODO: tons of room in m2p
    # TODO: maybe do full fmm?
    fmm = TSFMM(
        m, m, params = np.array([1.0, 0.25]), order = 4,
        quad_order = 2, float_type = float_type,
        mac = 2.5, max_pts_per_cell = 80, n_workers_per_block = 128
    )
    report_interactions(fmm)
    t.report('build')
    out = fmm.dot(v)
    t.report('first dot')
    out = fmm.dot(v)
    t.report('second dot')
    start = time.time()
    out = fmm.dot(v)
    t.report('third dot')
    t = time.time() - start
    interactions = m[1].shape[0] ** 2
    print('n/sec', m[1].shape[0] / t)
    print('billion interactions/sec', interactions / t / 1e9)

    filename = 'tests/fmm/taylorbenchmarkcorrect.npy'
    # np.save(filename, out)
    correct = np.load(filename)
    # print(out, correct, y1)
    np.testing.assert_almost_equal(out, correct)

#Real Sdr[3];
# Real Sdi[3];
# Real Sv[3][2];
# for (int d = -1; d <= 1; d++) {
#     Sv[d + 1][0] = allSreal[ni + 1][${order} + mi + d];
#     Sv[d + 1][1] = allSimag[ni + 1][${order} + mi + d];
# }
# Sdr[0] = 0.5 * (Sv[0][0] - Sv[2][0]);
# Sdi[0] = 0.5 * (Sv[0][1] - Sv[2][1]);
# Sdr[1] = -0.5 * (Sv[0][1] + Sv[2][1]);
# Sdi[1] = 0.5 * (Sv[0][0] + Sv[2][0]);
# Sdr[2] = -Sv[1][0];
# Sdi[2] = -Sv[1][1];
#int pos_mi = abs(mi);
#Real multRR = 1.0;
#Real multRI = 1.0;
#if (mi < 0) {
#    multRR = -1.0;
#    if (mi % 2 == 0) {
#        multRR = 1.0;
#        multRI = -1.0;
#    }
#}
#for (int d1 = 0; d1 < 3; d1++) {
#    % for d2 in range(3):
#    {
#        int idx = this_src_n_idx * ${4 * 2 * (order + 1) * (order + 1)}
#            + ni * ${4 * 2 * (order + 1)}
#            + pos_mi * 4 * 2
#            + ${d2} * 2;
#        Real Rr = multRR * multipoles[idx];
#        Real Ri = multRI * multipoles[idx + 1];
#        Real v = D${dn(d2)} * (Rr * Sdr[d1] + Ri * Sdi[d1]);
#        for (int bi = 0; bi < 3; bi++) {
#            sum[bi][d1] -= CsU1 * obsb[bi] * v;
#        }
#    }
#    % endfor
#    int idx = this_src_n_idx * ${4 * 2 * (order + 1) * (order + 1)}
#        + ni * ${4 * 2 * (order + 1)}
#        + pos_mi * 4 * 2
#        + 3 * 2;
#    Real Rr = multRR * multipoles[idx];
#    Real Ri = multRI * multipoles[idx + 1];
#    Real v =  Rr * Sdr[d1] + Ri * Sdi[d1];
#    for (int bi = 0; bi < 3; bi++) {
#        sum[bi][d1] += CsU1 * obsb[bi] * v;
#    }
#}

if __name__ == "__main__":
    benchmark()
