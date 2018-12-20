from tectosaur.fmm.ts_terms import *

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
