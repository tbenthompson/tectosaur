import numpy as np
from tectosaur.ops.sparse_integral_op import farfield_pts_wrapper
from tectosaur_fmm.fmm_wrapper import direct_eval
import scipy.spatial

def test_it():
    np.random.seed(1111)
    params = [1.0, 0.25]
    n_obs = 1000
    n_src = 1000
    obs_pts = np.random.rand(n_obs, 3)
    obs_ns = np.random.rand(n_obs, 3)
    src_pts = np.random.rand(n_src, 3)
    src_ns = np.random.rand(n_src, 3)
    input = np.random.rand(n_src * 3)

    dist = scipy.spatial.distance.cdist(obs_pts, src_pts)
    print(np.min(dist))

    for K in ['elasticU', 'elasticT', 'elasticA', 'elasticH']:
        print('starting ' + K)
        out_p2p_mat = direct_eval(K, obs_pts, obs_ns, src_pts, src_ns, params)
        out_p2p = out_p2p_mat.reshape((n_obs * 3, n_src * 3)).dot(input)
        out_tct = farfield_pts_wrapper(K, obs_pts, obs_ns, src_pts, src_ns, input, params)
        error = np.abs((out_tct - out_p2p) / out_tct)
        import seaborn
        import matplotlib.pyplot as plt
        seaborn.distplot(np.log10(error))
        plt.show()
        print(np.max(error))
        # np.testing.assert_almost_equal(error, 0, 4)
        # import ipdb; ipdb.set_trace()
