import numpy as np
import scipy.spatial
import seaborn
import matplotlib.pyplot as plt

from sphere_quad import lebedev, surrounding_surface_sphere

from tectosaur.fmm.surrounding_surf import surrounding_surf
from tectosaur.fmm.c2e import direct_matrix
from tectosaur.fmm.cfg import make_config

def plot_pt_distances(pts):
    src_r = np.sqrt(np.sum(pts ** 2, axis = 1))
    seaborn.distplot(src_r)
    plt.show()

def random_box_pts(dim, n, w):
    pts = np.random.rand(n, dim) * 2 * w - w
    ns = pts / np.linalg.norm(pts, axis = 1)[:,np.newaxis]
    return pts, ns

def plot_err_distribution(err):
    plt.figure(figsize = (12,12))
    seaborn.distplot(np.log10(np.abs(cor_filtered - obs_filtered) / np.abs(cor_filtered)))
    plt.show()

def calc_translation_error(basis, E, C, MAC):
    print(basis[0].shape[0], E, C, MAC)

    n = 1000

    k_name = "laplaceH2"
    params = np.array([1.0, 0.25])
    fmm_cfg = make_config(k_name, params, 1.1, 3.0, 1, np.float64)

    src_pts, src_ns  = random_box_pts(fmm_cfg.K.spatial_dim, n, 1.0 / np.sqrt(fmm_cfg.K.spatial_dim))
    obs_pts, obs_ns  = random_box_pts(fmm_cfg.K.spatial_dim, n, 10.0)

    def mat_fnc(op, on, sp, sn):
        return direct_matrix(
            fmm_cfg.gpu_module, fmm_cfg.K, op, on, sp, sn, params, fmm_cfg.float_type
        )

    basis_pts, basis_wts = basis
    check_surf = C * basis_pts
    equiv_surf = E * basis_pts
    surf_normals = basis_pts

    src_to_check = mat_fnc(check_surf, surf_normals, src_pts, src_ns)
    equiv_to_check = mat_fnc(check_surf, surf_normals, equiv_surf, surf_normals)
    equiv_to_obs = mat_fnc(obs_pts, obs_ns, equiv_surf, surf_normals)

    # SHOULD USE THE ORIGINAL FACTORS FROM THE SVD RATHER THAN COMPUTING
    # A PSEUDOINVERSE SINCE THIS MAINTAINS NUMERICAL PRECISION.
    c2e_svd = list(np.linalg.svd(equiv_to_check))
    c2e_svd[1] = 1.0 / c2e_svd[1]
    obs_vals = equiv_to_obs.dot(c2e_svd[2].T.dot(
        np.diag(c2e_svd[1]).dot(
            c2e_svd[0].T.dot(src_to_check.dot(
                np.ones(fmm_cfg.K.tensor_dim * src_pts.shape[0])
            ))
        )
    ))

    input = np.ones(fmm_cfg.K.tensor_dim * src_pts.shape[0])
    correct = mat_fnc(obs_pts, obs_ns, src_pts, src_ns).dot(input)
    # from tectosaur.ops.sparse_integral_op import farfield_pts_wrapper
    # correct = farfield_pts_wrapper(k_name, obs_pts, obs_ns, src_pts, src_ns, input, params)
    obs_r = np.sqrt(np.sum(obs_pts ** 2, axis = 1))

    obs_filtered = obs_vals[obs_r.repeat(fmm_cfg.K.tensor_dim) > MAC]
    cor_filtered = correct[obs_r.repeat(fmm_cfg.K.tensor_dim) > MAC]

    M = np.mean(np.abs(cor_filtered))
    err = np.abs((obs_filtered - cor_filtered)) / M
    max_err = np.max(err)
    mean_err = np.mean(err)
    print("MEAN: " + str(mean_err))
    print("MAX: " + str(max_err))
    return mean_err, max_err

if __name__ == '__main__':

    C = 3.0
    # basis_f = lambda p: surrounding_surface_sphere(lebedev(p)[0].shape[0])
    basis_f = lambda p: (surrounding_surf(p, 2), np.ones(p))
    # basis_f = lebedev
    # for basis_f in [lebedev, lambda p: surrounding_surface_sphere(lebedev(p)[0].shape[0])]:
    for MAC in [3.0]:
        np.random.seed(10)
        maxerrs = []
        meanerrs = []
        out_p = []

        # ps = np.arange(10, 300, 40)

        ps = np.arange(3, 50, 3).astype(np.int)
        for p in ps:
            print(p)
            # C = 3.0
            E = 1.1
            basis = basis_f(p)
            out_p.append(basis[0].shape[0])
            maxe, meane = calc_translation_error(basis, E, C, MAC)
            maxerrs.append(maxe)
            meanerrs.append(meane)

        plt.plot(out_p, np.log10(maxerrs), label = str(basis_f.__name__) + '   ' + str(C))
        # plt.plot(ps, np.log10(meanerrs))
    plt.legend()
    plt.show()


















    # plt.pcolor(pCs, Cs, np.log10(np.abs(err)))
    # plt.xlim([pc[0], pc[-1]])
    # plt.ylim([c[0], c[-1]])
    # plt.xlabel('$p$')
    # plt.ylabel('$C$')
    # cbar = plt.colorbar()
    # cbar.set_label('$\log_{10} \\textrm{error}$')
    # plt.show()

    # plt.plot([C,C],[-50, 50])
    # plt.plot([E,E],[-50, 50])
    # plt.plot(obs_r, error, '.b')
    # plt.plot(obs_r, correct, '.r')
    # plt.ylim([np.min(error), np.max(error)])
    # plt.show()

    # check_to_equiv = np.linalg.pinv(equiv_to_check, rcond = 1e-13)
    # src_to_equiv = check_to_equiv.dot(src_to_check)
    # src_to_obs = equiv_to_obs.dot(src_to_equiv)
    # obs_vals = src_to_obs.dot(np.ones(fmm_cfg.tensor_dim * src_pts.shape[0]))
