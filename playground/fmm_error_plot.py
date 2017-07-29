import numpy as np
import scipy.spatial
import seaborn
import matplotlib.pyplot as plt

from sphere_quad import lebedev, surrounding_surface_sphere

import tectosaur_fmm.fmm_wrapper as fmm

def plot_pt_distances(pts):
    src_r = np.sqrt(np.sum(pts ** 2, axis = 1))
    seaborn.distplot(src_r)
    plt.show()

def random_box_pts(n, w):
    pts = np.random.rand(n, 3) * 2 * w - w
    ns = pts / np.linalg.norm(pts, axis = 1)[:,np.newaxis]
    return pts, ns

def plot_err_distribution(err):
    plt.figure(figsize = (12,12))
    seaborn.distplot(np.log10(np.abs(cor_filtered - obs_filtered) / np.abs(cor_filtered)))
    plt.show()

def calc_translation_error(basis, E, C, MAC):
    print(basis[0].shape[0], E, C, MAC)

    n = 1000
    src_pts, src_ns  = random_box_pts(n, 1.0 / np.sqrt(3))
    obs_pts, obs_ns  = random_box_pts(n, 10.0)

    k_name = "elasticH"
    params = np.array([1.0, 0.25])
    tensor_dim = 3

    def mat_fnc(op, on, sp, sn):
        return fmm.direct_eval(
            k_name, op, on, sp, sn, params
        ).reshape((tensor_dim * op.shape[0], tensor_dim * sp.shape[0]))

    basis_pts, basis_wts = basis
    check_surf = C * basis_pts
    check_wts = 4 * np.pi * C ** 2 * np.repeat(basis_wts, tensor_dim)[:, np.newaxis]
    equiv_surf = E * basis_pts
    equiv_wts = 4 * np.pi * E ** 2 * np.repeat(basis_wts, tensor_dim)[np.newaxis, :]
    surf_normals = basis_pts

    src_to_check = mat_fnc(check_surf, surf_normals, src_pts, src_ns)
    src_to_check *= check_wts
    equiv_to_check = mat_fnc(check_surf, surf_normals, equiv_surf, surf_normals)
    equiv_to_check *= check_wts
    equiv_to_check *= equiv_wts
    equiv_to_obs = mat_fnc(obs_pts, obs_ns, equiv_surf, surf_normals)
    equiv_to_obs *= equiv_wts

    # SHOULD USE THE ORIGINAL FACTORS FROM THE SVD RATHER THAN COMPUTING
    # A PSEUDOINVERSE SINCE THIS MAINTAINS NUMERICAL PRECISION.
    c2e_svd = list(np.linalg.svd(equiv_to_check))
    c2e_svd[1] = 1.0 / c2e_svd[1]
    obs_vals = equiv_to_obs.dot(c2e_svd[2].T.dot(
        np.diag(c2e_svd[1]).dot(
            c2e_svd[0].T.dot(src_to_check.dot(
                np.ones(tensor_dim * src_pts.shape[0])
            ))
        )
    ))

    input = np.ones(tensor_dim * src_pts.shape[0])
    correct = mat_fnc(obs_pts, obs_ns, src_pts, src_ns).dot(input)
    # from tectosaur.ops.sparse_integral_op import farfield_pts_wrapper
    # correct = farfield_pts_wrapper(k_name, obs_pts, obs_ns, src_pts, src_ns, input, params)
    obs_r = np.sqrt(np.sum(obs_pts ** 2, axis = 1))

    obs_filtered = obs_vals[obs_r.repeat(tensor_dim) > MAC]
    cor_filtered = correct[obs_r.repeat(tensor_dim) > MAC]

    M = np.mean(np.abs(cor_filtered))
    err = np.abs((obs_filtered - cor_filtered)) / M
    max_err = np.max(err)
    mean_err = np.mean(err)
    print("MEAN: " + str(mean_err))
    print("MAX: " + str(max_err))
    return mean_err, max_err

if __name__ == '__main__':

    C = 3.0
    basis_f = lambda p: surrounding_surface_sphere(lebedev(p)[0].shape[0])
    # basis_f = lebedev
    # for basis_f in [lebedev, lambda p: surrounding_surface_sphere(lebedev(p)[0].shape[0])]:
    for MAC in [3.0]:
        np.random.seed(10)
        maxerrs = []
        meanerrs = []
        out_p = []

        # ps = np.arange(10, 300, 40)

        ps = np.arange(3, 20, 3).astype(np.int)
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
    # obs_vals = src_to_obs.dot(np.ones(tensor_dim * src_pts.shape[0]))
