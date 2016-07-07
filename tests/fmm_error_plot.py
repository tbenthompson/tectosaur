import numpy as np
import scipy.spatial
import seaborn

import cppimport
fmm = cppimport.imp("tectosaur.fmm.fmm").fmm.fmm

def surrounding_surface_sphere(order):
    pts = []
    a = 4 * np.pi / order;
    d = np.sqrt(a);
    M_theta = int(np.round(np.pi / d))
    d_theta = np.pi / M_theta;
    d_phi = a / d_theta;
    for m in range(M_theta):
        theta = np.pi * (m + 0.5) / M_theta;
        M_phi = int(np.round(2 * np.pi * np.sin(theta) / d_phi))
        for n in range(M_phi):
            phi = 2 * np.pi * n / M_phi;
            x = np.sin(theta) * np.cos(phi);
            y = np.sin(theta) * np.sin(phi);
            z = np.cos(theta);
            pts.append((x, y, z))
    return np.array(pts)

def main():
    import matplotlib.pyplot as plt


    src_pts = (np.random.rand(1000, 3) * 2.0 - 1.0) / np.sqrt(3)
    obs_pts = np.random.rand(1000, 3) * 20.0 - 10.0
    obs_ns = obs_pts / np.linalg.norm(obs_pts, axis = 1)[:,np.newaxis]
    src_ns = src_pts / np.linalg.norm(src_pts, axis = 1)[:,np.newaxis]
    # src_r = np.sqrt(np.sum(src_pts ** 2, axis = 1))
    # seaborn.distplot(src_r)
    # plt.show()

    basis_fnc = surrounding_surface_sphere

    # k_name = "invr"
    # params = []
    # tensor_dim = 1

    k_name = "elasticH"
    params = np.array([1.0, 0.25])
    tensor_dim = 3
    def mat_fnc(op, on, sp, sn):
        return fmm.direct_eval(k_name, op, on, sp, sn, params)\
            .reshape((tensor_dim * op.shape[0], tensor_dim * sp.shape[0]))

    pc = np.arange(20, 450, 50)
    # c = np.linspace(1.5, 6.0, 10)
    c = np.array([3.0])
    pCs, Cs = np.meshgrid(pc, c)
    err = np.zeros((c.shape[0],pc.shape[0]))
    for i in range(c.shape[0]):
        for j in range(pc.shape[0]):
            pC = pCs[i, j]
            C = Cs[i, j]
            MAC = C
            # TRIED USING p_check > p_equiv and it's distinctly NOT useful. Perhaps
            # a change in spherical basis could be useful.
            # Seems like a equivalent surface radius any larger is useless. This can't
            # below 1.0 because then the equivalent surface would be outside some of
            # the points of interest.
            E = 1.1
            print(C, pC)
            check_surf = C * basis_fnc(pC)
            equiv_surf = E * basis_fnc(pC)
            surf_normals = basis_fnc(pC)

            src_to_check = mat_fnc(check_surf, surf_normals, src_pts, src_ns)
            equiv_to_check = mat_fnc(check_surf, surf_normals, equiv_surf, surf_normals)
            equiv_to_obs = mat_fnc(obs_pts, obs_ns, equiv_surf, surf_normals)

            # SHOULD USE THE ORIGINAL FACTORS FROM THE SVD RATHER THAN COMPUTING
            # A PSEUDOINVERSE SINCE THIS MAINTAINS NUMERICAL PRECISION.
            c2e_svd = list(np.linalg.svd(equiv_to_check))
            c2e_svd[1] = c2e_svd[1] ** -1
            obs_vals = equiv_to_obs.dot(
                c2e_svd[2].T.dot(
                    np.diag(c2e_svd[1]).dot(
                        c2e_svd[0].T.dot(src_to_check.dot(np.ones(
                            tensor_dim * src_pts.shape[0]))))))

            correct = mat_fnc(obs_pts, obs_ns, src_pts, src_ns).dot(
                    np.ones(tensor_dim * src_pts.shape[0]))

            obs_r = np.sqrt(np.sum(obs_pts ** 2, axis = 1))
            obs_filtered = obs_vals[obs_r.repeat(tensor_dim) > MAC]
            cor_filtered = correct[obs_r.repeat(tensor_dim) > MAC]
            M = np.mean(np.abs(cor_filtered))
            max_err = np.max(np.abs((obs_filtered - cor_filtered))) / M
            mean_err = np.mean(np.abs((obs_filtered - cor_filtered))) / M
            print("MEAN: " + str(mean_err))
            print("MAX: " + str(max_err))
            seaborn.distplot(np.log10(np.abs(cor_filtered - obs_filtered) / np.abs(cor_filtered)))
            plt.show()
            err[i, j] = mean_err
    plt.plot(pCs[0,:], np.log10(np.abs(err[0,:])))
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

    # Gauss quadrature based check and equiv surfaces are dramatically inferior
    # to the uniform nystrom discretization of the sphere.
    # def basis_fnc(p):
    #     rt_p = int(np.ceil(np.sqrt(p)))
    #     uniform_theta = np.linspace(0, 2 * np.pi, rt_p + 1)[:-1]
    #     import tectosaur.quadrature
    #     # gauss_phi = (tectosaur.quadrature.gaussxw(rt_p)[0] + 1) * (np.pi / 2.0)
    #     gauss_phi = np.linspace(0, np.pi, rt_p)
    #     T, P = np.meshgrid(uniform_theta, gauss_phi)
    #     x = (np.cos(T) * np.sin(P)).flatten()
    #     y = (np.sin(T) * np.sin(P)).flatten()
    #     z = np.cos(P).flatten()
    #     pts = np.array([x, y, z]).T
    #     return pts

            # check_to_equiv = np.linalg.pinv(equiv_to_check, rcond = 1e-13)
            # src_to_equiv = check_to_equiv.dot(src_to_check)
            # src_to_obs = equiv_to_obs.dot(src_to_equiv)
            # obs_vals = src_to_obs.dot(np.ones(tensor_dim * src_pts.shape[0]))


if __name__ == '__main__':
    main()
