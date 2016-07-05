
def invr_direct_mat(obs_pts, src_pts):
    return (1.0 / (scipy.spatial.distance.cdist(obs_pts, src_pts)))

def test_invr_multipole_accuracy():
    import matplotlib.pyplot as plt


    src_pts = np.random.rand(1000, 3) * 2.0 - 1.0
    src_r = np.sqrt(np.sum(src_pts ** 2, axis = 1))
    # plt.plot(src_r, '.')
    # plt.show()
    obs_pts = np.random.rand(1000, 3) * 20.0 - 10.0

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

    basis_fnc = surrounding_surface_sphere

    pc = np.arange(5, 400, 20)
    c = np.linspace(1.5, 6.0, 30)
    pCs, Cs = np.meshgrid(pc, c)
    err = np.zeros((c.shape[0],pc.shape[0]))
    for i in range(c.shape[0]):
        for j in range(pc.shape[0]):
            pC = pCs[i, j]
            C = Cs[i, j]
            # TRIED USING p_check > p_equiv and it's distinctly NOT useful. Perhaps
            # a change in spherical basis could be useful.
            pE = pC
            # Seems like a equivalent surface radius any larger is useless. This can't
            # below 1.0 because then the equivalent surface would be outside some of
            # the points of interest.
            E = 1.1
            print(C, pC)
            check_surf = C * basis_fnc(pC)
            equiv_surf = E * basis_fnc(pE)

            src_to_check = invr_direct_mat(check_surf, src_pts)
            equiv_to_check = invr_direct_mat(check_surf, equiv_surf)
            equiv_to_obs = invr_direct_mat(obs_pts, equiv_surf)
            # SHOULD USE THE ORIGINAL FACTORS FROM THE SVD RATHER THAN COMPUTING
            # A PSEUDOINVERSE SINCE THIS MAINTAINS NUMERICAL PRECISION.
            # check_to_equiv = np.linalg.pinv(equiv_to_check, rcond = 1e-15)
            # src_to_equiv = check_to_equiv.dot(src_to_check)
            # src_to_obs = equiv_to_obs.dot(src_to_equiv)
            # obs_vals = src_to_obs.dot(np.ones(src_pts.shape[0]))
            c2e_svd = list(np.linalg.svd(equiv_to_check))
            c2e_svd[1] = c2e_svd[1] ** -1
            obs_vals = equiv_to_obs.dot(
                c2e_svd[2].T.dot(
                    np.diag(c2e_svd[1]).dot(
                        c2e_svd[0].T.dot(src_to_check.dot(np.ones(src_pts.shape[0]))))))

            correct = invr_direct_mat(obs_pts, src_pts).dot(np.ones(src_pts.shape[0]))

            obs_r = np.sqrt(np.sum(obs_pts ** 2, axis = 1))
            obs_filtered = obs_vals[obs_r > C]
            cor_filtered = correct[obs_r > C]
            max_err = np.max(np.abs((obs_filtered - cor_filtered) / cor_filtered))
            print("MEAN: " + str(np.mean(np.abs((obs_filtered - cor_filtered) / cor_filtered))))
            print("MAX: " + str(max_err))
            err[i, j] = max_err
    plt.pcolor(pCs, Cs, np.log10(np.abs(err)))
    plt.xlim([pc[0], pc[-1]])
    plt.ylim([c[0], c[-1]])
    plt.xlabel('$p$')
    plt.ylabel('$C$')
    cbar = plt.colorbar()
    cbar.set_label('$\log_{10} \\textrm{error}$')
    plt.show()

    # plt.plot([C,C],[-50, 50])
    # plt.plot([E,E],[-50, 50])
    # plt.plot(obs_r, error, '.b')
    # plt.plot(obs_r, correct, '.r')
    # plt.ylim([np.min(error), np.max(error)])
    # plt.show()
    return

    def make_multipole_tester_pts(n, source):
        if source:
            return np.random.rand(n, 3) * 2 - 1.0
        else:
            return np.random.rand(n * 10, 3) * 30.0 - 15.0
    for p in range(2, 250, 20):
        pts, pts2, est = run_full(500, make_multipole_tester_pts, 100000000, p, "invr")
        correct = invr_direct_mat(pts, pts2).dot(np.ones(pts2.shape[0]))
        obs_r = np.sqrt(np.sum(pts ** 2, axis = 1))
        error = np.log10(np.abs((correct - est) / correct))
        plt.title(str(p))
        plt.plot(obs_r, error, '.')
        plt.show()


