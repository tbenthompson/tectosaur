import numpy as np

import tectosaur.quadrature as quad
from tectosaur.dense_integral_op import DenseIntegralOp

import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')

def calc_I(eps):
    tri1 = [[0,0,0],[1,0,0],[0,1,0.0]]
    tri2 = [[1,0,0],[0,0,0],[0,-1,0]]
    tol = 1e-5
    rho_order = 80

    rho_gauss = quad.gaussxw(rho_order)
    rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
    return adaptive_integrate.integrate_coincident(
        "H", tri1, tol, eps, 1.0, 0.25,
        rho_q[0].tolist(), rho_q[1].tolist()
    )[0]

    # Basis 1 and 0 divergence should cancel between coincident and adjacent
    # return adaptive_integrate.integrate_coincident(
    #     "H", tri1, tol, eps, 1.0, 0.25,
    #     rho_q[0].tolist(), rho_q[1].tolist()
    # )[3]
    # return adaptive_integrate.integrate_adjacent(
    #     "H", tri1, tri2,
    #     tol, eps, 1.0, 0.25, rho_q[0].tolist(), rho_q[1].tolist()
    # )[0]

# for starting_eps in [0.1,0.05,0.025,0.01,0.001]:
def play(starting_eps, n_steps):
    eps = [starting_eps]
    vals = [calc_I(eps[0])]
    print("START")
    for i in range(n_steps):
        eps.append(eps[-1] / 2.0)
        vals.append(calc_I(eps[-1]))

        terms = make_terms(i + 2, True)
        mat = [[t(e) for t in terms] for e in eps]
        print(np.linalg.cond(mat))
        coeffs = np.linalg.solve(mat, vals)

        print("log coeff: " + str(coeffs[0]))
        result = coeffs[1]
        print("extrap to 0: " + str(result))

def test_barycentric_interp():
    pts = cheb(-1,1,5)
    wts = cheb_wts(-1,1,5)
    vals = [0,1,0,1,0]
    xs = np.linspace(-1,1,100)
    interp_vals = [barycentric_eval(pts, wts, vals, x) for x in xs]
    import matplotlib.pyplot as plt
    plt.plot(pts, vals, 'o')
    plt.plot(xs, interp_vals)
    plt.show()

def test_barycentric_interp2d():
    import matplotlib.pyplot as plt
    for N in range(3, 40, 1):
        pts1d = cheblob(-1,1,N)
        X,Y = np.meshgrid(pts1d, pts1d)
        pts = np.array([X.ravel(), Y.ravel()]).T
        wts1d = cheblob_wts(-1,1,N)
        wts = np.outer(wts1d, wts1d).ravel()

        f = lambda xs: np.sin((xs[:,0] + xs[:,1] - 1.0) * 5)
        vals = f(pts)


        ne = 100
        xs = np.linspace(-1, 1, ne)
        xe,ye = np.meshgrid(xs, xs)
        xhat = np.array([xe.ravel(), ye.ravel()]).T
        interp_vals = barycentric_evalnd(pts, wts, vals, xhat).reshape((ne,ne))

        plt.imshow(interp_vals)
        plt.title(N)
        plt.colorbar()
        plt.show()

def test_barycentric_interp3d():
    import matplotlib.pyplot as plt
    for N in range(5,18,2):
        pts1d = cheb(-1,1,N)
        X,Y,Z = np.meshgrid(pts1d, pts1d, pts1d)
        pts = np.array([X.ravel(), Y.ravel(), Z.ravel()]).T
        wts1d = cheb_wts(-1,1,N)
        wts = np.outer(wts1d, np.outer(wts1d, wts1d)).ravel()

        f = lambda xs: np.sin(xs[:,0] ** 3 - xs[:,1] * xs[:,2])
        vals = f(pts)

        ne = 30
        xs = np.linspace(-1, 1, ne)
        xe,ye,ze = np.meshgrid(xs, xs, xs)
        xhat = np.array([xe.ravel(), ye.ravel(), ze.ravel()]).T
        correct = f(xhat)
        interp_vals = barycentric_evalnd(pts, wts, vals, xhat).reshape((ne,ne,ne))
        print(np.max(correct - interp_vals.flatten()))

    import ipdb; ipdb.set_trace()

def test_rotation():
    sm = 1.0

    def test_tri(tri):
        standard_tri, labels, R, factor = standardize.standardize(np.array(tri))
        np.testing.assert_almost_equal(
            standard_tri,
            [[0,0,0],[1,0,0],[standard_tri[2][0],standard_tri[2][1],0]]
        )

        correct_full = calc(
            K, tri, rho_order, tol, eps_start, eps_step, n_steps, sm, pr
        ).reshape((3,3,3,3))

        standard_full = calc(
            K, standard_tri.tolist(), rho_order, tol, eps_start, eps_step, n_steps, 1.0, pr
        ).reshape((3,3,3,3))

        for sb1 in range(3):
            for sb2 in range(3):
                standardized = R.T.dot(standard_full[sb1,:,sb2,:]).dot(R) / (sm * factor ** 3)
                cb1 = labels[sb1]
                cb2 = labels[sb2]
                correct = correct_full[cb1,:,cb2,:]
                np.testing.assert_almost_equal(correct, standardized, 4)

        # np.testing.assert_almost_equal(correct, standardized, 4)

    test_tris = [
        [[0.0,0.0,0.0], [1.1,0.0,0.0], [0.4,0.3,0.0]]
        ,[[0.0,0.0,0.0], [0.0,1.1,0.0], [-0.3,0.4,0.0]]
        ,[[0.0,0.0,0.0], [0.0,0.0,1.1], [0.0,-0.3,0.4]]
        ,[[0.0,0.0,0.0], [0.0,0.3,1.1], [0.0,-0.3,0.4]]
        ,[[0.0,0.0,0.0], [0.0,-0.3,0.4], [0.0,0.35,1.1]]
        ,[[0.0,0.35,1.1], [0.0,0.0,0.0], [0.0,-0.3,0.4]]
        ,[[0.0, -0.3, 0.4], [0.0,0.35,1.1], [0.0,0.0,0.0]]
        ,[[1.0,0.0,0.0], [0.0,-0.3,0.45], [0.0,0.35,1.1]]
    ]
    for t in test_tris:
        test_tri(t)

    n_checks = 10
    while True:
        tri = np.random.rand(3,3).tolist()

        try:
            test_tri(tri)
            print("SUCCESS!")
        except Exception as e:
            # Exception implies the triangle was malformed (angle < 20 degrees)
            continue

        n_checks -= 1
        if n_checks == 0:
            break

def test_vert_adj():
    K = 'H'
    for abc in np.linspace(0.1, 0.9, 10):
        pts = np.array([[0,0,0],[1,0,0],[abc,1 - abc,0],[0,1,0],[0,-1,0],[0.5,-0.5,0]])
        tris = np.array([[0,2,3],[0,4,1]])
        op = DenseIntegralOp([0.01], 10, 10, 13, 10, 10, 3.0, K, 1.0, 0.25, pts, tris)
        res = adaptive_integrate.integrate_no_limit(
            K, pts[tris[0]].tolist(), pts[tris[1]].tolist(), 0.0001, 1.0, 0.25
        )
        np.testing.assert_almost_equal(res, op.mat[:9,9:].reshape(81), 5)
        print("PASS: " + str(abc))

def test_edge_adj():
    K = 'H'
    for abc in np.linspace(0.1, 0.9, 10):
        eps = 0.08
        pts = np.array([[0,0,0],[1,0,0],[abc,1 - abc,0],[0,1,0],[0,-1,0],[0.5,-0.5,0]])
        tris = np.array([[0,1,2],[1,0,4]])
        op = DenseIntegralOp([eps], 10, 15, 10, 10, 10, 3.0, K, 1.0, 0.25, pts, tris)

        rho_order = 100
        tol = 0.005
        rho_gauss = quad.gaussxw(rho_order)
        rho_q = quad.sinh_transform(rho_gauss, -1, eps * 2)
        res = adaptive_integrate.integrate_adjacent(
            K, pts[tris[0]].tolist(), pts[tris[1]].tolist(), tol, eps, 1.0, 0.25,
            rho_q[0].tolist(), rho_q[1].tolist()
        )
        np.testing.assert_almost_equal(res, op.mat[:9,9:].reshape(81), 4)
        print("PASS: " + str(abc))


if __name__ == '__main__':
    # test_vert_adj()
    test_edge_adj()
