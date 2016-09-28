import numpy as np
import cppimport
adaptive_integrate = cppimport.imp('adaptive_integrate')
import tectosaur.quadrature as quad
from coincident import richardson_limit

tri1 = [[0,0,0],[1,0,0],[0,1,0.0]]
tri2 = [[1,0,0],[0,0,0],[0,-1,0]]
tol = 1e-5
rho_order = 80
rho_gauss = quad.gaussxw(rho_order)

def make_terms(extrap_order):
    terms = [
        lambda e: np.log(e),
    ]
    for i in range(extrap_order + 1):
        terms.append(lambda e, i=i: e ** i)
    return terms

def calc_I(eps):
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
def run(starting_eps, n_steps):
    eps = [starting_eps]
    vals = [calc_I(eps[0])]
    print("START")
    for i in range(n_steps):
        eps.append(eps[-1] / 2.0)
        vals.append(calc_I(eps[-1]))

        terms = make_terms(i + 1)
        mat = [[t(e) for t in terms[:-1]] for e in eps]
        print(np.linalg.cond(mat))
        coeffs = np.linalg.solve(mat, vals)

        print("log coeff: " + str(coeffs[0]))
        result = coeffs[1]
        print("extrap to 0: " + str(result))

    # mat = [[t(e) for t in terms[1:]] for e in eps]
    # print(np.linalg.cond(mat))
    # coeffs = np.linalg.solve(mat, vals)
    # print(coeffs[0])
    # print(richardson_limit(2.0, vals))

# run(0.1, 7)
# run(0.01, 5)
run(0.001, 5)
run(0.0001, 3)
