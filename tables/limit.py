import numpy as np

def make_terms(n_terms, include_log):
    poly_terms = n_terms
    terms = []
    if include_log:
        poly_terms -= 1
        terms = [
            lambda e: np.log(e),
        ]
    for i in range(poly_terms):
        terms.append(lambda e, i=i: e ** i)
    return terms

def limit(eps_vals, f_vals, include_log):
    terms = make_terms(len(eps_vals), include_log)
    mat = [[t(e) for t in terms] for e in eps_vals]
    coeffs = np.linalg.solve(mat, f_vals)
    if include_log:
        return coeffs[1]
    else:
        return coeffs[0]

def richardson_limit(step_ratio, values):
    n_steps = len(values)
    last_level = values
    this_level = None

    for m in range(1, n_steps):
        this_level = []
        for i in range(n_steps - m):
            mult = step_ratio ** m
            factor = 1.0 / (mult - 1.0)
            low = last_level[i]
            high = last_level[i + 1]
            moreacc = factor * (mult * high - low)
            this_level.append(moreacc)
        last_level = this_level
    return this_level[0]

def calc_I(eps):
    tri1 = [[0,0,0],[1,0,0],[0,1,0.0]]
    tri2 = [[1,0,0],[0,0,0],[0,-1,0]]
    tol = 1e-5
    rho_order = 80

    import cppimport
    adaptive_integrate = cppimport.imp('adaptive_integrate')
    import tectosaur.quadrature as quad
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

if __name__ == '__main__':
    play(0.001, 5)
    play(0.0001, 3)
