import numpy as np
import cppimport
adaptive_integrator = cppimport.imp('adaptive_integrator')

tri = [[0,0,0],[1,0,0],[0.4,0.6,0.0]]
tol = 1e-4

def make_terms(extrap_order):
    terms = [
        lambda e: np.log(e),
    ]
    for i in range(extrap_order + 1):
        terms.append(lambda e, i=i: e ** i)
    return terms

def calc_I(eps):
    return adaptive_integrator.integrate("U", tri, tol, eps, 1.0, 0.25)[0]

# for starting_eps in [0.1,0.05,0.025,0.01,0.001]:
starting_eps = 0.1
eps = [starting_eps]
vals = [calc_I(eps[0])]
print("START")
for i in range(8):
    eps.append(eps[-1] / 2.0)
    vals.append(calc_I(eps[-1]))

    terms = make_terms(i)
    mat = [[t(e) for t in terms] for e in eps]
    print(np.linalg.cond(mat))
    coeffs = np.linalg.solve(mat, vals)

    print("log coeff: " + str(coeffs[0]))
    result = coeffs[1]
    print("extrap to 0: " + str(result))

