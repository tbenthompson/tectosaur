import matplotlib.pyplot as plt
import numpy as np

def make_terms(n_terms, include_log):
    poly_terms = n_terms
    terms = []
    if include_log:
        poly_terms -= 1
        terms = [lambda e: np.log(e)]
    for i in range(poly_terms):
        terms.append(lambda e, i=i: e ** i)
    return terms

def limit(eps_vals, f_vals, include_log):
    terms = make_terms(len(eps_vals), include_log)
    mat = [[t(e) for t in terms] for e in eps_vals]
    coeffs = np.linalg.solve(mat, f_vals)
    if include_log:
        return coeffs[1], coeffs[0]
    else:
        return coeffs[0]



include_log = True
x_start = 0.01
x_step = 2.0
n_x = 10
error_mag = 0.00000000000001


vals = []
for i in range(1000):
    model = lambda x: 1.0 + x + x ** 2 + 0.1 * x ** 3 + 0.08 * x ** 9
    xs = x_start * (x_step ** (-np.arange(n_x)))

    error_term = (2 * np.random.rand(n_x) - 1) * error_mag
    out = model(xs) + error_term
    extrap = limit(xs, out, True)
    vals.append(extrap[0])
vals = np.array(vals)
log_mag_error = np.log10(np.abs(vals - 1.0))

print(error_mag)
plt.hist(log_mag_error)
plt.show()
