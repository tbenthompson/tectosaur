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



# With sufficient number of steps, the error seems to be mostly independent of x_start and there doesn't appear to be any interaction between the x_start value and the error_mag.
include_log = True
x_start = 0.1
x_step = 2.0
n_x = 7
error_mag = 1e-14


vals = []
for i in range(1000):
    model = lambda x: 1.0 + np.cos(x) * np.sin(x)
    xs = x_start * (x_step ** (-np.arange(n_x)))

    error_term = (2 * np.random.rand(n_x) - 1) * error_mag
    out = model(xs) + error_term
    extrap = limit(xs, out, True)
    vals.append(extrap[0])
vals = np.array(vals)
log_mag_error = np.log10(np.abs(vals - 1.0))
log_mag_error[np.isinf(log_mag_error)] = -16

print(error_mag)
print(np.max(log_mag_error))
print(np.min(log_mag_error))
plt.hist(log_mag_error)
plt.show()
