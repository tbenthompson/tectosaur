import sys
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

def limit_coeffs(eps_vals, f_vals, include_log):
    terms = make_terms(len(eps_vals), include_log)
    mat = [[t(e) for t in terms] for e in eps_vals]
    coeffs = np.linalg.solve(mat, f_vals)
    return coeffs

def limit(eps_vals, f_vals, include_log):
    coeffs = limit_coeffs(eps_vals, f_vals, include_log)
    if include_log:
        return coeffs[1]
    else:
        return coeffs[0]

def richardson_limit(step_ratio, values):
    n_steps = len(values)

    if n_steps == 1:
        return values[0]

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
