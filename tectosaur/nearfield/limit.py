import numpy as np

def make_terms(n_terms, log_terms, inv_term = False):
    if log_terms > 0:
        assert(not inv_term)

    poly_terms = n_terms - log_terms - (1 if inv_term else 0)
    terms = []
    if inv_term:
        terms.append(lambda e: 1.0 / e)
    for i in range(log_terms):
        terms.append(lambda e, i=i: e ** i * np.log(e))
    for i in range(poly_terms):
        terms.append(lambda e, i=i: e ** i)
    return terms

def limit_coeffs(eps_vals, f_vals, log_terms, inv_term = False):
    terms = make_terms(len(eps_vals), log_terms, inv_term)
    mat = [[t(e) for t in terms] for e in eps_vals]
    coeffs = np.linalg.solve(mat, f_vals)
    return coeffs, terms

def limit_interp_eval(xs, coeffs, terms):
    return np.array([np.sum([c * t(x) for t,c in zip(terms, coeffs)]) for x in xs])

def limit(eps_vals, f_vals, log_terms, inv_term = False):
    coeffs, terms = limit_coeffs(eps_vals, f_vals, log_terms, inv_term)
    if log_terms > 0:
        return coeffs[log_terms], coeffs[0]
    elif inv_term:
        return coeffs[1], coeffs[0]
    else:
        return coeffs[0], 0

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

def aitken(seq, its = 1):
    assert(its > 0)
    if its > 1:
        S = aitken(seq, its - 1)
    else:
        S = seq
    return S[2:] - ((S[2:] - S[1:-1]) ** 2 / ((S[2:] - S[1:-1]) - (S[1:-1] - S[:-2])))

def richardson_quad(h_vals, include_log, quad_builder):
    n = len(h_vals)
    I = scipy.interpolate.BarycentricInterpolator(h_vals)
    xs = None
    ws = None
    for i in range(n):
        inner_xs, inner_ws = quad_builder(h_vals[i])
        if len(inner_xs.shape) == 1:
            inner_xs = inner_xs.reshape((inner_xs.shape[0], 1))
        inner_xs = np.hstack((
            inner_xs,
            np.array([[h_vals[i]]] * inner_xs.shape[0])
        ))

        y = [0] * n
        y[i] = 1.0
        n_log_terms = 1 if include_log else 0
        I0 = limit(h_vals, y, n_log_terms)[0]
        inner_ws *= I0

        if xs is None:
            xs = inner_xs
            ws = inner_ws
        else:
            xs = np.vstack((xs, inner_xs))
            ws = np.append(ws, inner_ws)
    return xs, ws
