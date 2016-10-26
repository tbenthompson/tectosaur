import numpy as np
import scipy.linalg
import scipy.interpolate
from math import factorial

from tectosaur.tri_gauss import get_tri_gauss
from tectosaur.limit import limit

# Derives the n-point gauss quadrature rule
def gaussxw(n):
    k = np.arange(1.0, n)
    a_band = np.zeros((2, n))
    a_band[1, 0:n-1] = k / np.sqrt(4 * k * k - 1)
    x, V = scipy.linalg.eig_banded(a_band, lower = True)
    w = 2 * np.real(np.power(V[0,:], 2))
    return x, w

# Change the domain of integration for a quadrature rule
def map_to(qr, interval):
    x01 = (qr[0] + 1) / 2
    outx = interval[0] + (interval[1] - interval[0]) * x01
    outw = (qr[1] / 2) * (interval[1] - interval[0])
    return outx, outw

# Integrate!
def quadrature(f, qr):
    return sum(f(qr[0]) * qr[1])

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
        if include_log:
            I0 = limit(h_vals, y, include_log)[0]
        else:
            I0 = limit(h_vals, y, include_log)
        inner_ws *= I0

        if xs is None:
            xs = inner_xs
            ws = inner_ws
        else:
            xs = np.vstack((xs, inner_xs))
            ws = np.append(ws, inner_ws)
    return xs, ws

# Sinh transform for integrals of the form \int_{-1}^1 f(x)<F12>
# check that the b-distance(eps) is being computed properly, i think it should
# be in transformed coordinates rather than the [-1,1] initial domain
def sinh_transform(quad_rule, a, b, iterated = False):
    n_q = len(quad_rule[0])
    mu_0 = 0.5 * (np.arcsinh((1.0 + a) / b) + np.arcsinh((1.0 - a) / b))
    eta_0 = 0.5 * (np.arcsinh((1.0 + a) / b) - np.arcsinh((1.0 - a) / b))

    start_q = quad_rule
    if iterated:
        xs = np.empty(n_q)
        ws = np.empty(n_q)
        a_1 = eta_0 / mu_0
        b_1 = np.pi / (2 * mu_0)
        mu_1 = 0.5 * (np.arcsinh((1.0 + a_1) / b_1) + np.arcsinh((1.0 - a_1) / b_1))
        eta_1 = 0.5 * (np.arcsinh((1.0 + a_1) / b_1) - np.arcsinh((1.0 - a_1) / b_1));
        for i in range(n_q):
            u = quad_rule[0][i]
            xs[i] = a_1 + b_1 * np.sinh(mu_1 * u - eta_1);
            jacobian = b_1 * mu_1 * np.cosh(mu_1 * u - eta_1);
            ws[i] = quad_rule[1][i] * jacobian;
        start_q = (xs, ws)

    x = np.empty(n_q)
    w = np.empty(n_q)
    for i in range(n_q):
        s = start_q[0][i]
        x[i] = a + b * np.sinh(mu_0 * s - eta_0)
        jacobian = b * mu_0 * np.cosh(mu_0 * s - eta_0)
        w[i] = start_q[1][i] * jacobian
    return np.array(x), np.array(w)

def aimi_diligenti(quad_rule, p, q):
    n_q = len(quad_rule[0])
    x = np.zeros(n_q)
    w = np.zeros(n_q)
    n_transform_quad = int(np.floor((p + q) / 2.0)) ** 2
    transform_quad = gaussxw(n_transform_quad)
    for i in range(n_q):
        t = (quad_rule[0][i] + 1.0) / 2.0
        F = factorial(p + q - 1) / (factorial(p - 1) * factorial(q - 1))
        x[i] = 2 * F * quadrature(
            lambda us: [u ** (p - 1) * (1 - u) ** (q - 1) for u in us],
            map_to(transform_quad, [0, t])
        ) - 1.0
        w[i] = quad_rule[1][i] * F * t ** (p - 1) * (1 - t) ** (q - 1)
    return x, w

def poly_transform01(quad_rule):
    n_q = len(quad_rule[0])
    x = np.empty(n_q)
    w = np.empty(n_q)
    for i in range(n_q):
        s = (quad_rule[0][i] + 1.0) / 2.0
        x[i] = 2.0 * (-s**2 * (2 * s - 3)) - 1.0
        J = -2 * s * (2 * s - 3) - 2 * s ** 2
        w[i] = quad_rule[1][i] * J
    return np.array(x), np.array(w)

def gauss2d_tri(N):
    pts, wts = get_tri_gauss(N * 2 - 1)
    if pts is not None:
        return np.array(pts), np.array(wts) * 0.5
    qg = gaussxw(N)

    q_rect_x, q_rect_y = np.meshgrid(qg[0], qg[0])
    q_rect_pts = np.array([q_rect_x.flatten(), q_rect_y.flatten()]).T
    q_rect_w = np.outer(qg[1], qg[1]).flatten()

    q_x01 = (q_rect_pts[:,0] + 1) / 2
    q_y01 = (q_rect_pts[:,1] + 1) / 2
    q_tri_pts = np.empty_like(q_rect_pts)
    q_tri_pts[:,0] = q_x01 * (1 - q_y01)
    q_tri_pts[:,1] = q_y01
    q_tri_w = (q_rect_w / 4.0) * (1 - q_y01)
    return q_tri_pts, q_tri_w

def gauss4d_tri(N_outer, N_inner):
    q_tri_pts_outer, q_tri_w_outer = gauss2d_tri(N_outer)
    q_tri_pts_inner, q_tri_w_inner = gauss2d_tri(N_inner)
    pts = []
    w = []
    for i in range(q_tri_pts_outer.shape[0]):
        for j in range(q_tri_pts_inner.shape[0]):
            pts.append((
                q_tri_pts_outer[i,0], q_tri_pts_outer[i,1],
                q_tri_pts_inner[j,0], q_tri_pts_inner[j,1]
            ))
            w.append(q_tri_w_outer[i] * q_tri_w_inner[j])

    return np.array(pts), np.array(w)
