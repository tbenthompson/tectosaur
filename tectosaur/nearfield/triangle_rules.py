import tectosaur.util.quadrature as quad
from tectosaur.util.paget import paget
import numpy as np

def vertex_interior_quad(n_theta, n_rho, a):
    # return quad.gauss2d_tri(10)
    theta_quad = quad.map_to(quad.gaussxw(n_theta), [0, np.pi / 2])
    if a == 0:
        # rho_quad = quad.telles_singular(n_rho, -1)
        # rho_quad = quad.gaussxw(n_rho)
        rho_quad = quad.tanh_sinh(n_rho)
    else:
        rho_quad = paget(n_rho, a)
    pts = []
    wts = []
    for t, tw in zip(*theta_quad):
        rho_max = 1.0 / (np.cos(t) + np.sin(t))
        q = quad.map_to(rho_quad, [0, rho_max])
        for r, rw in zip(*q):
            assert(r > 0)
            srcxhat = r * np.cos(t)
            srcyhat = r * np.sin(t)
            pts.append((srcxhat, srcyhat))
            # The r**2 comes from two factors:
            # 1) The first r comes from the jacobian for the polar transform
            # 2) The second r corrects for the implicit 1/r in the quad rule
            weight = tw * rw * (r ** (a + 1))
            wts.append(weight)
    return np.array(pts), np.array(wts)

def vertex_adj_quad(n_theta, n_beta, n_alpha):
    theta_quad = quad.gaussxw(n_theta)
    beta_quad = quad.gaussxw(n_beta)
    alpha_quad = quad.gaussxw(n_alpha)

    def rho_max(theta):
        return 1.0 / (np.cos(theta) + np.sin(theta))

    pts = []
    wts = []
    def alpha_integral(w, tp, tq, b, alpha_max):
        q = quad.map_to(alpha_quad, [0, alpha_max])
        for a, aw in zip(*q):
            jacobian = a ** 3 * np.cos(b) * np.sin(b)
            rho_p = a * np.cos(b)
            rho_q = a * np.sin(b)
            obsxhat = rho_p * np.cos(tp)
            obsyhat = rho_p * np.sin(tp)
            srcxhat = rho_q * np.cos(tq)
            srcyhat = rho_q * np.sin(tq)

            pts.append((obsxhat, obsyhat, srcxhat, srcyhat))
            wts.append(jacobian * aw * w)

    def beta_integral(w, tp, tq, beta_min, beta_max, alpha_max_fnc):
        q = quad.map_to(beta_quad, [beta_min, beta_max])
        for b, bw in zip(*q):
            alpha_max = alpha_max_fnc(b)
            alpha_integral(bw * w, tp, tq, b, alpha_max)

    def theta_q_integral(w, tp):
        q = quad.map_to(theta_quad, [0, np.pi / 2])
        for tq, tqw in zip(*q):
            beta_split = np.arctan(rho_max(tq) / rho_max(tp))
            beta_integral(w * tqw, tp, tq, 0, beta_split,
                lambda b: rho_max(tp) / np.cos(b))
            beta_integral(w * tqw, tp, tq, beta_split, np.pi / 2,
                lambda b: rho_max(tq) / np.sin(b))

    def theta_p_integral():
        q = quad.map_to(theta_quad, [0, np.pi / 2])
        for tp, tpw in zip(*q):
            theta_q_integral(tpw, tp)

    theta_p_integral()
    return np.array(pts), np.array(wts)

def coincident_quad(nq):
    """
    Coincident quadrature rule from Erichsen and Sauter (1998):
    Efficient automatic quadrature in 3D Galerkin BEM
    """
    qtri = quad.gauss2d_tri(nq)
    qline = quad.map_to(quad.gaussxw(nq), [0.0, 1.0])

    mappings = [
        lambda eta, w1, w2, w3: (
            w1 + w2 + w3,
            w1 + w2,
            w1 * (1 - eta) + w2 + w3,
            w2
        ),
        lambda eta, w1, w2, w3: (
            w1 * (1 - eta) + w2 + w3,
            w1 * (1 - eta) + w2,
            w1 + w2 + w3,
            w2
        ),
        lambda eta, w1, w2, w3: (
            w1 + w2 + w3,
            w1 * eta + w2,
            w2 + w3,
            w2
        ),
        lambda eta, w1, w2, w3: (
            w1 * (1 - eta) + w2 + w3,
            w2,
            w1 + w2 + w3,
            w1 + w2,
        ),
        lambda eta, w1, w2, w3: (
            w1 + w2 + w3,
            w2,
            w1 * (1 - eta) + w2 + w3,
            w1 * (1 - eta) + w2
        ),
        lambda eta, w1, w2, w3: (
            w2 + w3,
            w2,
            w1 + w2 + w3,
            w1 * eta + w2
        )
    ]

    pts = []
    wts = []
    for i in range(len(mappings)):
        m = mappings[i]
        for w12_idx in range(qtri[0].shape[0]):
            w1 = qtri[0][w12_idx,0]
            w2 = qtri[0][w12_idx,1]
            f12 = qtri[1][w12_idx]
            for x3, f3 in zip(*qline):
                w3 = x3 * (1 - w1 - w2)
                f3 *= (1 - w1 - w2)
                for eta, feta in zip(*qline):
                    F = f12 * f3 * feta * w1
                    x1, x2, y1, y2 = m(eta, w1, w2, w3)
                    obsxhat = 1 - x1
                    obsyhat = x2
                    srcxhat = 1 - y1
                    srcyhat = y2

                    pts.append((obsxhat, obsyhat, srcxhat, srcyhat))
                    wts.append(F)
    return np.array(pts), np.array(wts)

def edge_adj_quad(nq):
    """
    Edge adjacent quadrature rule from Sauter and Schwab book.
    """
    qtri = quad.gauss2d_tri(nq)
    qline = quad.map_to(quad.gaussxw(nq), [0.0, 1.0])

    mappings = [
        lambda e1, e2, e3, P: (
            P,
            P * e1 * e3,
            P * (1 - e1 * e2),
            P * e1 * (1 - e2),
            P ** 3 * e1 ** 2
        ),
        lambda e1, e2, e3, P: (
            P,
            P * e1,
            P * (1 - e1 * e2 * e3),
            P * e1 * e2 * (1 - e3),
            P ** 3 * e1 ** 2 * e2
        ),
        lambda e1, e2, e3, P: (
            P * (1 - e1 * e2),
            P * e1 * (1 - e2),
            P,
            P * e1 * e2 * e3,
            P ** 3 * e1 ** 2 * e2
        ),
        lambda e1, e2, e3, P: (
            P * (1 - e1 * e2 * e3),
            P * e1 * e2 * (1 - e3),
            P,
            P * e1,
            P ** 3 * e1 ** 2 * e2
        ),
        lambda e1, e2, e3, P: (
            P * (1 - e1 * e2 * e3),
            P * e1 * (1 - e2 * e3),
            P,
            P * e1 * e2,
            P ** 3 * e1 ** 2 * e2
        ),
    ]

    pts = []
    wts = []
    for i in range(len(mappings)):
        m = mappings[i]
        for e1, f1 in zip(*qline):
            for e2, f2 in zip(*qline):
                for e3, f3 in zip(*qline):
                    for P, f4 in zip(*qline):
                        F = f1 * f2 * f3 * f4
                        x1, x2, y1, y2, jac = m(e1, e2, e3, P)
                        F *= jac

                        obsxhat = 1 - x1
                        obsyhat = x2
                        srcxhat = 1 - y1
                        srcyhat = y2
                        srcxhat = (1 - srcyhat) - srcxhat

                        pts.append((obsxhat, obsyhat, srcxhat, srcyhat))
                        wts.append(F)
    return np.array(pts), np.array(wts)
