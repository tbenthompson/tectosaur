import tectosaur.util.quadrature as quad
import numpy as np

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

def coincident_quad(n_theta, n_rho, n_outer):
    q_theta = quad.gaussxw(n_theta)
    q_rho = quad.gaussxw(n_rho)
    q_outer = quad.gauss2d_tri(n_outer)

    theta_lims = [
        lambda x, y: -np.arctan(y / (1 - x)),
        lambda x, y: np.pi - np.arctan((1 - y) / x),
        lambda x, y: np.pi + np.arctan(y / x),
    ]
    rho_lims = [
        lambda x, y, t: (1 - y - x) / (np.cos(t) + np.sin(t)),
        lambda x, y, t: -x / np.cos(t),
        lambda x, y, t: -y / np.sin(t)
    ]

    pts = []
    wts = []
    for I in range(3):
        for (obsxhat, obsyhat), wouter in zip(*q_outer):
            T1 = theta_lims[I](obsxhat, obsyhat)
            T2 = theta_lims[(I + 1) % 3](obsxhat, obsyhat)
            if T2 < T1:
                T2 += 2 * np.pi
            for theta, wtheta in zip(*quad.map_to(q_theta, [T1, T2])):
                Rmax = rho_lims[I](obsxhat, obsyhat, theta)
                for rho, wrho in zip(*quad.map_to(q_rho, [0, Rmax])):
                    w = wouter * wrho * wtheta * rho
                    srcxhat = obsxhat + rho * np.cos(theta)
                    srcyhat = obsyhat + rho * np.sin(theta)
                    pts.append((obsxhat, obsyhat, srcxhat, srcyhat))
                    wts.append(w)

    return np.array(pts), np.array(wts)


def edge_adj_quad(n_theta, n_rho, n_outer):
    q_theta = quad.gaussxw(n_theta)
    q_rho = quad.gaussxw(n_rho)
    q_outer = quad.gauss2d_tri(n_outer)

    theta_lims = [
        lambda x: 0,
        lambda x: np.pi - np.arctan(1 / (1 - x)),
        lambda x: np.pi
    ]
    rho_lims = [
        lambda x, t: x / (np.cos(t) + np.sin(t)),
        lambda x, t: -(1 - x) / np.cos(t)
    ]

    pts = []
    wts = []
    for I in range(2):
        for (obsxhat, obsyhat), wouter in zip(*q_outer):
            T1 = theta_lims[I](obsxhat)
            T2 = theta_lims[I + 1](obsxhat)
            for theta, wtheta in zip(*quad.map_to(q_theta, [T1, T2])):
                Rmax = rho_lims[I](obsxhat, theta)
                for rho, wrho in zip(*quad.map_to(q_rho, [0, Rmax])):
                    w = wouter * wrho * wtheta * rho

                    srcxhat = rho * np.cos(theta) + (1 - obsxhat)
                    srcyhat = rho * np.sin(theta)

                    pts.append((obsxhat, obsyhat, srcxhat, srcyhat))
                    wts.append(w)

    return np.array(pts), np.array(wts)
