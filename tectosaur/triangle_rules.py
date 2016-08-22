import tectosaur.quadrature as quad
import numpy as np
import cppimport

# The inner integrals is split into three based on the observation point. Then,
# polar coordinates are introduced to reduce the order of the singularity by one.
# This function simply returns the points and weights of the resulting quadrature
# formula.
def coincident_quad(eps, n_outer, n_theta, n_rho):
    rho_quad = quad.sinh_transform(quad.gaussxw(n_rho), -1, eps)
    theta_quad = quad.poly_transform01(quad.gaussxw(n_theta))
    outer_quad = quad.gauss2d_tri(n_outer)

    theta_lims = [
        lambda x, y: np.pi - np.arctan((1 - y) / x),
        lambda x, y: np.pi + np.arctan(y / x),
        lambda x, y: 2 * np.pi - np.arctan(y / (1 - x))
    ]
    rho_lims = [
        lambda x, y, t: (1 - y - x) / (np.cos(t) + np.sin(t)),
        lambda x, y, t: -x / np.cos(t),
        lambda x, y, t: -y / np.sin(t)
    ]

    pts = []
    wts = []
    def inner_integral(ox, oy, wxy, desc):
        rho_max_fnc = desc[0]
        mapped_theta_quad = quad.map_to(
            theta_quad, [desc[1](ox, oy), desc[2](ox, oy)]
        )
        for i in range(len(mapped_theta_quad[0])):
            t = mapped_theta_quad[0][i]
            wt = mapped_theta_quad[1][i]
            rho_q = quad.map_to(rho_quad, [0, rho_max_fnc(ox, oy, t)])
            rs = rho_q[0]
            wr = rho_q[1]
            xs = ox + rs * np.cos(t)
            ys = oy + rs * np.sin(t)
            jacobian = rs
            pts.extend([(ox, oy, x, y) for x,y in zip(xs, ys)])
            wts.extend(wxy * wr * wt * jacobian)

    def outer_integral(desc):
        for pt, wt in zip(*outer_quad):
            inner_integral(pt[0], pt[1], wt, desc)

    outer_integral((
        rho_lims[0],
        lambda x, y: theta_lims[2](x, y) - 2 * np.pi,
        theta_lims[0]
    ))
    outer_integral((rho_lims[1], theta_lims[0], theta_lims[1]))
    outer_integral((rho_lims[2], theta_lims[1], theta_lims[2]))
    return np.array(pts), np.array(wts)

def edge_adj_quad(eps, n_outer, n_theta, n_rho):
    print(n_rho,eps)
    rho_quad = quad.sinh_transform(quad.gaussxw(n_rho), -1, eps)
    theta_quad = quad.poly_transform01(quad.gaussxw(n_theta))
    outer_quad = quad.gauss2d_tri(n_outer)

    pts = []
    wts = []

    def theta_integral(w, x, y, theta_min, theta_max, L_fnc):
        qt = quad.map_to(theta_quad, [theta_min, theta_max])
        for theta, thetaw in zip(*qt):
            qr = quad.map_to(rho_quad, [0, L_fnc(theta)])
            for rho, rhow in zip(*qr):
                srcxhat = rho * np.cos(theta) + (1 - x)
                srcyhat = rho * np.sin(theta)
                pts.append((x, y, srcxhat, srcyhat))
                wts.append(w * rho * thetaw * rhow)


    for pt, wt in zip(*outer_quad):
        x = pt[0]
        y = pt[1]
        L_fnc1 = lambda t: x / (np.cos(t) + np.sin(t))
        L_fnc2 = lambda t: -(1 - x) / np.cos(t)
        theta1 = np.pi - np.arctan(1 / (1 - x))
        theta_integral(wt, x, y, 0, theta1, L_fnc1)
        theta_integral(wt, x, y, theta1, np.pi, L_fnc2)

    pts = np.array(pts)
    wts = np.array(wts)
    return pts, wts

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

