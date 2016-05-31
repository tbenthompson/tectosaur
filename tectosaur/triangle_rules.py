import tectosaur.quadrature as quad
import numpy as np

def coincident_quad(eps, n_outer_sing, n_outer_smooth, n_theta, n_rho):
    rho_quad = quad.sinh_transform(quad.gaussxw(n_rho), -1, eps)
    theta_quad = quad.gaussxw(n_theta)
    outer_smooth_quad = quad.aimi_diligenti(quad.gaussxw(n_outer_smooth), 3, 3)
    outer_sing_quad1 = quad.sinh_transform(quad.gaussxw(n_outer_sing), 1, eps)
    outer_sing_quad23 = quad.sinh_transform(quad.gaussxw(n_outer_sing), -1, eps)

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

    def obsyhat_integral(desc, ox, wx):
        q = quad.map_to(desc[4], [0, 1 - ox])
        for oy, wy in zip(*q):
            inner_integral(ox, oy, wx * wy, desc)

    def obsxhat_integral(desc):
        q = quad.map_to(desc[3], [0, 1])
        for ox, wx in zip(*q):
            obsyhat_integral(desc, ox, wx)

    def I1():
        desc = (
            rho_lims[0],
            lambda x, y: theta_lims[2](x, y) - 2 * np.pi,
            theta_lims[0]
        )
        q_beta = quad.map_to(outer_smooth_quad, [0, np.pi / 2])
        for b, wb in zip(*q_beta):
            alpha_max = 1.0 / (np.cos(b) + np.sin(b))
            q_alpha = quad.map_to(outer_sing_quad1, [0, alpha_max])
            for a, wa in zip(*q_alpha):
                ox = a * np.cos(b)
                oy = a * np.sin(b)
                inner_integral(ox, oy, wa * wb * a, desc)

    def I2():
        obsxhat_integral((
            rho_lims[1],
            theta_lims[0],
            theta_lims[1],
            outer_sing_quad23,
            outer_smooth_quad,
        ))

    def I3():
        obsxhat_integral((
            rho_lims[2],
            theta_lims[1],
            theta_lims[2],
            outer_smooth_quad,
            outer_sing_quad23,
        ))

    I1()
    I2()
    I3()

    return np.array(pts), np.array(wts)

def edge_adj_quad(eps, n_x, n_theta, n_beta, n_alpha, basis_cancellation):
    # alpha and beta here are capital lambda and captial psi in
    # sutradhar et al
    if basis_cancellation:
        alpha_quad = quad.gaussxw(n_alpha)
    else:
        alpha_quad = quad.sinh_transform(quad.gaussxw(n_alpha), -1, eps)
    beta_quad = quad.gaussxw(n_beta)
    theta_quad = quad.gaussxw(n_theta)
    x_quad = quad.aimi_diligenti(quad.gaussxw(n_x), 2, 2)

    pts = []
    wts = []

    def alpha_integral(w, obsxhat, theta, beta, alpha_max):
        q = quad.map_to(alpha_quad, [0, alpha_max])
        for alpha, alphaw in zip(*q):
            jacobian = alpha ** 2 * np.cos(beta)

            rho = alpha * np.cos(beta)
            obsyhat = alpha * np.sin(beta)

            srcxhat = rho * np.cos(theta) + (1 - obsxhat)
            srcyhat = rho * np.sin(theta)

            pts.append((obsxhat, obsyhat, srcxhat, srcyhat))
            wts.append(jacobian * alphaw * w)


    def beta_integral(w, x, theta, beta_min, beta_max, alpha_max_fnc):
        q = quad.map_to(beta_quad, [beta_min, beta_max])
        for beta, betaw in zip(*q):
            alpha_integral(betaw * w, x, theta, beta, alpha_max_fnc(beta))

    def theta_integral(w, x, theta_min, theta_max, L_fnc):
        q = quad.map_to(theta_quad, [theta_min, theta_max])
        for theta, thetaw in zip(*q):
            L_theta = L_fnc(theta)
            beta1 = np.arctan((1 - x) / L_theta)
            alpha_max_f1 = lambda b: L_theta / np.cos(b)
            alpha_max_f2 = lambda b: (1 - x) / np.sin(b)
            beta_integral(w * thetaw, x, theta, 0, beta1, alpha_max_f1)
            beta_integral(w * thetaw, x, theta, beta1, np.pi / 2, alpha_max_f2)

    def x_integral():
        q = quad.map_to(x_quad, [0, 1])
        for x, xw in zip(*q):
            theta1 = np.pi - np.arctan(1 / (1 - x))
            L_fnc1 = lambda t: x / (np.cos(t) + np.sin(t))
            L_fnc2 = lambda t: -(1 - x) / np.cos(t)
            theta_integral(xw, x, 0, theta1, L_fnc1)
            theta_integral(xw, x, theta1, np.pi, L_fnc2)

    x_integral()
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

