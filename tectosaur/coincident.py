import tectosaur.quadrature as quad
import numpy as np

def coincident_quad(eps, n_outer_sing, n_outer_smooth, n_theta, n_rho):
    rho_quad = quad.sinh_transform(quad.gaussxw(n_rho), -1, eps)
    # theta_quad = quad.aimi_diligenti(quad.gaussxw(n_theta), 2, 2)
    theta_quad = quad.gaussxw(n_theta)
    outer_smooth_quad = quad.aimi_diligenti(quad.gaussxw(n_outer_smooth), 3, 3)
    # outer_smooth_quad = quad.gaussxw(n_outer_smooth)
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
        # obsxhat_integral((
        #     rho_lims[0],
        #     lambda x, y: theta_lims[2](x, y) - 2 * np.pi,
        #     theta_lims[0],
        #     quad.gaussxw(n_outer_smooth),
        #     quad.gaussxw(n_outer_smooth)
        # ))
        # return
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

def outer_integral():
    pass
