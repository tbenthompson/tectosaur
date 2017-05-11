import numpy as np
import tectosaur.mesh as mesh
from tectosaur.sparse_integral_op import SparseIntegralOp
from tectosaur.nearfield_op import pairs_quad
import tectosaur.quadrature as quad

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

def vert_adj(q, kernel, sm, pr, pts, obs_tris, src_tris):
    return pairs_quad(kernel, sm, pr, pts, obs_tris, src_tris, q, False, False)

pts, va_obs_tris, va_src_tris = np.load('playground/vert_adj_test2.npy')
results = []
# for nq in [(9, 16, 6), (15, 30, 15)]:
for nq in [(4, 4, 4), (15, 30, 15)]:
    q = vertex_adj_quad(*nq)
    va_mat_rot = vert_adj(
        q, 'H', 1.0, 0.25, pts, va_obs_tris[:,:], va_src_tris[:,:]
    )
    results.append(va_mat_rot)
    # print(va_mat_rot[0,0,0,0,0])
results = np.array(results)
worst_idx = np.unravel_index(np.argmax(np.abs(results[-1] - results[-2])), results[0].shape)
print(worst_idx)
seq = results[:,worst_idx[0], worst_idx[1], worst_idx[2], worst_idx[3], worst_idx[4]]
print(seq)
print(np.abs(seq[1:] - seq[:-1]))
# import ipdb; ipdb.set_trace()

#[ 0.18721465  0.18720269]
#[ 0.18728053  0.18728054]
#[ 0.18707482  0.18726057  0.18727893  0.18727988  0.18728031]


# m = mesh.make_sphere([0, 0, 0], 1, 0)
# np.random.seed(113)
# results = []
# v = None
# for nq in [18, 20, 22]:
#     nq = (nq, 2 * nq)
#     Hop = SparseIntegralOp(
#         [0.5], 2, 2, (nq[0], nq[1], nq[0]), 3, 6, 4.0,
#         'H', 1.0, 0.25, m[0], m[1], use_tables = True, remove_sing = False
#     )
#     if v is None:
#         v = np.random.rand(Hop.shape[1])
#     Hdv = Hop.dot(v)
#     print(Hdv[0])
#     results.append(Hdv)
# results = np.array(results)
# print(results[:, 0])
