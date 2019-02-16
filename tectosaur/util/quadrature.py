import numpy as np
import scipy.linalg
from tectosaur.util.tri_gauss import get_tri_gauss

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

def gauss2d_rect(N):
    qg = gaussxw(N)

    q_rect_x, q_rect_y = np.meshgrid(qg[0], qg[0])
    q_rect_pts = np.array([q_rect_x.flatten(), q_rect_y.flatten()]).T
    q_rect_w = np.outer(qg[1], qg[1]).flatten()
    return q_rect_pts, q_rect_w

def gauss2d_tri(N):
    pts, wts = get_tri_gauss(N * 2 - 1)
    if pts is not None:
        return np.array(pts), np.array(wts) * 0.5

    q_rect_pts, q_rect_w = gauss2d_rect(N)
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

def tanh_sinh(N):
    import mpmath
    ts = mpmath.calculus.quadrature.TanhSinh(mpmath.mp)
    pts_wts = ts.calc_nodes(N, 40)
    np_pts_wts = np.array(pts_wts).astype(np.float)
    pts = np_pts_wts[:,0].copy()
    wts = np_pts_wts[:,1].copy() / (2 ** (N - 1))
    return pts, wts

def telles_singular(N, x0):
    """
    Use a cubic polynomial transformation to turn a 1D log singular
    integral into an integrable form.
    This should also be able to accurately integrate terms like
    log(|r|) where r = (x - y).
    This should also be able to accurately integrate terms like 1/r if
    the singularity is outside, but near the domain of integration.
    See
    "A SELF-ADAPTIVE CO-ORDINATE TRANSFORMATION FOR EFFICIENT NUMERICAL
    EVALUATION OF GENERAL BOUNDARY ELEMENT INTEGRALS", Telles, 1987.
    for a description of the method. I use the same notation adopted in that
    paper. The interval of integration is [-1, 1]
    Note there is a printing error in the Jacobian of the transformation
    from gamma coordinates to eta coordinates in the Telles paper. The
    formula in the paper is
    (3 * (gamma - gamma_bar ** 2)) / (1 + 3 * gamma_bar ** 2)
    It SHOULD read:
    (3 * (gamma - gamma_bar) ** 2) / (1 + 3 * gamma_bar ** 2)
    """
    eta_bar = x0
    eta_star = eta_bar ** 2 - 1.0

    # The location of the singularity in gamma space
    term1 = (eta_bar * eta_star + np.abs(eta_star))
    term2 = (eta_bar * eta_star - np.abs(eta_star))

    # Fractional powers of negative numbers are multiply valued and python
    # recognizes this. So, I specify that I want the real valued third root
    gamma_bar = np.sign(term1) * np.abs(term1) ** (1.0 / 3.0) + \
            np.sign(term2) * np.abs(term2) ** (1.0 / 3.0) + \
            eta_bar

    gamma, gamma_weights = gaussxw(N)
    x = ((gamma - gamma_bar) ** 3 + gamma_bar * (gamma_bar ** 2 + 3))\
            / (1 + 3 * gamma_bar ** 2)

    w = gamma_weights * (3 * (gamma - gamma_bar) ** 2) \
            / (1 + 3 * gamma_bar ** 2)

    # If I accidentally choose a Gaussian integration scheme that
    # exactly samples the singularity, this method will fail. This can
    # be easily remedied by simply increasing the order of the method.
    # For example, this happens if x0 == 0 and N is odd. Just use an even
    # order in that case.
    if (np.abs(x - x0) < 1e-12).any():
        raise Exception("Telles integration has sampled the " +
                "singularity. Choose a different order of integration.")

    return x, w
