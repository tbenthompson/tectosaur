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
