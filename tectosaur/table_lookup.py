import time
import numpy as np
from tectosaur.interpolate import barycentric_evalnd, cheb, cheb_wts, from_interval
from tectosaur.standardize import standardize
from tectosaur.geometry import tri_normal, remove_proj, vec_angle

minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01
filename, n_A, n_B, n_pr = ('coincidenttable_limits.npy', 8, 8, 8)
filename, n_A, n_B, n_pr = ('testtable.npy', 3, 3, 3)

def coincident_interp_pts_wts(n_A, n_B, n_pr):
    Ahats = cheb(-1, 1, n_A)
    Bhats = cheb(-1, 1, n_B)
    prhats = cheb(-1, 1, n_pr)
    Ah,Bh,Nh = np.meshgrid(Ahats, Bhats,prhats)
    interp_pts = np.array([Ah.ravel(),Bh.ravel(), Nh.ravel()]).T

    Awts = cheb_wts(-1,1,n_A)
    Bwts = cheb_wts(-1,1,n_B)
    prwts = cheb_wts(-1,1,n_pr)
    interp_wts = np.outer(Awts,np.outer(Bwts,prwts)).ravel()
    return interp_pts, interp_wts

def coincident_lookup(table_and_pts_wts, sm, pr, tri):

    standard_tri, labels, translation, R, scale = standardize(tri)

    A, B = standard_tri[2][0:2]

    Ahat = from_interval(minlegalA, 0.5, A)
    Bhat = from_interval(minlegalB, maxlegalB, B)
    prhat = from_interval(0.0, 0.5, pr)

    interp_vals = np.empty(81)
    for j in range(81):
        interp_vals[j] = barycentric_evalnd(
            interp_pts, interp_wts, table[:, j], np.array([[Ahat, Bhat, prhat]])
        )[0]
    interp_vals = interp_vals.reshape((3,3,3,3))

    out = np.empty((3,3,3,3))
    for sb1 in range(3):
        for sb2 in range(3):
            cb1 = labels[sb1]
            cb2 = labels[sb2]
            correct = R.T.dot(interp_vals[sb1,:,sb2,:]).dot(R) / (sm * scale ** 1)
            out[cb1,:,cb2,:] = correct
    return out

def coincident_table(kernel, sm, pr, pts, tris):
    interp_pts, interp_wts = coincident_interp_pts(n_A, n_B, n_pr)

    tri_pts = pts[tris]

    table = np.load(kernel + filename)
    out = np.empty((tris.shape[0], 3, 3, 3, 3))

    for i in range(tris.shape[0]):
        out[i,:,:,:,:] = coincident_lookup(
            (table, interp_pts, interp_wts), sm, pr, tri_pts[i,:,:]
        )

    return out

def get_adjacent_theta(obs_tri, src_tri):
    p = obs_tri[1] - obs_tri[0]
    L1 = obs_tri[2] - obs_tri[0]
    L2 = src_tri[2] - src_tri[0]
    T1 = remove_proj(L1, p)
    T2 = remove_proj(L2, p)

    n1 = tri_normal(obs_tri, normalize = True)
    samedir = n1.dot(T2 - T1) > 0

    theta = vec_angle(T1, T2)

    if samedir:
        return theta
    else:
        return 2 * np.pi - theta

def adjacent_interp_pts_wts(n_theta, n_pr):
    thetahats = cheb(-1, 1, n_theta)
    prhats = cheb(-1, 1, n_pr)
    Th,Nh = np.meshgrid(thetahats,prhats)
    interp_pts = np.array([Th.ravel(), Nh.ravel()]).T

    thetawts = cheb_wts(-1, 1, n_theta)
    prwts = cheb_wts(-1, 1, n_pr)
    interp_wts = np.outer(thetawts, prwts).ravel()
    return interp_pts, interp_wts

min_angle = 20
rho = 0.5 * np.tan(np.deg2rad(min_angle))
def separate_tris(obs_tri, src_tri):
    A,B = obs_tri[2][:2]
    obs_split_pt = [0.5, rho, 0.0]
    obs_tris = (
        [obs_tri[0], obs_tri[1], obs_split_pt]
        [obs_tri[1], obs_tri[2], obs_split_pt]
        [obs_tri[2], obs_tri[0], obs_split_pt]
    )

    src_split_pt = [0.5, rho * V[0], rho * V[1]]
    src_tris = (
        [src_tri[0], src_tri[1], src_split_pt]
        [src_tri[1], src_tri[2], src_split_pt]
        [src_tri[2], src_tri[0], src_split_pt]
    )
    return obs_tris, src_tris

def adjacent_table(kernel, sm, pr, pts, obs_tris, src_tris):
    n_theta = n_pr = 3
    interp_pts, interp_wts = adjacent_interp_pts_wts(n_theta, n_pr)

    tri_pts = pts[tris]

    for i in range(obs_tris.shape[0]):
        theta = get_adjacent_theta(tri_pts[obs_tris[i]], tri_pts[src_tris[i]])
        thetahat = from_interval(0, np.pi, theta)
        prhat = from_interval(0, 0.5, pr)
        print(theta, thetahat, prhat)
