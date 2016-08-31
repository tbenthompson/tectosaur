import pickle
import numpy as np
import scipy.spatial

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import okada_wrapper

import tectosaur.mesh as mesh
from tectosaur.mass_op import MassOp
from tectosaur.sparse_integral_op import SparseIntegralOp, FMMIntegralOp, interp_galerkin_mat
from tectosaur.dense_integral_op import DenseIntegralOp
from tectosaur.quadrature import gauss2d_tri
import tectosaur.constraints as constraints
import tectosaur.geometry as geometry

from solve import iterative_solve, direct_solve, SumOp
from tectosaur.util.timer import Timer

def spherify(center, r, pts):
    D = scipy.spatial.distance.cdist(pts, center.reshape((1,center.shape[0])))
    return (r / D) * (pts - center) + center

def make_sphere(center, r, refinements):
    pts = np.array([[0,-r,0],[r,0,0],[0,0,r],[-r,0,0],[0,0,-r],[0,r,0]])
    pts += center
    tris = np.array([[1,0,2],[2,0,3],[3,0,4],[4,0,1],[5,1,2],[5,2,3],[5,3,4],[5,4,1]])
    m = pts, tris
    for i in range(refinements):
        m = mesh.refine(m)
    spherified_m = [spherify(center, r, m[0]), m[1]]
    return spherified_m

def plot_sphere_3d(pts, tris):
    fig = plt.figure()
    ax = Axes3D(fig)
    verts = pts[tris]
    coll = Poly3DCollection(verts)
    coll.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.add_collection3d(coll)
    plt.show()

def main():
    refine = 3
    m = make_sphere(np.array([0,0,0]), 1.0, refine)
    tri_pts = m[0][m[1]]
    # plot_sphere_3d(*m)

    sm = 1.0
    pr = 0.25
    cs = constraints.constraints(m[1], np.array([]), m[0])

    selfop = MassOp(3, m[0], m[1]).mat

    # igmat = interp_galerkin_mat(tri_pts, gauss2d_tri(3))
    # selfop = igmat[0].T.dot(igmat[0])
    # import ipdb; ipdb.set_trace()

    eps = [0.08, 0.04, 0.02, 0.01]
    t = Timer()
    Uop = DenseIntegralOp(
        eps, (20,20,20,20), (25,19,19,19), 9, 4, 9, 4.0,
        'U', sm, pr, m[0], m[1]
    )
    t.report('U')
    Top = DenseIntegralOp(
        eps, (20,20,20,20), (25,19,19,19), 9, 4, 9, 4.0,
        'T', sm, pr, m[0], m[1]
    )
    t.report('T')



    # solving: u(x) + int(T*u) = int(U*t)
    # traction values radial direction because the sphere is centered at (0,0,0)

    unscaled_ns = geometry.unscaled_normals(tri_pts)
    unscaled_ns /= geometry.jacobians(unscaled_ns)[:,np.newaxis]
    input_nd = np.tile(unscaled_ns[:,np.newaxis,:], (1, 3, 1))
    # input_nd = tri_pts / np.linalg.norm(tri_pts, axis = 2)[:,:,np.newaxis]
    input = input_nd.reshape(tri_pts.shape[0] * 9)
    avg_face_input = np.mean(input_nd, axis = 1)
    input_mag = np.sqrt(np.sum(avg_face_input ** 2, axis = 1))

    solve_for = 'disp'
    if solve_for == 'disp':
        lhs = Top.mat + selfop
        rhs = Uop.dot(input)

        cm, c_rhs = constraints.build_constraint_matrix(cs, lhs.shape[0])
        cm = cm.tocsr().todense()
        cmT = cm.T
        lhs_constrained = cmT.dot(lhs.dot(cm))
        rhs_constrained = cmT.dot((rhs + lhs.dot(c_rhs)).T)
        constrained_soln = np.linalg.solve(lhs_constrained, rhs_constrained)
        soln = cm.dot(constrained_soln)

        disp = np.array(soln).reshape((int(lhs.shape[0] / 9), 3, 3))
        avg_face_disp = np.mean(disp, axis = 1)
        disp_mag = np.sqrt(np.sum(avg_face_disp ** 2, axis = 1))
        to_plot = disp_mag
    elif solve_for == 'trac':
        lhs = Uop.mat
        rhs = (-Top.mat - selfop).dot(input)
        soln = np.linalg.solve(lhs, rhs.T)
        trac = np.array(soln).reshape((int(lhs.shape[0] / 9), 3, 3))
        avg_face_trac = np.mean(trac, axis = 1)
        trac_mag = np.sqrt(np.sum(avg_face_trac ** 2, axis = 1))
        to_plot = trac_mag


    for var in [to_plot, input_mag]:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        cmap = plt.get_cmap('Blues')
        triang = tri.Triangulation(m[0][:,0], m[0][:,1], m[1])
        collec = ax.plot_trisurf(triang, m[0][:,2], cmap=cmap, shade=False, linewidth=0.)
        collec.set_array(var)
        collec.autoscale()
        plt.colorbar(collec)
    plt.show()








if __name__ == '__main__':
    main()
