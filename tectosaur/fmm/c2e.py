import attr
import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.mesh.modify import concat
from tectosaur.ops.dense_integral_op import farfield_tris
from tectosaur.constraint_builders import continuity_constraints
from tectosaur.constraints import build_constraint_matrix

@attr.s
class Ball:
    center = attr.ib()
    R = attr.ib()

def inscribe_surf(ball, scaling, surf):
    new_pts = surf[0] * ball.R * scaling + ball.center
    return (new_pts, surf[1])


# A tikhonov regularization least squares solution via the SVD eigenvalue
# relation.
def reg_lstsq_inverse(M, alpha):
    U, eig, VT = np.linalg.svd(M)
    inv_eig = eig / (eig ** 2 + alpha ** 2)
    return (VT.T * inv_eig).dot(U.T)

def c2e_solve(gpu_module, surf, bounds, check_r, equiv_r, K, params, alpha, float_type):
    equiv_surf = inscribe_surf(bounds, equiv_r, surf)
    check_surf = inscribe_surf(bounds, check_r, surf)

    new_pts, new_tris = concat(check_surf, equiv_surf)
    n_check_tris = check_surf[1].shape[0]
    check_tris = new_tris[:n_check_tris]
    equiv_tris = new_tris[n_check_tris:]


    # morepts = np.load('ptspts.npy')
    # C = np.array([-7./9, 0, -7/9.])
    # R = 0.37514778
    # new_pts2 = (new_pts * R) + C
    # import ipdb
    # ipdb.set_trace()

    mat = farfield_tris(
        K.name, [1.0, 0.25], new_pts, check_tris, equiv_tris, 5, np.float64
    )
    nrows = mat.shape[0] * 9
    ncols = mat.shape[3] * 9
    equiv_to_check = mat.reshape((nrows, ncols))

    continuity = False
    if continuity:
        # assumes check and equiv tris are the same
        cs = continuity_constraints(check_tris, np.array([]))
        cm, c_rhs = build_constraint_matrix(cs, equiv_to_check.shape[1])

        #DO TRANSFORMS IN CONSTRAINED SPACE? OR MAKE THE CONSTRAINT MATRIX ORTHOGONAL?
        e2c_constrained = cm.T.dot((cm.T.dot(equiv_to_check)).T).T

        # import ipdb
        # ipdb.set_trace()
        # uu,ss,vv = np.linalg.svd(e2c_constrained)
        # import matplotlib.pyplot as plt
        # plt.plot(np.log10(ss))
        # plt.show()
        c2e_constrained = reg_lstsq_inverse(e2c_constrained, alpha)
        return cm.dot(cm.dot(c2e_constrained).T).T
    else:
        c2e = reg_lstsq_inverse(equiv_to_check, alpha)
        return c2e

def build_c2e(tree, check_r, equiv_r, cfg):
    def make(R):
        return c2e_solve(
            cfg.gpu_module, cfg.surf,
            Ball([0] * cfg.K.spatial_dim, R), check_r, equiv_r,
            cfg.K, cfg.params, cfg.alpha, cfg.float_type
        )

    n_rows = cfg.surf[1].shape[0] * 9
    levels_to_compute = tree.max_height + 1
    if type(cfg.K.scale_type) is int:
        return make(1.0)

    c2e_ops = np.empty(levels_to_compute * n_rows * n_rows)
    for i in range(levels_to_compute):
        start_idx = i * n_rows * n_rows
        end_idx = (i + 1) * n_rows * n_rows
        c2e_ops[start_idx:end_idx] = make(tree.root().bounds.R / (2.0 ** i))

    return c2e_ops
