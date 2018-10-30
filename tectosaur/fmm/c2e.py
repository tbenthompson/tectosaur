import attr
import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.mesh.modify import concat
from tectosaur.ops.dense_integral_op import FarfieldTriMatrix
from tectosaur.constraint_builders import continuity_constraints
from tectosaur.constraints import build_constraint_matrix
from tectosaur.ops.dense_integral_op import RegularizedDenseIntegralOp
from tectosaur.util.timer import Timer

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

def c2e_solve(surf, bounds, check_r, equiv_r, alpha):
    check_surf = inscribe_surf(bounds, check_r, surf)
    e2c = RegularizedDenseIntegralOp(
        10,10,10,10,10,10000.0,'elasticRH3','elasticRH3',[1.0,0.25],
        check_surf[0], check_surf[1], np.float64,
    ).mat
    c2e = reg_lstsq_inverse(e2c, alpha)
    return c2e

def build_c2e(tree, check_r, equiv_r, cfg):
    def make(bounds):
        return c2e_solve(
            cfg.surf, bounds, check_r, equiv_r, cfg.alpha
        )
    return make(Ball(center = (0.0,0.0,0.0), R = 1.0))
