import attr
import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.mesh.modify import concat
from tectosaur.ops.dense_integral_op import FarfieldTriMatrix
from tectosaur.constraint_builders import continuity_constraints
from tectosaur.constraints import build_constraint_matrix
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

def c2e_solve(assembler, surf, bounds, check_r, equiv_r, alpha):
    t = Timer()
    check_surf = inscribe_surf(bounds, check_r, surf)
    equiv_surf = inscribe_surf(bounds, equiv_r, surf)

    new_pts, new_tris = concat(check_surf, equiv_surf)
    n_check_tris = check_surf[1].shape[0]
    check_tris = new_tris[:n_check_tris]
    equiv_tris = new_tris[n_check_tris:]

    print("GO")
    mat = assembler.assemble(new_pts, check_tris, equiv_tris)
    t.report('assemble')

    nrows = mat.shape[0] * 9
    ncols = mat.shape[3] * 9
    equiv_to_check = mat.reshape((nrows, ncols))
    t.report('reshape')

    c2e = reg_lstsq_inverse(equiv_to_check, alpha)
    t.report('invert')
    return c2e

def build_c2e(tree, check_r, equiv_r, cfg):
    def make(assembler, bounds):
        return c2e_solve(
            assembler, cfg.surf, bounds, check_r, equiv_r, cfg.alpha
        )

    assembler = FarfieldTriMatrix(cfg.K.name, [1.0, 0.25], 2, np.float32)
    c2e_ops = []
    for n in tree.nodes:
        c2e_ops.append(make(assembler, n.bounds))
    return c2e_ops
