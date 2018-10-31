import attr
import numpy as np
from multiprocessing import Pool
import cloudpickle

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

def caller(data):
    f = cloudpickle.loads(data)
    return f()

def build_c2e(tree, check_r, equiv_r, cfg):
    t = Timer()

    assembler = FarfieldTriMatrix(cfg.K.name, cfg.params, 4, np.float64)
    check_surf = inscribe_surf(Ball(center = (0,0,0), R = 1), check_r, cfg.surf)
    equiv_surf = inscribe_surf(Ball(center = (0,0,0), R = 1), equiv_r, cfg.surf)

    new_pts, new_tris = concat(check_surf, equiv_surf)
    n_check_tris = check_surf[1].shape[0]
    check_tris = new_tris[:n_check_tris]
    equiv_tris = new_tris[n_check_tris:]

    mat = assembler.assemble(new_pts, check_tris, equiv_tris)

    nrows = mat.shape[0] * 9
    ncols = mat.shape[3] * 9
    equiv_to_check = mat.reshape((nrows, ncols))
    t.report('build e2cs')

    U, eig, VT = np.linalg.svd(equiv_to_check)
    out = (U.T.copy(), eig.copy(), VT.T.copy())
    t.report('svd')
    return out
