import numpy as np
from tectosaur.ops.sparse_farfield_op import (
    TriToTriDirectFarfieldOp, PtToPtFMMFarfieldOp, PtToPtDirectFarfieldOp)
import tectosaur.mesh.mesh_gen as mesh_gen
from test_farfield import make_meshes

# TODO: dim
def test_tri_ones():
    m, surf1_idxs, surf2_idxs = make_meshes(n_m = 2, sep = 5, w = 1)
    ops = [
        C(
            3, 'tensor_one3', [], m[0], m[1], np.float64,
            surf1_idxs, surf2_idxs
        ) for C in [
            TriToTriDirectFarfieldOp,
            PtToPtDirectFarfieldOp,
            PtToPtFMMFarfieldOp(250, 3.0, 250)
        ]
    ]

    x = np.random.rand(ops[0].shape[1])
    x = np.ones(ops[0].shape[1])
    ys = [op.dot(x) for op in ops]
    for y in ys:
        print(y)
