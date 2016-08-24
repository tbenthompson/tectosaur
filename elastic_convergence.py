import numpy as np
import scipy.sparse
import tectosaur.mesh as mesh
from tectosaur.constraints import constraints, build_constraint_matrix
from tectosaur.dense_integral_op import DenseIntegralOp

sm = 1.0
pr = 0.25
w = 4
corners = [[w, w, 0], [w, -w, 0], [-w, -w, 0], [-w, w, 0]]
m = mesh.rect_surface(4, 4, corners)
cs = constraints(m[1], np.empty((0,3)), m[0])
cm = build_constraint_matrix(cs, m[1].shape[0] * 9)
cm = cm[0].todense()

old_iop = None
for i, nq in enumerate(range(5, 45, 3)):
    iop = DenseIntegralOp(
        [0.08, 0.04, 0.02, 0.01],
        (17, 17, 17, 14),
        (23, 13, 9, 15),
        # [0.08, 0.04, 0.02, 0.01, 0.005],
        # (23, 23, 23, nq),
        # (29, 19, 15, 21),
        8, 4, sm, pr, m[0], m[1]
    )

    # (15,15,7,15)
    if i == 0:
        old_iop = iop
        continue
    Khat = cm.T.dot(iop.mat.dot(cm))
    Khat_old = cm.T.dot(old_iop.mat.dot(cm))
    print(i, nq)
    print(np.max(Khat_old - Khat))
    print(np.max(Khat_old))
    old_iop = iop
