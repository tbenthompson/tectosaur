import numpy as np
import scipy.sparse
import tectosaur.mesh as mesh
from tectosaur.constraints import constraints, build_constraint_matrix
from tectosaur.dense_integral_op import DenseIntegralOperator

sm = 1.0
pr = 0.25
w = 4
corners = [[w, w, 0], [w, -w, 0], [-w, -w, 0], [-w, w, 0]]
m = mesh.rect_surface(10, 10, corners)
cs = constraints(m[1], np.empty((0,3)), m[0])
cm = build_constraint_matrix(cs, m[1].shape[0] * 9)
cm = cm.todense()

old_iop = None
for i, nq in enumerate(range(1, 6)):
    iop = DenseIntegralOperator(15, 15, 8, nq, sm, pr, m[0], m[1])
    print(iop.shape)
    if i == 0:
        old_iop = iop
        continue
    Khat = cm.T.dot(iop.mat.dot(cm))
    print(Khat.shape)
    Khat_old = cm.T.dot(old_iop.mat.dot(cm))
    print(np.max(Khat_old - Khat))
    print(np.max(Khat_old))
    old_iop = iop
