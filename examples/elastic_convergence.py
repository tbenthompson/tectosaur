import numpy as np
import scipy.sparse
import tectosaur.mesh as mesh
from tectosaur.constraints import constraints, build_constraint_matrix
from tectosaur.dense_integral_op import DenseIntegralOp

sm = 1.0
pr = 0.25
w = 4
corners = [[w, w, 0], [w, -w, 0], [-w, -w, 0], [-w, w, 0]]
m = mesh.make_rect(4, 4, corners)
cs = constraints(m[1], np.empty((0,3)), m[0])
cm = build_constraint_matrix(cs, m[1].shape[0] * 9)
cm = cm[0].todense()

old_iop = None
for i, nq in enumerate(range(2, 20, 1)):
    eps = (2.0 ** -np.arange(0, nq)) / (6.7)
    # eps = np.linspace(1.1, 0.9, nq)
    # eps = [1.0, 0.3]
    # eps = np.linspace((1 + nq) / 10.0, 0.1, nq)
    # if i == 0:
    #     eps = [0.2, 0.1]
    # elif i == 1:
    #     eps = [0.2, 0.1, 0.05]
    # elif i == 2:
    #     eps = [0.2, 0.1, 0.05, 0.025]
    # elif i == 3:
    #     eps = [0.2, 0.1, 0.05, 0.025, 0.0125]

    print(eps)
    iop = DenseIntegralOp(
        eps,
        (17, 17, 17, 15),
        (23, 13, 9, 15),
        7, 3, 6, 4.0, 'U', sm, pr, m[0], m[1]
    )

    if i == 0:
        old_iop = iop
        continue
    Khat = cm.T.dot(iop.mat.dot(cm))
    Khat_old = cm.T.dot(old_iop.mat.dot(cm))
    diff = (Khat - Khat_old)[27:33,27:33]
    # import matplotlib.pyplot as plt
    # plt.imshow(Khat - Khat_old, interpolation = 'none')
    # plt.colorbar()
    # plt.show()
    print(i, nq)
    print(np.max(diff))
    print(np.max(Khat[27:33,27:33]))
    old_iop = iop






        # (12, 11, 12, 8),
        # (7, 7, 8, 12),
        # 6, 3, 6, 4.0, 'U', sm, pr, m[0], m[1]



        # (17, 17, 17, 14),
        # (23, 13, 9, 15),
        # nq, 3, 6, 4.0, 'H', sm, pr, m[0], m[1]
