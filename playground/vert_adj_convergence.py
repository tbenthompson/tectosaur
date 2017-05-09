import numpy as np
import tectosaur.mesh as mesh
from tectosaur.sparse_integral_op import SparseIntegralOp
from tectosaur.nearfield_op import vert_adj

# pts, va_obs_tris, va_src_tris = np.load('playground/vert_adj_test2.npy')
# results = []
# for nq in [5, 15]:
#     va_mat_rot = vert_adj(
#         nq, 'U', 1.0, 0.25, pts, va_obs_tris[:,:], va_src_tris[:,:]
#     )
#     results.append(va_mat_rot)
#     # print(va_mat_rot[0,0,0,0,0])
# results = np.array(results)
# worst_idx = np.unravel_index(np.argmax(np.abs(results[0] - results[1])), results[0].shape)
# import ipdb; ipdb.set_trace()

#[ 0.18721465  0.18720269]
#[ 0.18728053  0.18728054]
#[ 0.18707482  0.18726057  0.18727893  0.18727988  0.18728031]


m = mesh.make_sphere([0, 0, 0], 1, 0)
np.random.seed(113)
results = []
v = None
for nq in [18, 20, 22]:
    nq = (nq, 2 * nq)
    Hop = SparseIntegralOp(
        [0.5], 2, 2, (nq[0], nq[1], nq[0]), 3, 6, 4.0,
        'H', 1.0, 0.25, m[0], m[1], use_tables = True, remove_sing = False
    )
    if v is None:
        v = np.random.rand(Hop.shape[1])
    Hdv = Hop.dot(v)
    print(Hdv[0])
    results.append(Hdv)
results = np.array(results)
print(results[:, 0])
