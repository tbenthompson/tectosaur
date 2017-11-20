import numpy as np
import matplotlib.pyplot as plt
import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.mesh.modify as mesh_modify
from tectosaur.ops.dense_integral_op import DenseIntegralOp
from tectosaur.ops.mass_op import MassOp

n, w = 10, 10.0
surf = mesh_gen.make_rect(n, n, [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]])
n_fault, L, top_depth = 9, 1.0, -1.0
fault = mesh_gen.make_rect(n_fault, n_fault, [
    [-L, 0, top_depth], [-L, 0, top_depth - 1],
    [L, 0, top_depth - 1], [L, 0, top_depth]
])
all_mesh = mesh_modify.concat(surf, fault)
surface_subset = np.arange(surf[1].shape[0])
fault_subset = np.arange(surf[1].shape[0], all_mesh[1].shape[0])
all_set = np.arange(all_mesh[1].shape[0])

A_set, B_set = all_set, all_set#fault_subset
pairs = [
    ('elasticU3', 'elasticU3', False),
    ('elasticT3', 'elasticA3', True),
    ('elasticH3', 'elasticH3', False)
]
for K1, K2, M in pairs:
    opA = DenseIntegralOp(
        7, 4, 3, 2.0, K1, [1.0, 0.25], all_mesh[0], all_mesh[1], np.float32,
        obs_subset = A_set, src_subset = B_set
    ).mat
    opB = DenseIntegralOp(
        7, 4, 3, 2.0, K2, [1.0, 0.25], all_mesh[0], all_mesh[1], np.float32,
        obs_subset = B_set, src_subset = A_set
    ).mat
    if M:
        mass = MassOp(3, all_mesh[0], all_mesh[1])
        opA += mass.mat
    plt.imshow(np.log10(np.abs(opA - opB.T)))
    plt.colorbar()
    plt.show()
