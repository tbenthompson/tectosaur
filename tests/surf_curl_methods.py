import numpy as np

basis_gradient = [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]
e = [[[int((i - j) * (j - k) * (k - i) / 2) for k in range(3)]
    for j in range(3)] for i in range(3)]

tri = np.random.rand(3,3)
# tri = np.array([[0,0,0],[1.1,0,0],[0,1.1,0]])
# tri = np.array([[0,0,0],[1,1,0],[0,1,0]])

surf_curl = np.empty((3,3))
g1 = tri[1] - tri[0]
g2 = tri[2] - tri[0]
unscaled_normal = np.cross(g1, g2)
jacobian_mag = np.linalg.norm(unscaled_normal)
normal = unscaled_normal / jacobian_mag
for basis_idx in range(3):
    for s in range(3):
        surf_curl[basis_idx][s] = (
            + basis_gradient[basis_idx][0] * g2[s]
            - basis_gradient[basis_idx][1] * g1[s]
        ) / jacobian_mag;

print(tri, jacobian_mag, normal)
print(basis_gradient)


jacobian = np.array([
    g1, g2, unscaled_normal
]).T
inv_jacobian = np.linalg.inv(jacobian)

real_basis_gradient = np.zeros((3,3))
for basis_idx in range(3):
    for j in range(3):
        real_basis_gradient[basis_idx][j] = sum(
            [basis_gradient[basis_idx][d] * inv_jacobian[d][j] for d in range(2)]
        )

surf_curl2 = np.zeros((3,3))
for basis_idx in range(3):
    for s in range(3):
        for b in range(3):
            for c in range(3):
                surf_curl2[basis_idx][s] += e[b][c][s] * normal[b] * real_basis_gradient[basis_idx][c]

print(surf_curl)
print(surf_curl2)
np.testing.assert_almost_equal(surf_curl, surf_curl2)
