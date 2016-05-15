import numpy as np
from math import pi
import sympy as sp
import dill

# I was unable to find a good source on the exact form of the
# fundamental solutions for the elastic boundary integral equations
# Some are slightly wrong, and most others simply give the stress forms
# for the hypersingular equation.
#
# Furthermore, getting the kernels is super annoying because then I don't
# know where to look for bugs. So, I compare hand-derived version (few terms)
# implemented in the c++ with a symbolically derived version computed here.
# I could use the symbolically derived versions directly, but they are much
# much larger

x1, x2, y1, y2, z1, z2 = sp.symbols('x1, x2, y1, y2, z1, z2')
sm, pr = sp.symbols('sm, pr')
nx, ny, nz = sp.symbols('nx, ny, nz')
mx, my, mz = sp.symbols('mx, my, mz')
sm_val = 30e9
pr_val = 0.25
rho = 3000
g = 9.8
b0_val = 0.0
b1_val = rho * g
b0, b1 = sp.symbols('b0, b1')
b = [b0, b1]

kronecker = np.identity(3)

def disp_creator(k, j, dimension):
    if dimension == 2:
        return disp_creator_2d(k, j)
    elif dimension == 3:
        return disp_creator_3d(k, j)

def disp_creator_2d(k, j):
    C1 = 1 / (8 * sp.pi * sm * (1 - pr))
    C2 = (3 - 4 * pr)
    delta = [x1 - x2, y1 - y2]
    r2 = sum([d ** 2 for d in delta])
    disp = C1 * (-C2 * kronecker[k,j] * sp.log(sp.sqrt(r2)) +
                 (delta[k] * delta[j] / r2))
    return disp

def disp_creator_3d(k, j):
    C1 = 1 / (16 * sp.pi * sm * (1 - pr))
    C2 = (3 - 4 * pr)
    delta = [x1 - x2, y1 - y2, z1 - z2]
    r2 = sum([d**2 for d in delta])

    disp = (C1 / sp.sqrt(r2)) *\
           (C2 * kronecker[k,j] + delta[k] * delta[j] / r2)
    return disp

def get_dimension_indices(dim):
    if dim == 2:
        return [0,1]
    elif dim == 3:
        return [0,1,2]

def stress_from_strain(strain, strain_trace, shear_mod, lame_lambda, l, m):
    return lame_lambda * strain_trace * kronecker[l,m] +\
           2 * shear_mod * strain[l][m]

# Symbolically finds the tractions corresponding to a displacement field.
# t_i = c_ijkl ((u_k,l + u_l,k) / 2) * n_j
def traction_operator(disp_vec, pos_vec, index, normal, dimension):
    dim_indices = get_dimension_indices(dimension)
    disp_grad = [[sp.diff(disp_vec[l], pos_vec[m])
                    for m in dim_indices] for l in dim_indices]
    strain = [[(disp_grad[l][m] + disp_grad[m][l]) / 2
                    for m in dim_indices] for l in dim_indices]
    strain_trace = sum([strain[d][d] for d in dim_indices])

    lame_lambda = (2 * sm * pr) / (1 - 2 * pr)
    stress = [[stress_from_strain(strain, strain_trace, sm, lame_lambda, l, m)
                for m in dim_indices] for l in dim_indices]
    trac = sum([stress[index][d] * normal[d] for d in dim_indices])
    return trac

def galerkin_tensor_2d(k, j):
    C1 = (-1.0 / (8 * sp.pi * sm))
    delta = [x1 - x2, y1 - y2]
    r2 = sum([d**2 for d in delta])
    r = sp.sqrt(r2)
    return C1 * kronecker[k, j] * r2 * sp.log(r)

def gravity_displacement(k, j, dimension):
    galerkin_vec = [
        galerkin_tensor_2d(k, d) for d in range(dimension)
    ]
    src_pos_vec = [x1, y1, z1]
    src_n = [nx, ny, nz]
    diffed_galerkin_vec = [[sp.diff(galerkin_vec[d1], src_pos_vec[d2])
        for d2 in range(dimension)] for d1 in range(dimension)]
    term1 = sum([diffed_galerkin_vec[j][d] * src_n[d] for d in range(dimension)])
    term2 = sum([diffed_galerkin_vec[d][j] * src_n[d] for d in range(dimension)])
    term2 *= (-1.0 / (2 * (1 - pr)))
    grav_disp = term1 + term2
    return grav_disp

def derive_gravity_kernels(k, j, dimension):
    if j > 0:
        return 0.0, 0.0
    grav_disp = sum([gravity_displacement(k, i, dimension) * b[i]
                     for i in range(dimension)])
    obs_disp_vec = [sum([gravity_displacement(dir, i, dimension) * b[i]
                         for i in range(dimension)]) for dir in range(dimension)]
    obs_pos_vec = [x2, y2, z2]
    obs_n = [mx, my, mz]
    grav_trac = traction_operator(obs_disp_vec, obs_pos_vec, k, obs_n, dimension)
    import ipdb; ipdb.set_trace()
    return grav_disp, grav_trac

# Displacement kernel is given.
# Traction kernel is the traction operator w.r.t. the source coords.
# Adjoint traction kernel is the traction operator w.r.t. the observation coords.
# Hypersingular kernel is the double traction operator w.r.t. both coords.
def derive_kernels(k, j, dimension):
    disp = disp_creator(k, j, dimension)

    dim_indices = get_dimension_indices(dimension)

    src_disp_vec = [disp_creator(k, dir, dimension) for dir in dim_indices]
    src_pos_vec = [x1, y1, z1]
    src_n = [nx, ny, nz]

    obs_disp_vec = [disp_creator(dir, j, dimension) for dir in dim_indices]
    obs_pos_vec = [x2, y2, z2]
    obs_n = [mx, my, mz]

    trac = traction_operator(src_disp_vec, src_pos_vec, j, src_n, dimension)

    adj_trac = traction_operator(obs_disp_vec, obs_pos_vec, k, obs_n, dimension)

    hypersingular_disp_vec = [
        traction_operator(
            [disp_creator(obs_d, src_d, dimension) for src_d in dim_indices],
            src_pos_vec, j, src_n, dimension
        )
        for obs_d in dim_indices
    ]
    hyp = traction_operator(hypersingular_disp_vec, obs_pos_vec, k, obs_n, dimension)

    return disp, trac, adj_trac, hyp

def lambdify_kernels(K):
    args = (sm, pr, b0, b1, x1, y1, z1, x2, y2, z2, nx, ny, nz, mx, my, mz)
    return [[sp.utilities.lambdify(args, K[i][j], "numpy")
        for j in range(3)] for i in range(3)]

def save_kernels(where):
    zeros = lambda: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    U = zeros()
    T = zeros()
    A = zeros()
    H = zeros()
    for i in range(3):
        for j in range(3):
            U[i][j], T[i][j], A[i][j], H[i][j] = derive_kernels(i, j, 3)
    all_kernels = [lambdify_kernels(K) for K in [U, T, A, H]]
    dill.settings['recurse'] = True
    with open(where, 'wb') as f:
        dill.dump(all_kernels, f)

if __name__ == "__main__":
    save_kernels('tests/3d_kernels.pkl')
