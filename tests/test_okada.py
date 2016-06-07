import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import okada_wrapper
from slow_test import slow
from tectosaur.mesh import rect_surface, mesh_concat
from tectosaur.integral_op import self_integral_operator
from tectosaur.adjacency import find_touching_pts
from tectosaur.timer import Timer
import matplotlib.pyplot as plt
import matplotlib.tri as tri

def plot_matrix(v):
    offset = np.min(np.abs(v) > 0)
    plt.imshow(np.log10(np.abs(v + offset)), interpolation = 'none')
    plt.show()

def test_remove_duplicates():
    surface1 = rect_surface(2, 2, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    surface2 = rect_surface(2, 2, [[0, 0, 0], [-1, 0, 0], [-1, 1, 0], [0, 1, 0]])
    m_f = mesh_concat(surface1, surface2)
    assert(m_f[0].shape[0] == 6)
    assert(m_f[1].shape[0] == 4)

def constraints(surface_tris, fault_tris):
    n_surf_dofs = surface_tris.size
    n_fault_dofs = fault_tris.size
    total_dofs_per_dim = n_surf_dofs + n_fault_dofs
    print(total_dofs_per_dim)

    touching_pt = find_touching_pts(surface_tris)
    constraints = []
    for tpt in touching_pt:
        indepedent_dof = tpt[0][0] * 3 + tpt[0][1]
        for dependent in tpt[1:]:
            dependent_dof = dependent[0] * 3 + dependent[1]
            if dependent_dof <= indepedent_dof:
                continue
            for d in range(3):
                dof1 = d * total_dofs_per_dim + indepedent_dof
                dof2 = d * total_dofs_per_dim + dependent_dof
                constraints.append((
                    [(1.0, dof1), (-1.0, dof2)], 0.0
                ))
    assert(len(touching_pt) == surface_tris.size - len(constraints) / 3)

    # X component = 1
    # Y comp = Z comp = 0
    slip = [-1, 0, 0]
    for d in range(3):
        first_dof = total_dofs_per_dim * d + n_surf_dofs
        last_dof = total_dofs_per_dim * (d + 1)
        for i in range(first_dof, last_dof):
            constraints.append(([(1.0, i)], slip[d]))
    constraints = sorted(constraints, key = lambda x: x[0][0][1])
    return constraints

def reshape_iop(iop):
    def swaps(a, swap_lst):
        out = a
        for s in swap_lst:
            out = np.swapaxes(out, s[0], s[1])
        return out

    # Currently o, s, b1, b2, d1, d2
    # Want d1, o, b1, d2, s, b2
    iop = swaps(iop, [(0, 4), (4, 1), (5, 3)])
    n2 = iop.shape[1]
    iop = iop.reshape((n2 * 9, n2 * 9))
    return iop

def insert_constraints(lhs, rhs, cs):
    c_start = lhs.shape[0] - len(cs)
    for i, c in enumerate(cs):
        idx1 = c_start + i
        rhs[idx1] = c[1]
        for dw in c[0]:
            coeff = dw[0]
            idx2 = dw[1]
            lhs[idx1, idx2] = coeff
            lhs[idx2, idx1] = coeff

def solve(iop, cs):
    timer = Timer()
    n_iop = iop.shape[0]
    n = n_iop + len(cs)
    rhs = np.zeros(n)
    lhs_cs = sparse.dok_matrix((n, n))
    insert_constraints(lhs_cs, rhs, cs)
    lhs_cs_sparse = sparse.csr_matrix(lhs_cs)
    timer.report("Build constraints matrix")

    import skcuda.linalg as culg
    culg.init()
    import pycuda.gpuarray as gpuarray
    iop_gpu = gpuarray.to_gpu(iop.astype(np.float32))

    iter = [0]
    def mv(v):
        iter[0] += 1
        print("Iteration: " + str(iter[0]))
        mvtimer = Timer(1)
        out = np.empty(n)
        mvtimer.report("Make vec")
        out = lhs_cs_sparse.dot(v)
        mvtimer.report("Mult constraints")
        v_gpu = gpuarray.to_gpu(v[:n_iop].reshape((n_iop,1)).astype(np.float32))
        iop_times_v_gpu = culg.dot(iop_gpu, v_gpu)
        out[:n_iop] += iop_times_v_gpu.get()[:,0]
        # out[:n_iop] += iop.dot(v[:n_iop])
        mvtimer.report("Mult integral op")
        return out

    def print_resid(r):
        print(r)

    A = sparse.linalg.LinearOperator((n, n), matvec = mv)
    soln = sparse.linalg.gmres(A, rhs, callback = print_resid)
    timer.report("GMRES")
    print(soln)

    # timer = Timer()
    # iop_n = iop.shape[0]
    # lhs = np.zeros((iop_n + len(cs), iop_n + len(cs)))
    # rhs = np.zeros(iop_n + len(cs))
    # lhs[:iop_n,:iop_n] = iop
    # insert_constraints(lhs, rhs, cs)
    # timer.report("Assembly")
    # # plot_matrix(lhs)

    # # soln = np.linalg.solve(lhs, rhs)
    # soln = sp_la.gmres(lhs, rhs)
    # timer.report("Solution")
    return soln[0]

def make_free_surface():
    n = 35
    w = 5
    surface = rect_surface(n, n, [[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]])
    for i in range(surface[0].shape[0]):
        x = surface[0][i,0]
        y = surface[0][i,1]
        surface[0][i,2] += max(min(y, 0.5), 0)
    return surface

def make_fault(top_depth):
    return rect_surface(6, 6, [
        [-0.5, 0, top_depth], [-0.5, 0, top_depth - 1],
        [0.5, 0, top_depth - 1], [0.5, 0, top_depth]
    ])

# TODO: Generate D*N*B * D*N*B matrix instead of reshaping
@slow
def test_okada():

    sm = 1.0
    pr = 0.25

    timer = Timer()
    top_depth = -0.25
    surface = make_free_surface()
    all_mesh = mesh_concat(surface, make_fault(top_depth))
    surface_tris = all_mesh[1][:surface[1].shape[0]]
    fault_tris = all_mesh[1][surface[1].shape[0]:]
    timer.report("Mesh")

    iop = self_integral_operator(sm, pr, all_mesh[0], all_mesh[1])
    iop = reshape_iop(iop)
    timer.report("Integrals")
    cs = constraints(surface_tris, fault_tris)
    timer.report("Constraints")

    soln = solve(iop, cs)
    timer.report("Solve")

    disp = soln[:iop.shape[0]].reshape((3, int(iop.shape[0]/3))).T[:-6]
    vals = [None] * surface[0].shape[0]
    for i in range(surface[1].shape[0]):
        for d in range(3):
            idx = surface[1][i, d]
            vals[idx] = disp[3 * i + d,:]
    vals = np.array(vals)
    timer.report("Extract surface displacement")

    triang = tri.Triangulation(surface[0][:,0], surface[0][:,1], surface[1])
    for d in range(3):
        plt.figure()
        plt.tripcolor(triang, vals[:,d], shading = 'gouraud', cmap = 'PuOr', vmin = -0.02, vmax = 0.02)
        plt.title("u " + ['x', 'y', 'z'][d])
        plt.colorbar()


    lam = 2 * sm * pr / (1 - 2 * pr)
    alpha = (lam + sm) / (lam + 2 * sm)

    timer.restart()
    n_pts = surface[0].shape[0]
    obs_pts = surface[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = surface[0][i, :]
        pt[2] = 0
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, 0, 90, [-0.5, 0.5], [top_depth - 1, top_depth], [1, 0, 0]
        )
        u[i, :] = uv
    timer.report("Okada")


    # plt.figure()
    # plt.quiver(obs_pts[:, 0], obs_pts[:, 1], u[:, 0], u[:, 1])
    # plt.figure()
    # plt.streamplot(obs_pts[:, 0].reshape((n,n)), obs_pts[:, 1].reshape((n,n)), u[:, 0].reshape((n,n)), u[:, 1].reshape((n,n)))
    for d in range(3):
        plt.figure()
        plt.tripcolor(
            obs_pts[:, 0], obs_pts[:, 1], surface[1],
            u[:, d], shading='gouraud', cmap = 'PuOr',
            vmin = -0.02, vmax = 0.02
        )
        plt.title("Okada u " + ['x', 'y', 'z'][d])
        plt.colorbar()
    plt.show()

if __name__ == '__main__':
    test_okada()
