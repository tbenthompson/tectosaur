# python examples/okada.py 61 7
# python examples/okada.py 21 3
#n = [8, 16, 32, 64, 128, 256]
#l2 = [0.0149648012534, 0.030572079265, 0.00867837671259, 0.00105034618493, 6.66984415273e-05, 4.07689295549e-06]
#linf = [0.008971091166208367, 0.014749192806577716, 0.0093510756645549115, 0.0042803891552975898, 0.0013886177492512669, 0.000338113427521]

import sys
import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import okada_wrapper
import scipy.spatial

import tectosaur
import tectosaur.mesh.refine as mesh_refine
import tectosaur.mesh.modify as mesh_modify
import tectosaur.mesh.mesh_gen as mesh_gen
import tectosaur.constraints as constraints
from tectosaur.constraint_builders import continuity_constraints, \
    constant_bc_constraints, free_edge_constraints
from tectosaur.util.timer import Timer
from tectosaur.interior import interior_integral
from tectosaur.ops.sparse_integral_op import SparseIntegralOp
from tectosaur.ops.sparse_farfield_op import PtToPtFMMFarfieldOp
from tectosaur.ops.mass_op import MassOp
from tectosaur.ops.sum_op import SumOp
from tectosaur.check_for_problems import check_for_problems

import solve

from tectosaur.util.logging import setup_root_logger
logger = setup_root_logger(__name__)

class Okada:
    def __init__(self, n_surf, n_fault, top_depth = 0.0, fault_L = 1.0):
        log_level = logging.INFO
        tectosaur.logger.setLevel(log_level)
        solve.logger.setLevel(log_level)
        logger.setLevel(log_level)
        self.k_params = [1.0, 0.25]
        self.fault_L = fault_L
        self.top_depth = top_depth
        self.load_soln = False
        self.float_type = np.float32
        self.n_surf = n_surf
        self.n_fault = n_fault#max(2, n_surf // 5)
        self.all_mesh, self.surface_tris, self.fault_tris = make_meshes(
            self.fault_L, self.top_depth, self.n_surf, self.n_fault
        )
        logger.info('n_elements: ' + str(self.all_mesh[1].shape[0]))

        self.n_surf_tris = self.surface_tris.shape[0]
        self.n_fault_tris = self.fault_tris.shape[0]
        self.n_tris = self.all_mesh[1].shape[0]
        self.surf_tri_idxs = np.arange(self.n_surf_tris)
        self.fault_tri_idxs = np.arange(self.n_surf_tris, self.n_tris)
        self.n_surf_dofs = self.n_surf_tris * 9
        self.n_dofs = self.n_tris * 9
        # mesh_gen.plot_mesh3d(*all_mesh)
        # _,_,_,_ = check_for_problems(self.all_mesh, check = True)

    def run(self, build_and_solve = None):
        if build_and_solve is None:
            build_and_solve = build_and_solve_T
        if not self.load_soln:
            soln = build_and_solve(self)
            np.save('okada.npy', soln)
        else:
            soln = np.load('okada.npy')
        return soln

    def okada_exact(self):
        obs_pts = self.all_mesh[0]
        sm, pr = self.k_params
        lam = 2 * sm * pr / (1 - 2 * pr)
        alpha = (lam + sm) / (lam + 2 * sm)
        print(lam, sm, pr, alpha)

        n_pts = obs_pts.shape[0]
        u = np.empty((n_pts, 3))
        for i in range(n_pts):
            pt = obs_pts[i, :]
            [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
                alpha, pt, -self.top_depth, 90.0,
                [-self.fault_L, self.fault_L], [-1.0, 0.0], [1.0, 0.0, 0.0]
            )
            if suc != 0:
                u[i, :] = 0
            else:
                u[i, :] = uv
        return u

    def xsec_plot(self, solns, okada_soln = None, show = True):
        xsec_pts = []
        xsec_idxs = []
        xsec_vals = [[] for j in range(len(solns))]
        xsec_vals_okada = []
        for i in range(self.surface_tris.shape[0]):
            for pt_idx in range(3):
                p = self.all_mesh[0][self.surface_tris[i,pt_idx],:]
                if np.abs(p[0]) > 0.001:
                    continue
                xsec_pts.append(p)
                for j in range(len(solns)):
                    xsec_vals[j].append([solns[j][i * 9 + pt_idx * 3 + d] for d in range(3)])
                if okada_soln is not None:
                    xsec_vals_okada.append([
                        okada_soln[self.all_mesh[1][i,pt_idx]][d] for d in range(3)
                    ])
                xsec_idxs.append([i * 9 + pt_idx * 3 + d for d in range(3)])
        xsec_pts = np.array(xsec_pts)
        xsec_vals = np.array(xsec_vals)
        xsec_vals_okada = np.array(xsec_vals_okada)
        plt.figure()
        for j in range(len(solns)):
            plt.plot(xsec_pts[:,1], xsec_vals[j,:,0], 'o-', label = str(j))
        if okada_soln is not None:
            plt.plot(xsec_pts[:,1], xsec_vals_okada[:,0], 'o-', label = 'okada')
        # plt.savefig('okada_xsec.pdf', bbox_inches = 'tight')
        plt.legend()
        if show:
            plt.show()

    def plot_interior_displacement(self, soln):
        nxy = 40
        nz = 40
        d = 0
        xs = np.linspace(-10, 10, nxy)
        zs = np.linspace(-0.1, -4.0, nz)
        X, Y, Z = np.meshgrid(xs, xs, zs)
        obs_pts = np.array([X.flatten(), Y.flatten(), Z.flatten()]).T.copy()
        t = Timer(output_fnc = logger.debug)
        interior_disp = -interior_integral(
            obs_pts, obs_pts, self.all_mesh, soln, 'elasticT3',
            3, 8, self.k_params, self.float_type,
            fmm_params = None#[100, 3.0, 3000, 25]
        ).reshape((nxy, nxy, nz, 3))
        t.report('eval %.2E interior pts' % obs_pts.shape[0])

        for i in range(nz):
            plt.figure()
            plt.pcolor(xs, xs, interior_disp[:,:,i,d])
            plt.colorbar()
            plt.title('at z = ' + ('%.3f' % zs[i]) + '    u' + ['x', 'y', 'z'][d])
            plt.show()

    def get_pt_soln(self, soln):
        pt_soln = np.empty((self.all_mesh[0].shape[0], 3))
        pt_soln[self.all_mesh[1]] = soln.reshape((-1, 3, 3))
        return pt_soln

    def plot_results(self, soln, okada_soln):
        pts, tris = self.all_mesh
        est = self.get_pt_soln(soln)

        vmax = np.max(okada_soln)
        for d in range(3):
            plt.figure()
            plt.tripcolor(
                pts[:,0], pts[:, 1], tris,
                est[:,d], #shading='gouraud',
                cmap = 'PuOr', vmin = -vmax, vmax = vmax
            )
            plt.title("u " + ['x', 'y', 'z'][d])
            plt.colorbar()

        for d in range(3):
            plt.figure()
            plt.tripcolor(
                pts[:, 0], pts[:, 1], tris,
                okada_soln[:, d], #shading='gouraud',
                cmap = 'PuOr', vmin = -vmax, vmax = vmax
            )
            plt.title("Okada u " + ['x', 'y', 'z'][d])
            plt.colorbar()

        for d in range(3):
            plt.figure()
            plt.tripcolor(
                pts[:, 0], pts[:, 1], tris,
                okada_soln[:, d] - est[:,d], #shading='gouraud',
                cmap = 'PuOr'
            )
            plt.title("Diff u " + ['x', 'y', 'z'][d])
            plt.colorbar()

        plt.show()

    def print_error(self, soln, okada_soln):
        est = self.get_pt_soln(soln)
        pts = self.all_mesh[0]
        close = np.sqrt(np.sum(pts ** 2, axis = 1)) < 4.0
        not0 = np.abs(pts[:,1]) > 1e-5
        test = np.logical_and(close, not0)
        diff = okada_soln[test,:] - est[test,:]
        l2diff = np.sum(diff ** 2)
        l2correct = np.sum(okada_soln[test,:] ** 2)
        linferr = np.max(np.abs(diff))
        print("L2diff: " + str(l2diff))
        print("L2correct: " + str(l2correct))
        print("L2relerr: " + str(l2diff / l2correct))
        print("maxerr: " + str(linferr))
        return linferr

def make_free_surface(w, n):
    corners = [[-w, -w, 0], [-w, w, 0], [w, w, 0], [w, -w, 0]]
    return mesh_gen.make_rect(n, n, corners)

def make_fault(L, top_depth, n_fault):
    m = mesh_gen.make_rect(n_fault, n_fault, [
        [-L, 0, top_depth], [-L, 0, top_depth - 1],
        [L, 0, top_depth - 1], [L, 0, top_depth]
    ])
    return m

def make_meshes(fault_L, top_depth, n_surf, n_fault):
    t = Timer(output_fnc = logger.debug)
    surf_w = 10
    surface = make_free_surface(surf_w, n_surf)
    t.report('make free surface')
    fault = make_fault(fault_L, top_depth, n_fault)
    t.report('make fault')
    all_mesh = mesh_modify.concat(surface, fault)
    t.report('concat meshes')
    surface_tris = all_mesh[1][:surface[1].shape[0]]
    fault_tris = all_mesh[1][surface[1].shape[0]:]
    return all_mesh, surface_tris, fault_tris

def build_constraints(surface_tris, fault_tris, pts):
    n_surf_tris = surface_tris.shape[0]
    n_fault_tris = fault_tris.shape[0]

    cs = continuity_constraints(surface_tris, fault_tris)

    # X component = 1
    # Y comp = Z comp = 0
    cs.extend(constant_bc_constraints(
        n_surf_tris, n_surf_tris + n_fault_tris, [1.0, 0.0, 0.0]
    ))
    cs.extend(free_edge_constraints(surface_tris))

    return cs

def build_and_solve_T(data):
    timer = Timer(output_fnc = logger.debug)
    cs = build_constraints(data.surface_tris, data.fault_tris, data.all_mesh[0])
    timer.report("Constraints")

    T_op = SparseIntegralOp(
        6, 2, 5, 2.0,
        'elasticT3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type,
        farfield_op_type = PtToPtFMMFarfieldOp(150, 3.0, 450)
    )
    timer.report("Integrals")

    mass_op = MassOp(3, data.all_mesh[0], data.all_mesh[1])
    iop = SumOp([T_op, mass_op])
    timer.report('mass op/sum op')

    soln = solve.iterative_solve(iop, cs, tol = 1e-6)
    timer.report("Solve")
    return soln

def build_and_solve_H(data):
    timer = Timer(output_fnc = logger.debug)
    cs = build_constraints(data.surface_tris, data.fault_tris, data.all_mesh[0])
    timer.report("Constraints")

    iop = SumOp([SparseIntegralOp(
        8, 3, 6, 3.0,
        'elasticH3', data.k_params, data.all_mesh[0], data.all_mesh[1],
        data.float_type,
        farfield_op_type = PtToPtFMMFarfieldOp(150, 3.0, 450)
    )])
    timer.report("Integrals")

    soln = solve.iterative_solve(iop, cs, tol = 1e-6)
    timer.report("Solve")
    return soln

def main():
    t = Timer(output_fnc = logger.info)
    obj = Okada(int(sys.argv[1]), n_fault = int(sys.argv[2]))
    soln = obj.run()
    t.report('tectosaur')
    okada_soln = obj.okada_exact()
    t.report('okada')
    obj.xsec_plot([soln], okada_soln)
    # obj.plot_interior_displacement(soln)
    obj.print_error(soln, okada_soln)
    t.report('check')
    # obj.plot_results(soln, okada_soln)

if __name__ == '__main__':
    main()
