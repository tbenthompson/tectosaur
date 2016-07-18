import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from tectosaur.util.timer import Timer
from tectosaur.constraints import lagrange_constraints, build_constraint_matrix

# Kˆ uˆ = ˆf, in which Kˆ = TT K T, ˆf = TT (f − K g).
def lagrange_direct_solve(iop, constraints):
    n = iop.shape[0] + len(constraints)
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[:iop.shape[0],:iop.shape[0]] = iop.mat
    lagrange_constraints(A, b, constraints)
    soln = np.linalg.solve(A, b)
    return soln

def direct_solve(iop, constraints):
    cm, rhs = build_constraint_matrix(constraints, iop.shape[0])
    cm = cm.tocsr()
    cmT = cm.T
    iop_constrained = cmT.dot(cmT.dot(iop.mat.T).T)
    rhs_constrained = cmT.dot(-iop.mat.dot(rhs))
    AAA = np.load('/home/tbent/projects/3bem/devlib/AAA.npy')
    AAA = np.swapaxes(np.swapaxes(AAA.reshape((3, 9, 3, 9)), 0, 1), 2, 3).reshape((27,27))
    import ipdb; ipdb.set_trace()
    soln_constrained = np.linalg.solve(iop_constrained, rhs_constrained)
    soln = cm.dot(soln_constrained)
    return soln

def iterative_solve(iop, constraints):
    timer = Timer()
    n_iop = iop.shape[0]
    n = n_iop + len(constraints)
    rhs = np.zeros(n)
    lhs_cs_dok = sparse.dok_matrix((n, n))
    lagrange_constraints(lhs_cs_dok, rhs, constraints)
    lhs_cs_csr = sparse.csr_matrix(lhs_cs_dok)
    timer.report("Build constraints matrix")

    # n_coo = iop.nearfield.tocoo()
    # c_coo = lhs_cs_csr.tocoo()
    # vals = np.concatenate((n_coo.data, c_coo.data))
    # rows = np.concatenate((n_coo.row, c_coo.row))
    # cols = np.concatenate((n_coo.col, c_coo.col))
    # N_plus_C = scipy.sparse.coo_matrix((vals, (rows, cols))).tocsc()
    # P = sparse.linalg.spilu(N_plus_C)
    # def prec_f(x):
    #     return P.solve(x.astype(np.float32))
    # M = sparse.linalg.LinearOperator((n, n), matvec = prec_f)
    # timer.report("Build sparse ILU preconditioner")

    iter = [0]
    def mv(v):
        iter[0] += 1
        print(iter[0])
        out = np.empty(n)
        out = lhs_cs_csr.dot(v)
        out[:n_iop] += iop.dot(v[:n_iop])
        return out

    A = sparse.linalg.LinearOperator((n, n), matvec = mv)

    def report_res(R):
        print(R)
    soln = sparse.linalg.gmres(A, rhs, tol = 1e-6, callback = report_res, restart = 100)
    timer.report("GMRES")
    return soln[0]
