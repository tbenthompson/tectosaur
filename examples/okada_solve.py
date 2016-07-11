import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from tectosaur.util.timer import Timer

from okada_constraints import insert_constraints

def solve(iop, constraints):
    timer = Timer()
    n_iop = iop.shape[0]
    n = n_iop + len(constraints)
    rhs = np.zeros(n)
    lhs_cs_dok = sparse.dok_matrix((n, n))
    insert_constraints(lhs_cs_dok, rhs, constraints)
    lhs_cs_csr = sparse.csr_matrix(lhs_cs_dok)
    timer.report("Build constraints matrix")

    # n_coo = iop.nearfield_no_correction.tocoo()
    # c_coo = lhs_cs_csr.tocoo()
    # vals = np.concatenate((n_coo.data, c_coo.data))
    # rows = np.concatenate((n_coo.row, c_coo.row))
    # cols = np.concatenate((n_coo.col, c_coo.col))
    # N_plus_C = scipy.sparse.coo_matrix((vals, (rows, cols))).tocsc()
    # P = sparse.linalg.spilu(N_plus_C)
    # def prec_fun(x):
    #     return P.solve(x.astype(np.float32))
    # M = sparse.linalg.LinearOperator((n, n), matvec = prec_fun)
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
    soln = sparse.linalg.gmres(A, rhs, tol = 7e-5, callback = report_res)
    timer.report("GMRES")
    return soln[0]
