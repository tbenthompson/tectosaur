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

    iter = [0]
    def mv(v):
        iter[0] += 1
        print(iter[0])
        out = np.empty(n)
        out = lhs_cs_csr.dot(v)
        out[:n_iop] += iop.dot(v[:n_iop])
        return out

    A = sparse.linalg.LinearOperator((n, n), matvec = mv)
    soln = sparse.linalg.lgmres(A, rhs)
    timer.report("GMRES")
    return soln[0]
