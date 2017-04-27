import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from tectosaur.util.timer import Timer
from tectosaur.constraints import lagrange_constraints, build_constraint_matrix

# master-slave constrained direct solves look like:
# Kˆ uˆ = ˆf, in which Kˆ = TT K T, ˆf = TT (f − K g).
def direct_solve(iop, constraints):
    cm, rhs = build_constraint_matrix(constraints, iop.shape[0])
    cm = cm.tocsr()
    cmT = cm.T
    iop_constrained = cmT.dot(cmT.dot(iop.mat.T).T)
    rhs_constrained = cmT.dot(-iop.mat.dot(rhs))
    soln_constrained = np.linalg.solve(iop_constrained, rhs_constrained)
    soln = cm.dot(soln_constrained)
    return soln

def iterative_solve(iop, constraints, rhs = None):
    timer = Timer()
    cm, c_rhs = build_constraint_matrix(constraints, iop.shape[0])
    timer.report('Build constraint matrix')
    cm = cm.tocsr()
    cmT = cm.T
    nearfield_constrained = cmT.dot(iop.nearfield.dot(cm))
    if rhs is None:
        rhs_constrained = cmT.dot(-iop.dot(c_rhs))
    else:
        rhs_constrained = cmT.dot(rhs - iop.dot(c_rhs))
    timer.report('Constrain linear system')

    n = rhs_constrained.shape[0]

    iter = [0]
    def mv(v):
        iter[0] += 1
        print(iter[0])
        out = nearfield_constrained.dot(v)
        out += cmT.dot(iop.farfield_dot(cm.dot(v)))
        return out

    P = sparse.linalg.spilu(cmT.dot(iop.nearfield.mat_no_correction.dot(cm)))
    timer.report("Build preconditioner")
    def prec_f(x):
        return P.solve(x)
    M = sparse.linalg.LinearOperator((n, n), matvec = prec_f)
    A = sparse.linalg.LinearOperator((n, n), matvec = mv)

    def report_res(R):
        print(R)
        pass
    soln = sparse.linalg.gmres(
        A, rhs_constrained, M = M, tol = 1e-8, callback = report_res, restart = 100
    )
    timer.report("GMRES")
    return cm.dot(soln[0]) + c_rhs

def lagrange_iterative_solve(iop, constraints):
    timer = Timer()
    n_iop = iop.shape[0]
    n = n_iop + len(constraints)
    rhs = np.zeros(n)
    lhs_cs_dok = sparse.dok_matrix((n, n))
    lagrange_constraints(lhs_cs_dok, rhs, constraints)
    lhs_cs_csr = lhs_cs_dok.tocsr()
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

    def report_res(R):
        print(R)
        pass
    soln = sparse.linalg.gmres(A, rhs, tol = 5e-6, callback = report_res, restart = 100)
    timer.report("GMRES")
    return soln[0]

def lagrange_direct_solve(iop, constraints):
    n = iop.shape[0] + len(constraints)
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[:iop.shape[0],:iop.shape[0]] = iop.mat
    lagrange_constraints(A, b, constraints)
    soln = np.linalg.solve(A, b)
    return soln
