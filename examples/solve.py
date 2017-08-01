import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from tectosaur import setup_logger
from tectosaur.util.timer import Timer
from tectosaur.constraints import build_constraint_matrix

logger = setup_logger(__name__)

# master-slave constrained direct solves look like:
# Kˆ uˆ = ˆf, in which Kˆ = TT K T, ˆf = TT (f − K g).
def direct_solve(iop, constraints, rhs = None):
    cm, c_rhs = build_constraint_matrix(constraints, iop.shape[0])
    cm = cm.tocsr()
    cmT = cm.T
    iop_constrained = cmT.dot(cmT.dot(iop.mat.T).T)
    if rhs is None:
        rhs_constrained = cmT.dot(-iop.mat.dot(c_rhs))
    else:
        rhs_constrained = cmT.dot(rhs - iop.mat.dot(c_rhs))
    soln_constrained = np.linalg.solve(iop_constrained, rhs_constrained)
    soln = cm.dot(soln_constrained)
    return soln

def iterative_solve(iop, constraints, rhs = None, tol = 1e-8):
    timer = Timer(logger = logger)
    cm, c_rhs = build_constraint_matrix(constraints, iop.shape[1])
    timer.report('Build constraint matrix')
    cm = cm.tocsr()
    timer.report('constraint matrix tocsr')
    cmT = cm.T
    if rhs is None:
        rhs_constrained = cmT.dot(-iop.dot(c_rhs))
    else:
        rhs_constrained = cmT.dot(rhs - iop.dot(c_rhs))
    timer.report('constrain rhs')

    n = rhs_constrained.shape[0]

    iter = [0]
    def mv(v):
        iter[0] += 1
        logger.debug('iteration # ' + str(iter[0]))
        t = Timer(logger = logger)
        cm_res = cm.dot(v)
        t.report('constraint matrix multiply')
        iop_res = iop.dot(cm_res)
        t.report('integral operator multiply')
        out = cmT.dot(iop_res)
        t.report('constraint matrix transpose multiply')
        return out

    # P = sparse.linalg.spilu(cmT.dot(iop.nearfield_no_correction_dot(cm)))
    timer.report("Build preconditioner")
    def prec_f(x):
        # return P.solve(x)
        return x
    M = sparse.linalg.LinearOperator((n, n), matvec = prec_f)
    A = sparse.linalg.LinearOperator((n, n), matvec = mv)

    def report_res(R):
        logger.debug('residual: ' + str(R))
        pass
    soln = sparse.linalg.gmres(
        A, rhs_constrained, M = M, tol = tol, callback = report_res, restart = 200
    )
    timer.report("GMRES")
    return cm.dot(soln[0]) + c_rhs
