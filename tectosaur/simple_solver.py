import numpy as np
import scipy.sparse.linalg

from tectosaur.util.timer import Timer

import logging
logger = logging.getLogger(__name__)

defaults = dict(
    solver_tol = 1e-8,
    log_level = logging.DEBUG
)

def iterative_solve(iop, cm, rhs, prec, cfg):
    rhs_constrained = cm.T.dot(rhs)
    n = rhs_constrained.shape[0]

    def mv(v):
        t = Timer(output_fnc = logger.debug)
        mv.iter += 1
        out = cm.T.dot(iop.dot(cm.dot(v)))
        t.report('iteration # ' + str(mv.iter))
        return out
    mv.iter = 0
    A = scipy.sparse.linalg.LinearOperator((n, n), matvec = mv)
    M = scipy.sparse.linalg.LinearOperator((n, n), matvec = prec)

    soln = scipy.sparse.linalg.gmres(
        A, rhs_constrained, M = M, tol = cfg['solver_tol'],
        callback = report_res, restart = 500
    )

    out = cm.dot(soln[0])
    return out
