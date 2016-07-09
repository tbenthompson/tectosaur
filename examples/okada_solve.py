import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import pycuda.gpuarray as gpuarray
import skcuda.linalg as culg

from tectosaur.util.timer import Timer
from tectosaur.linalg import gpu_mvp

from okada_constraints import insert_constraints

def solve(iop, constraints):
    timer = Timer()
    culg.init()
    timer.report("culg.init()")
    n_iop = iop.mat.shape[0]
    n = n_iop + len(constraints)
    rhs = np.zeros(n)
    lhs_cs_dok = sparse.dok_matrix((n, n))
    insert_constraints(lhs_cs_dok, rhs, constraints)
    lhs_cs_csr = sparse.csr_matrix(lhs_cs_dok)
    timer.report("Build constraints matrix")

    iop = gpuarray.to_gpu(iop.mat.astype(np.float32))

    iter = [0]
    def mv(v):
        iter[0] += 1
        # print("It: " + str(iter[0]))
        # mvtimer = Timer(tabs = 1)
        out = np.empty(n)
        # mvtimer.report("Make vec")
        out = lhs_cs_csr.dot(v)
        # mvtimer.report("Mult constraints")
        out[:n_iop] += np.squeeze(gpu_mvp(iop, v[:n_iop,np.newaxis].astype(np.float32)))
        # out[:n_iop] += iop.dot(v[:n_iop])
        # mvtimer.report("Mult integral op")
        return out

    A = sparse.linalg.LinearOperator((n, n), matvec = mv)
    soln = sparse.linalg.lgmres(A, rhs)
    timer.report("GMRES")
    return soln[0]
