import scipy.sparse
import numpy as np

from cppimport import cppimport
fast_constraints = cppimport('tectosaur.fast_constraints')

for k in dir(fast_constraints):
    locals()[k] = getattr(fast_constraints, k)

def build_constraint_matrix(cs, n_total_dofs):
    rows, cols, vals, rhs = fast_constraints.build_constraint_matrix(cs, n_total_dofs)
    n_rows = n_total_dofs
    n_cols = n_total_dofs - np.count_nonzero(rhs)
    cm = scipy.sparse.csr_matrix((vals, (rows, cols)), shape = (n_rows, n_cols))
    return cm, rhs
