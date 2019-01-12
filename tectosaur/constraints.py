import scipy.sparse
import numpy as np

from tectosaur.util.cpp import imp
fast_constraints = imp('tectosaur.fast_constraints')

for k in dir(fast_constraints):
    locals()[k] = getattr(fast_constraints, k)

def build_constraint_matrix(cs, n_total_dofs):
    rows, cols, vals, rhs_rows, rhs_cols, rhs_vals, rhs_in, n_unique_cs = \
        fast_constraints.build_constraint_matrix(cs, n_total_dofs)
    n_rows = n_total_dofs
    n_cols = n_total_dofs - n_unique_cs
    cm = scipy.sparse.csr_matrix((vals, (rows, cols)), shape = (n_rows, n_cols))
    rhs_mat = scipy.sparse.csr_matrix((rhs_vals, (rhs_rows, rhs_cols)), shape = (n_rows, len(cs)))
    return cm, rhs_mat.dot(rhs_in), rhs_mat
