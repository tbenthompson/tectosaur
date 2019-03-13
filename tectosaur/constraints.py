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

def simple_constraint_matrix(cs, n_cols):
    rows = []
    cols = []
    data = []
    rhs = np.zeros((len(cs)))
    for i in range(len(cs)):
        c = cs[i]
        for j in range(len(c.terms)):
            rows.append(i)
            cols.append(c.terms[j].dof)
            data.append(c.terms[j].val)
        rhs[i] = c.rhs
    return (
        scipy.sparse.csr_matrix((data, (rows, cols)), shape = (len(cs), n_cols)),
        rhs
    )


