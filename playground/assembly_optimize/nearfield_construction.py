import sys
import numpy as np
import scipy.sparse
from tectosaur.util.timer import Timer
import cppimport

helpers = cppimport.imp("helpers")
make_bsr_matrix = helpers.make_bsr_matrix
derotate_adj_mat = helpers.derotate_adj_mat

# Before optimizations:
# total took 3.184598207473755
# dot took 0.02

# now with bsr_matrix:
# total build took 0.14
# dot took 0.01

def pairs_sparse_mat(obs_idxs, src_idxs, integrals):
    return integrals.reshape((-1, 9, 9)), obs_idxs, src_idxs

def co_sparse_mat(co_indices, co_mat, correction):
    return pairs_sparse_mat(co_indices, co_indices, co_mat - correction)

@profile
def adj_sparse_mat(adj_mat, tri_idxs, obs_clicks, src_clicks, correction_mat):
    adj_mat = adj_mat.astype(np.float32)
    derotate_adj_mat(adj_mat, obs_clicks, src_clicks)
    return pairs_sparse_mat(tri_idxs[:,0],tri_idxs[:,1], adj_mat - correction_mat)

def near_sparse_mat(near_mat, near_pairs, near_correction):
    return pairs_sparse_mat(near_pairs[:, 0], near_pairs[:, 1], near_mat - near_correction)

@profile
def build_nearfield(co_data, ea_data, va_data, near_data, shape):
    t = Timer(tabs = 2)
    co_vals,co_rows,co_cols = co_sparse_mat(*co_data)
    ea_vals,ea_rows,ea_cols = adj_sparse_mat(*ea_data)
    va_vals,va_rows,va_cols = adj_sparse_mat(*va_data)
    near_vals,near_rows,near_cols = near_sparse_mat(*near_data)
    t.report("build pairs")
    rows = np.concatenate((co_rows, ea_rows, va_rows, near_rows))
    cols = np.concatenate((co_cols, ea_cols, va_cols, near_cols))
    vals = np.concatenate((co_vals, ea_vals, va_vals, near_vals))
    t.report("stack pairs")

    data, indices, indptr = make_bsr_matrix(shape[0], shape[1], 9, 9, vals, rows, cols)
    t.report("to bsr")
    mat = scipy.sparse.bsr_matrix((data, indices, indptr))
    t.report('make bsr')

    return mat


np.random.seed(133)
D = np.load('build_nearfield_input.npy')
# D = np.load('smaller_build_nearfield_input.npy')
co_data, ea_data, va_data, near_data = D
n = (np.max(co_data[0]) + 1) * 9
shape = (n, n)

t = Timer()
mat = build_nearfield(co_data, ea_data, va_data, near_data, shape)
t.report('total')
v = np.random.rand(mat.shape[1])
t.report('make v')
for i in range(10):
    out = mat.dot(v)
t.report('dot')
np.testing.assert_almost_equal(np.load('check.npy'), out)
# np.testing.assert_almost_equal(np.load('smaller_check.npy'), out)
# np.save('small_check_mat.npy', mat.todense())
# np.save('smaller_check.npy', out)
