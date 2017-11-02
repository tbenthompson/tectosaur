import numpy as np
import cppimport
import scipy.spatial

fast_modify = cppimport.imp('tectosaur.mesh.fast_modify')

def remove_duplicate_pts(m, threshold = None):
    dim = m[0].shape[1]
    if threshold is None:
        default_threshold_factor = 1e-13
        spatial_range = np.max(np.max(m[0], axis = 0) - np.min(m[0], axis = 0))
        threshold = spatial_range * default_threshold_factor
    kd = scipy.spatial.cKDTree(m[0])
    pairs = np.array(list(kd.query_pairs(threshold)))
    if len(pairs.shape) == 1:
        ordered_by_dependent = np.array([[-1,-1]])
    else:
        sorted_pairs = np.sort(pairs, axis = 1)
        ordered_by_dependent = sorted_pairs[np.argsort(sorted_pairs[:,1]),:]
    fnc = getattr(fast_modify, 'remove_duplicate_pts' + str(dim))
    return fnc(m[0], m[1], ordered_by_dependent)

def concat_two(m1, m2):
    return remove_duplicate_pts(concat_two_no_remove_duplicates(m1, m2))

def concat_two_no_remove_duplicates(m1, m2):
    return np.vstack((m1[0], m2[0])), np.vstack((m1[1], m2[1] + m1[0].shape[0]))

def concat(*ms):
    m_full = ms[0]
    for m in ms[1:]:
        m_full = concat_two(m_full, m)
    return m_full

def flip_normals(m):
    return (m[0], np.array([[m[1][i,0],m[1][i,2],m[1][i,1]] for i in range(m[1].shape[0])]))
