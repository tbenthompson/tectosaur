import numpy as np
import cppimport

fast_modify = cppimport.imp('tectosaur.mesh.fast_modify')

def remove_duplicate_pts(m):
    return fast_modify.remove_duplicate_pts(m[0], m[1])

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
