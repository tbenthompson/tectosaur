import numpy as np

import cppimport
_fmm = cppimport.imp("tectosaur._fmm._fmm")._fmm._fmm
for k in dir(_fmm):
    locals()[k] = getattr(_fmm, k)

def eval(obs_kd, src_kd, fmm_mat, input_vals, n_outputs):
    tdim = fmm_mat.tensor_dim
    n_surf = fmm_mat.translation_surface_order
    n_multipoles = n_surf * len(src_kd.nodes) * tdim
    n_locals = n_surf * len(obs_kd.nodes) * tdim

    est = fmm_mat.p2p.matvec(input_vals, n_outputs)

    m_check = fmm_mat.p2m.matvec(input_vals, n_multipoles)
    multipoles = fmm_mat.uc2e[0].matvec(m_check, n_multipoles)

    for m2m, uc2e in zip(fmm_mat.m2m[1:], fmm_mat.uc2e[1:]):
        m_check = m2m.matvec(multipoles, n_multipoles)
        multipoles += uc2e.matvec(m_check, n_multipoles)

    l_check = fmm_mat.p2l.matvec(input_vals, n_locals)
    l_check += fmm_mat.m2l.matvec(multipoles, n_locals)

    locals = np.zeros(n_locals)
    for l2l, dc2e in zip(fmm_mat.l2l, fmm_mat.dc2e):
        l_check += l2l.matvec(locals, n_locals)
        locals += dc2e.matvec(l_check, n_locals)

    est += fmm_mat.m2p.matvec(multipoles, n_outputs)

    est += fmm_mat.l2p.matvec(locals, n_outputs)
    return est
