import numpy as np
import tectosaur as tct
from scipy.sparse.linalg import cg, lsqr
import scipy.sparse

def get_disp_slip_to_traction2(m, cfg, H):
    csUS = tct.continuity_constraints(m.pts, m.tris, m.get_start('fault'))
    csUF = tct.continuity_constraints(m.pts, m.get_tris('fault'), m.get_end('fault'))
    csU = tct.build_composite_constraints((csUS, 0), (csUF, m.n_dofs('surf')))
    csU.extend(tct.free_edge_constraints(m.tris))
    cmU, c_rhsU, _ = tct.build_constraint_matrix(csU, m.n_dofs())
    np.testing.assert_almost_equal(c_rhsU, 0.0)

    csT_continuity, csT_admissibility = tct.traction_admissibility_constraints(
        m.pts, m.tris
    )
    csT = csT_continuity + tct.free_edge_constraints(m.tris)
    cmT, c_rhsT, _ = tct.build_constraint_matrix(csT, m.n_dofs())
    np.testing.assert_almost_equal(c_rhsT, 0.0)

    cm_admissibility, rhs_admissibility = tct.simple_constraint_matrix(
        csT_admissibility, m.n_dofs()
    )

    traction_mass_op = tct.MassOp(
        cfg['tectosaur_cfg']['quad_mass_order'], m.pts, m.tris
    )
    constrained_traction_mass_op = cmU.T.dot(traction_mass_op.mat.dot(cmT))

    def f(disp_slip):
        def callback(x):
            callback.iter += 1
            print(callback.iter)
        callback.iter = 0

        full_lhs = scipy.sparse.vstack((
            constrained_traction_mass_op,
            cm_admissibility.dot(cmT)
        ))
        full_rhs = np.concatenate((
            -cmU.T.dot(H.dot(disp_slip.flatten())),
            rhs_admissibility
        ))

        # soln = cg(full_lhs, full_rhs)#, callback = callback)
        soln = lsqr(full_lhs, full_rhs)
        out = cfg['sm'] * (cmT.dot(soln[0]) + c_rhsT)
        return out
    return f

