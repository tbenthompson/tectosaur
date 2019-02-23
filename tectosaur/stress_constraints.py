import numpy as np
from tectosaur.constraints import ConstraintEQ, Term

basis_gradient = np.array([[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]]).T
def rot_mat(tri):
    v1 = tri[1] - tri[0]
    v1 /= np.linalg.norm(v1)
    v2 = tri[2] - tri[0]
    n = np.cross(v1, v2)
    n /= np.linalg.norm(n)
    L1 = v1
    L2 = np.cross(n, v1)
    L3 = n
    return np.array([L1,L2,L3])

def jacobian(tri):
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    n = np.cross(v1, v2)
    J = np.array([v1, v2, n])
    return J[:2,].T # map from xhat to x

def inv_jacobian(tri):
    v1 = tri[1] - tri[0]
    v2 = tri[2] - tri[0]
    n = np.cross(v1, v2)
    J = np.array([v1, v2, n])
    Jinv = np.linalg.inv(J.T)
    return Jinv[:2,:] # map from x to xhat

def calc_gradient(tri, disp):
    # map from x to rotated frame
    x_to_xp = rot_mat(tri)
    np.testing.assert_almost_equal(x_to_xp.T, np.linalg.inv(x_to_xp))

    # map from triangle reference coords to x
    x_to_xhat = inv_jacobian(tri)

    # displacement in the rotated frame
    disp_xp = disp.dot(x_to_xp.T)

    # displacement derivatives with respect to triangle reference coords
    # disp_xp_dxhat[i,j] = d[u_i]/dxhat_j
    disp_xp_dxhat = basis_gradient.dot(disp_xp).T

    # displacement derivatives with respect to x
    # disp_xp_dx[i,j] = d[u_i]/dx_j
    disp_xp_dx = disp_xp_dxhat.dot(x_to_xhat)

    # displacement derivatives with respect to rotated frame
    # disp_xp_dxp[i,j] = d[u_i]/dxp_j
    disp_xp_dxp = disp_xp_dx.dot(x_to_xp.T)

    return disp_xp_dxp, x_to_xp

def derive_stress(tri_data, sm, pr):
    tri, disp, tri_idx, _ = tri_data
    disp_xp_dxp, x_to_xp = calc_gradient(tri, disp)

    strain = 0.5 * (disp_xp_dxp + disp_xp_dxp.T)
    # if np.any(strain) > 0:
    #     from IPython.core.debugger import Tracer
    #     Tracer()()

    lame_lambda = 2 * sm * pr / (1 - 2 * pr)
    youngs = 2 * sm * (1 + pr)

    e11 = np.array([strain[0,0], 0,0,0])
    e22 = np.array([strain[1,1], 0,0,0])
    e33 = np.array(
        [-pr/(1-pr) * (strain[0,0] + strain[1,1])]
        + ((1 + pr) * (1 - 2 * pr)/(youngs * (1 - pr)) * x_to_xp[2]).tolist()
    )
    e12 = np.array([strain[0,1], 0,0,0])
    e13 = np.array([0] + ((0.5 / sm) * x_to_xp[0]).tolist())
    e23 = np.array([0] + ((0.5 / sm) * x_to_xp[1]).tolist())

    etrace = e11 + e22 + e33

    E = np.array([[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]])

    S = 2 * sm * E
    for k in range(3):
        S[k,k] += lame_lambda * etrace

    return (E, S, x_to_xp)

def rotate_tensor(tensor, R):
    return np.swapaxes(np.swapaxes(np.array([
        R.T.dot(tensor[:,:,k]).dot(R)
        for k in range(4)
    ]),0,1),1,2)

def Sdot(stress, vec):
    return np.swapaxes(np.array([
        stress[:,:,k].dot(vec)
        for k in range(4)
    ]),0,1)

def stress_constraints(tri_data1, tri_data2, sm, pr):
    out = []
    E1, S1, x_to_xp1 = derive_stress(tri_data1, sm, pr)
    E2, S2, x_to_xp2 = derive_stress(tri_data2, sm, pr)

    tri1, _, tri_idx1, corner_idx1 = tri_data1
    dof_start1 = tri_idx1 * 9 + corner_idx1 * 3
    n1 = np.cross(tri1[1] - tri1[0], tri1[2] - tri1[0])
    n1 /= np.linalg.norm(n1)

    tri2, _, tri_idx2, corner_idx2 = tri_data2
    dof_start2 = tri_idx2 * 9 + corner_idx2 * 3
    n2 = np.cross(tri2[1] - tri2[0], tri2[2] - tri2[0])
    n2 /= np.linalg.norm(n2)

    S1xx = rotate_tensor(S1, x_to_xp1)
    np.testing.assert_almost_equal(Sdot(S1xx, n1), np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1]]))
    S2xx = rotate_tensor(S2, x_to_xp2)
    np.testing.assert_almost_equal(Sdot(S2xx, n2), np.array([[0,1,0,0],[0,0,1,0],[0,0,0,1]]))

    def make_constraint(lhs, rhs):
        terms = []
        for k in range(3):
            terms.append(Term(lhs[1+k], dof_start1 + k))
            terms.append(Term(-rhs[1+k], dof_start2 + k))
        out.append(ConstraintEQ(terms, -lhs[0] + rhs[0]))

    lhs1 = n2.dot(Sdot(S1xx, n1))
    rhs1 = n2.dot(Sdot(S2xx, n1))
    make_constraint(lhs1, rhs1)

    r2 = np.cross(n1, n2)
    r2 /= np.linalg.norm(r2)

    r1 = np.cross(n1, r2)
    r3 = np.cross(n2, r2)

    lhs2 = S1xx[0,0] + S1xx[1,1] + S1xx[2,2]
    rhs2 = S2xx[0,0] + S2xx[1,1] + S2xx[2,2]
    # norm = np.linalg.norm(lhs2[1:])
    # lhs2 /= norm
    # rhs2 /= norm
    # lhs2[0] *= -1.5
    # rhs2[0] *= -1.5
    # make_constraint(lhs2, rhs2)

    nmid = (n1 + n2) / 2.0
    nmid /= np.linalg.norm(nmid)
    lhs3 = nmid.dot(Sdot(S1xx, r2))
    rhs3 = nmid.dot(Sdot(S2xx, r2))
    norm = np.linalg.norm(lhs3[1:])
    lhs3 /= norm
    rhs3 /= norm
    # lhs3[0] /= -7.4
    # rhs3[0] /= -7.4
    # make_constraint(lhs3, rhs3)
    if np.any(S1xx[:,:,0]) != 0:
        print('C1: ', lhs1, rhs1)
        print('C2: ', lhs2, rhs2)
        print('C3: ', lhs3, rhs3)
        from IPython.core.debugger import Tracer
        Tracer()()

    return out

def equilibrium_constraint(tri_data):
    tri, tri_idx, corner_idx = tri_data

    x_to_xp = rot_mat(tri) # map from x to rotated frame
    x_to_xhat = inv_jacobian(tri) # map from triangle reference coords to x
    xp_to_xhat = x_to_xhat.dot(x_to_xp.T)

    vals = np.zeros((3,3))
    for b in range(3):
        for i in range(3):
            for Ik in range(2):
                for Ip in range(3):
                    vals[b, i] += (
                        basis_gradient[Ik,b]
                        * xp_to_xhat[Ik,Ip]
                        * x_to_xp[Ip,i]
                    )
    terms = []
    for b in range(3):
        for d in range(3):
            terms.append(Term(vals[b,d], tri_idx * 9 + b * 3 + d))

    return ConstraintEQ(terms, 0.0)

def constant_stress_constraint(tri_data1, tri_data2):
    tri1, tri_idx1, corner_idx1 = tri_data1
    dof_start1 = tri_idx1 * 9 + corner_idx1 * 3
    n1 = np.cross(tri1[1] - tri1[0], tri1[2] - tri1[0])
    n1 /= np.linalg.norm(n1)

    tri2, tri_idx2, corner_idx2 = tri_data2
    dof_start2 = tri_idx2 * 9 + corner_idx2 * 3
    n2 = np.cross(tri2[1] - tri2[0], tri2[2] - tri2[0])
    n2 /= np.linalg.norm(n2)

    terms = []
    for d in range(3):
        terms.append(Term(n1[d], dof_start2 + d))
        terms.append(Term(-n2[d], dof_start1 + d))
    return ConstraintEQ(terms, 0.0)

def stress_constraints2(tri_data1, tri_data2):
    out = []
    tri1, tri_idx1, corner_idx1 = tri_data1
    dof_start1 = tri_idx1 * 9 + corner_idx1 * 3
    n1 = np.cross(tri1[1] - tri1[0], tri1[2] - tri1[0])
    n1 /= np.linalg.norm(n1)

    tri2, tri_idx2, corner_idx2 = tri_data2
    dof_start2 = tri_idx2 * 9 + corner_idx2 * 3
    n2 = np.cross(tri2[1] - tri2[0], tri2[2] - tri2[0])
    n2 /= np.linalg.norm(n2)

    out.append(equilibrium_constraint(tri_data1))
    out.append(equilibrium_constraint(tri_data2))

    return out
