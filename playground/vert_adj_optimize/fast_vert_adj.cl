<%
from tectosaur.integral_utils import pairs_func_name, kernel_names, dn, kronecker
from tectosaur.quadrature import gaussxw

qx, qw = gaussxw(nq)
%>
#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

<%def name="static_array(name, py_arr)">
__constant Real ${name}[${py_arr.shape[0]}] = {
% for i in range(py_arr.shape[0] - 1):
    ${py_arr[i]},
% endfor
    ${py_arr[-1]}
};
</%def>

${static_array("qx", qx)}
${static_array("qw", qw)}

Real pt_to_interval(Real x, Real a, Real b) {
    return a + (b - a) * (x + 1.0) / 2.0;
}

Real wt_to_interval(Real w, Real a, Real b) {
    return w / 2.0 * (b - a);
}

Real rho_max(Real theta) {
    return 1.0 / (cos(theta) + sin(theta));
}

<%def name="tensor_kernels2(k_name)">
// 6
Real invr = 1.0 / sqrt(r2);
Real invr2 = invr * invr;
Real invr3 = invr2 * invr;
Real Dorx = invr * Dx;
Real Dory = invr * Dy;
Real Dorz = invr * Dz;

// 15
Real rn = lx * Dorx + ly * Dory + lz * Dorz;
Real rm = nx * Dorx + ny * Dory + nz * Dorz;
Real mn = nx * lx + ny * ly + nz * lz;

// 5
Real Q = CsH0 * invr3;
Real A = Q * 3 * rn;
Real B = Q * CsH1;
Real C = Q * CsH3;

// 15
Real MTx = Q*CsH2*lx + A*CsH1*Dorx;
Real MTy = Q*CsH2*ly + A*CsH1*Dory;
Real MTz = Q*CsH2*lz + A*CsH1*Dorz;

// 12
Real NTx = B*nx + C*Dorx*rm;
Real NTy = B*ny + C*Dory*rm;
Real NTz = B*nz + C*Dorz*rm;

// 36
Real DTx = B*3*lx*rm + C*Dorx*mn + A*(nu*nx - 5*Dorx*rm);
Real DTy = B*3*ly*rm + C*Dory*mn + A*(nu*ny - 5*Dory*rm);
Real DTz = B*3*lz*rm + C*Dorz*mn + A*(nu*nz - 5*Dorz*rm);

// 4
Real ST = A*nu*rm + B*mn;

// 48
K[0][0] = lx*NTx + nx*MTx + Dorx*DTx + ST;
K[0][1] = lx*NTy + nx*MTy + Dorx*DTy;
K[0][2] = lx*NTz + nx*MTz + Dorx*DTz;
K[1][0] = ly*NTx + ny*MTx + Dory*DTx;
K[1][1] = ly*NTy + ny*MTy + Dory*DTy + ST;
K[1][2] = ly*NTz + ny*MTz + Dory*DTz;
K[2][0] = lz*NTx + nz*MTx + Dorz*DTx;
K[2][1] = lz*NTy + nz*MTy + Dorz*DTy;
K[2][2] = lz*NTz + nz*MTz + Dorz*DTz + ST;

// 37
</%def>

<%def name="basis(prefix)">
float ${prefix}b[3];
${prefix}b[0] = 1 - ${prefix}xhat - ${prefix}yhat;
${prefix}b[1] = ${prefix}xhat;
${prefix}b[2] = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name,ndims)">
% for dim in range(ndims):
Real ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b[${basis}] * ${tri_name(basis,dim)};
% endfor
% endfor
</%def>

<%def name="calc_quad_pt_contrib(k_name)">
    % for which, ptname in [("obs", "x"), ("src", "y")]:
        ${basis(which)}
        ${pts_from_basis(
            ptname, which,
            lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
        )}
    % endfor

    Real Dx = yx - xx;
    Real Dy = yy - xy; 
    Real Dz = yz - xz;
    Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

    float K[3][3];
    ${tensor_kernels2(k_name)}
    
    for (int d_obs = 0; d_obs < 3; d_obs++) {
    for (int d_src = 0; d_src < 3; d_src++) {
    Real kernel_val = quadw * K[d_obs][d_src];
    for (int b_obs = 0; b_obs < 3; b_obs++) {
    for (int b_src = 0; b_src < 3; b_src++) {
        int idx = b_obs * 27 + d_obs * 9 + b_src * 3 + d_src;
        Real val = obsb[b_obs] * srcb[b_src] * kernel_val;
        Real y = val - kahanC[i_group * 81 + idx];
        Real t = result_temp[i_group * 81 + idx] + y;
        kahanC[i_group * 81 + idx] = (t - result_temp[i_group * 81 + idx]) - y;
        result_temp[i_group * 81 + idx] = t;
    }
    }
    }
    }
</%def>

<%def name="beta_integral(k_name, beta_min, beta_max, lim_idx)">
for (int i_b = 0; i_b < ${nq}; i_b++) {
    Real b = pt_to_interval(qxx[i_b], ${beta_min}, ${beta_max}); 
    Real bw = wt_to_interval(qww[i_b], ${beta_min}, ${beta_max}); 
    Real cosb = cos(b);
    Real sinb = sin(b);
    Real outer_wt_jacobian = tpw * tqw * bw * cosb * sinb * tris_jacobian;
    % if lim_idx == 0:
    Real alpha_max = 0.5 / ((costp + sintp) * cosb);
    % else:
    Real alpha_max = 0.5 / ((costq + sintq) * sinb);
    % endif
    for (int i_a = 0; i_a < ${nq}; i_a++) {
        // Real a = pt_to_interval(qxx[i_a], 0, alpha_max);
        // Real aw = wt_to_interval(qww[i_a], 0, alpha_max);
        Real a = alpha_max * (qxx[i_a] + 1);
        Real aw = qww[i_a] * alpha_max;
        Real rho_p = a * cosb;
        Real rho_q = a * sinb;
        Real obsxhat = rho_p * costp;
        Real obsyhat = rho_p * sintp;
        Real srcxhat = rho_q * costq;
        Real srcyhat = rho_q * sintq;
        Real quadw = outer_wt_jacobian * aw * a * a * a;

        ${calc_quad_pt_contrib(k_name)}
    }
}
</%def>

<%def name="vert_adj_pairs(k_name)">
__kernel
void vert_adj_pairs${k_name}(__global Real* result, 
    __global Real* pts, __global int* obs_tris, __global int* src_tris, 
    Real G, Real nu, int n_items)
{
    const int i = get_global_id(0);
    const int i_group = get_local_id(0);

    if (i > n_items) {
        return;
    }

    ${prim.get_triangle("obs_tri", "obs_tris", "i")}
    ${prim.get_triangle("src_tri", "src_tris", "i")}
    ${prim.tri_info("obs", "n")}
    ${prim.tri_info("src", "l")}
    Real tris_jacobian = obs_jacobian * src_jacobian;

    __local Real result_temp[${group_size} * 81];
    __local Real kahanC[${group_size} * 81];

    for (int iresult = 0; iresult < 81; iresult++) {
        result_temp[i_group * 81 + iresult] = 0;
        kahanC[i_group * 81 + iresult] = 0;
    }
    
    __local Real qxx[${nq}];
    __local Real qww[${nq}];
    for (int qi = 0; qi < ${nq}; qi++) {
        qxx[qi] = qx[qi];
        qww[qi] = qw[qi];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    ${prim.constants()}

    for (int i_tp = 0; i_tp < ${nq}; i_tp++) {
        Real tp = pt_to_interval(qxx[i_tp], 0, M_PI / 2);
        Real tpw = wt_to_interval(qww[i_tp], 0, M_PI / 2);
        Real costp = cos(tp);
        Real sintp = sin(tp);
        for (int i_tq = 0; i_tq < ${nq}; i_tq++) {
            Real tq = pt_to_interval(qxx[i_tq], 0, M_PI / 2); 
            Real tqw = wt_to_interval(qww[i_tq], 0, M_PI / 2); 
            Real costq = cos(tq);
            Real sintq = sin(tq);

            Real beta_split = atan((costp + sintp) / (costq + sintq));

            ${beta_integral(k_name, "0", "beta_split", 0)}
            ${beta_integral(k_name, "beta_split", "M_PI / 2", 1)}
        }
    }

    for (int iresult = 0; iresult < 81; iresult++) {
        result[i * 81 + iresult] = result_temp[i_group * 81 + iresult];
    }
}
</%def>

${prim.geometry_fncs()}

${vert_adj_pairs("H")}
% for k_name in kernel_names:

% endfor
