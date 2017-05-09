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

<%def name="calc_quad_pt_contrib(k_name)">
    % for which, ptname in [("obs", "x"), ("src", "y")]:
        ${prim.basis(which)}
        ${prim.pts_from_basis(
            ptname, which,
            lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
        )}
    % endfor

    Real Dx = yx - xx;
    Real Dy = yy - xy; 
    Real Dz = yz - xz;
    Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;
    if (r2 == 0.0) {
        continue;
    }

    ${prim.tensor_kernels(k_name)}

    % for d_obs in range(3):
    % for d_src in range(3):
    {
        Real kernel_val = obs_jacobian * src_jacobian * quadw * K${d_obs}${d_src};
        % for b_obs in range(3):
        % for b_src in range(3):
        {
            int idx = ${prim.temp_result_idx(d_obs, d_src, b_obs, b_src)};
            result_temp[idx] += obsb${b_obs} * srcb${b_src} * kernel_val;
        }
        % endfor
        % endfor
    }
    % endfor
    % endfor
</%def>

<%def name="beta_integral(k_name, beta_min, beta_max, lim_idx)">
for (int i_b = 0; i_b < ${nq}; i_b++) {
    Real b = pt_to_interval(qx[i_b], ${beta_min}, ${beta_max}); 
    Real bw = wt_to_interval(qw[i_b], ${beta_min}, ${beta_max}); 
    % if lim_idx == 0:
    Real alpha_max = rho_max(tp) / cos(b);
    % else:
    Real alpha_max = rho_max(tq) / sin(b);
    % endif
    for (int i_a = 0; i_a < ${nq}; i_a++) {
        Real a = pt_to_interval(qx[i_a], 0, alpha_max);
        Real aw = wt_to_interval(qw[i_a], 0, alpha_max);
        Real jacobian = (a * a * a) * cos(b) * sin(b);
        Real rho_p = a * cos(b);
        Real rho_q = a * sin(b);
        Real obsxhat = rho_p * cos(tp);
        Real obsyhat = rho_p * sin(tp);
        Real srcxhat = rho_q * cos(tq);
        Real srcyhat = rho_q * sin(tq);
        Real quadw = tpw * tqw * bw * aw * jacobian;

        ${calc_quad_pt_contrib(k_name)}
    }
}
</%def>

<%def name="vert_adj_pairs(k_name)">
__kernel
void vert_adj_pairs${k_name}(__global Real* result, 
    __global Real* pts, __global int* obs_tris, __global int* src_tris, 
    Real G, Real nu)
{
    const int i = get_global_id(0);

    ${prim.get_triangle("obs_tri", "obs_tris", "i")}
    ${prim.get_triangle("src_tri", "src_tris", "i")}
    ${prim.tri_info("obs", "n")}
    ${prim.tri_info("src", "l")}

    Real result_temp[81];

    for (int iresult = 0; iresult < 81; iresult++) {
        result_temp[iresult] = 0;
    }

    ${prim.constants()}

    for (int i_tp = 0; i_tp < ${nq}; i_tp++) {
        Real tp = pt_to_interval(qx[i_tp], 0, M_PI / 2);
        Real tpw = wt_to_interval(qw[i_tp], 0, M_PI / 2);
        for (int i_tq = 0; i_tq < ${nq}; i_tq++) {
            Real tq = pt_to_interval(qx[i_tq], 0, M_PI / 2); 
            Real tqw = wt_to_interval(qw[i_tq], 0, M_PI / 2); 

            Real beta_split = atan(rho_max(tq) / rho_max(tq));

            ${beta_integral(k_name, "0", "beta_split", 0)}
            ${beta_integral(k_name, "beta_split", "M_PI / 2", 1)}
        }
    }

    for (int iresult = 0; iresult < 81; iresult++) {
        result[i * 81 + iresult] = result_temp[iresult];
    }
}
</%def>

${prim.geometry_fncs()}

${vert_adj_pairs("H")}
% for k_name in kernel_names:

% endfor
