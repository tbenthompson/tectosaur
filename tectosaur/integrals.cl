<%
from tectosaur.nearfield_op import pairs_func_name

import tectosaur.util.kernel_exprs
kernel_names = ['U', 'T', 'A', 'H']
kernels = tectosaur.util.kernel_exprs.get_kernels(float_type)
def dn(dim):
    return ['x', 'y', 'z'][dim]

kronecker = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
%>
#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

<%def name="integrate_pair(k_name, limit)">
    ${prim.tri_info("obs", "n")}
    ${prim.tri_info("src", "l")}

    Real result_temp[81];

    for (int iresult = 0; iresult < 81; iresult++) {
        result_temp[iresult] = 0;
    }

    ${prim.constants()}
    
    for (int iq = 0; iq < n_quad_pts; iq++) {
        <% 
        qd = 4
        if limit:
            qd = 5
        %>
        Real obs_geom_xhat = quad_pts[iq * ${qd} + 0];
        Real obs_geom_yhat = quad_pts[iq * ${qd} + 1];
        Real src_geom_xhat = quad_pts[iq * ${qd} + 2];
        Real src_geom_yhat = quad_pts[iq * ${qd} + 3];
        % if limit:
            Real eps = quad_pts[iq * ${qd} + 4];
        % endif
        Real quadw = quad_wts[iq];

        % for which, ptname in [("obs", "x"), ("src", "y")]:
            ${prim.basis(which + "_geom_")}
            ${prim.pts_from_basis(
                ptname, which + "_geom_",
                lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
            )}
            % for d in range(3):
                Real ${which}b${d} = ${which}_geom_b${d};
            % endfor
        % endfor

        % if limit:
            % for dim in range(3):
                x${dn(dim)} -= eps * sqrt(obs_jacobian) * n${dn(dim)};
            % endfor
        % endif

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
    }
</%def>

<%def name="single_pairs(k_name, limit)">
__kernel
void ${pairs_func_name(limit, k_name)}(__global Real* result, 
    int n_quad_pts, __global Real* quad_pts, __global Real* quad_wts,
    __global Real* pts, __global int* obs_tris, __global int* src_tris, 
    Real G, Real nu)
{
    const int i = get_global_id(0);

    ${prim.get_triangle("obs_tri", "obs_tris", "i")}
    ${prim.get_triangle("src_tri", "src_tris", "i")}
    ${integrate_pair(k_name, limit)}
    
    for (int iresult = 0; iresult < 81; iresult++) {
        result[i * 81 + iresult] = result_temp[iresult];
    }
}
</%def>

<%def name="farfield_tris(k_name)">
__kernel
void farfield_tris${k_name}(__global Real* result,
    int n_quad_pts, __global Real* quad_pts, __global Real* quad_wts,
    __global Real* pts, int n_obs_tris, __global int* obs_tris, 
    int n_src_tris, __global int* src_tris, Real G, Real nu)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    ${prim.get_triangle("obs_tri", "obs_tris", "i")}
    ${prim.get_triangle("src_tri", "src_tris", "j")}
    ${integrate_pair(k_name, limit = False)}

    % for d_obs in range(3):
    % for d_src in range(3):
    % for b_obs in range(3):
    % for b_src in range(3):
    result[
        (i * 9 + ${b_obs} * 3 + ${d_obs}) * n_src_tris * 9 +
        (j * 9 + ${b_src} * 3 + ${d_src})
        ] = result_temp[${prim.temp_result_idx(d_obs, d_src, b_obs, b_src)}];
    % endfor
    % endfor
    % endfor
    % endfor
}
</%def>

<%def name="farfield_pts(k_name, need_obsn, need_srcn, constants_code)">
__kernel
void farfield_pts${k_name}(
    __global Real* result, __global Real* obs_pts, __global Real* obs_ns,
    __global Real* src_pts, __global Real* src_ns, __global Real* input,
    Real G, Real nu, int n_obs, int n_src)
{
    int i = get_global_id(0);
    int local_id = get_local_id(0);

    % for d in range(3):
    Real obsp${dn(d)};
    % endfor
    if (i < n_obs) {
        % for d in range(3):
        obsp${dn(d)} = obs_pts[i * 3 + ${d}];
        % endfor
    }

    % if need_obsn:
    % for d in range(3):
    Real M${dn(d)};
    % endfor
    if (i < n_obs) {
        % for d in range(3):
        M${dn(d)} = obs_ns[i * 3 + ${d}];
        % endfor
    }
    % endif

    % if need_srcn:
    __local Real sh_src_ns[3 * ${block_size}];
    % endif
    __local Real sh_src_pts[3 * ${block_size}];
    __local Real sh_input[3 * ${block_size}];

    
    ${prim.constants()}

    Real sumx = 0.0;
    Real sumy = 0.0;
    Real sumz = 0.0;

    int j = 0;
    int tile = 0;
    for (; j < n_src; j += ${block_size}, tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int idx = tile * ${block_size} + local_id;
        if (idx < n_src) {
            for (int k = 0; k < 3; k++) {
                % if need_srcn:
                sh_src_ns[local_id * 3 + k] = src_ns[idx * 3 + k];
                % endif
                sh_src_pts[local_id * 3 + k] = src_pts[idx * 3 + k];
                sh_input[local_id * 3 + k] = input[idx * 3 + k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i >= n_obs) {
            continue;
        }
        for (int k = 0; k < ${block_size} && k < n_src - j; k++) {
            Real Dx = sh_src_pts[k * 3] - obspx;
            Real Dy = sh_src_pts[k * 3 + 1] - obspy;
            Real Dz = sh_src_pts[k * 3 + 2] - obspz;

            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;
            if (r2 == 0.0) {
                continue;
            }
            % if need_srcn:
            % for d in range(3):
            Real N${dn(d)} = sh_src_ns[k * 3 + ${d}];
            % endfor
            % endif

            % for d in range(3):
            Real S${dn(d)} = sh_input[k * 3 + ${d}];
            % endfor

            ${prim.vector_kernels(k_name)}
        }
    }

    if (i < n_obs) {
        % for d in range(3):
        result[i * 3 + ${d}] = sum${dn(d)};
        % endfor
    }
}
</%def>

${prim.geometry_fncs()}

${farfield_pts("U", False, False, U_const_code)}
${farfield_pts("T", False, True, T_const_code)}
${farfield_pts("A", True, False, A_const_code)}
${farfield_pts("H", True, True, H_const_code)}

% for k_name in kernel_names:
${farfield_tris(k_name)}
${single_pairs(k_name, limit = True)}
${single_pairs(k_name, limit = False)}
% endfor
