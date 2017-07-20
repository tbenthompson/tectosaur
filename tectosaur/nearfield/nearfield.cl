<%
from tectosaur.nearfield.nearfield_op import pairs_func_name
from tectosaur.kernels import elastic_kernels, kernels

def dn(dim):
    return ['x', 'y', 'z'][dim]
%>
#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define Real ${float_type}

<%namespace name="prim" file="../integral_primitives.cl"/>

<%def name="integrate_pair(K, limit, check0)">
    ${prim.tri_info("obs", "nobs")}
    ${prim.tri_info("src", "nsrc")}

    Real result_temp[81];
    Real kahanC[81];

    for (int iresult = 0; iresult < 81; iresult++) {
        result_temp[iresult] = 0;
        kahanC[iresult] = 0;
    }

    ${K.constants_code}
    
    for (int iq = 0; iq < n_quad_pts; iq++) {
        <% 
        qd = 4
        if limit:
            qd = 5
        %>
        Real obsxhat = quad_pts[iq * ${qd} + 0];
        Real obsyhat = quad_pts[iq * ${qd} + 1];
        Real srcxhat = quad_pts[iq * ${qd} + 2];
        Real srcyhat = quad_pts[iq * ${qd} + 3];
        % if limit:
            Real eps = quad_pts[iq * ${qd} + 4];
        % endif
        Real quadw = quad_wts[iq];

        % for which, ptname in [("obs", "x"), ("src", "y")]:
            ${prim.basis(which)}
            ${prim.pts_from_basis(
                ptname, which,
                lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
            )}
        % endfor

        % if limit:
            % for dim in range(3):
                x${dn(dim)} -= eps * sqrt(obs_jacobian) * nobs${dn(dim)};
            % endfor
        % endif

        Real Dx = yx - xx;
        Real Dy = yy - xy; 
        Real Dz = yz - xz;
        Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

        % if check0:
        if (r2 == 0.0) {
            continue;
        }
        % endif

        ${K.tensor_code}
        obsb0 *= quadw;
        obsb1 *= quadw;
        obsb2 *= quadw;

        Real val, y, t;
        % for b_obs in range(3):
            % for d_obs in range(3):
                % for b_src in range(3):
                    % for d_src in range(3):
                        <%
                            idx = b_obs * 27 + d_obs * 9 + b_src * 3 + d_src;
                        %> 
                        val = obsb${b_obs} * srcb${b_src} * K${d_obs}${d_src};
                        y = val - kahanC[${idx}];
                        t = result_temp[${idx}] + y;
                        kahanC[${idx}] = (t - result_temp[${idx}]) - y;
                        result_temp[${idx}] = t;
                    % endfor
                % endfor
            % endfor
        % endfor
    }
</%def>

<%def name="single_pairs(K, limit, check0)">
__kernel
void ${pairs_func_name(limit, K.name, check0)}(__global Real* result, 
    int n_quad_pts, __global Real* quad_pts, __global Real* quad_wts,
    __global Real* pts, __global int* obs_tris, __global int* src_tris, 
    __global Real* params)
{
    const int i = get_global_id(0);

    ${prim.get_triangle("obs_tri", "obs_tris", "i")}
    ${prim.get_triangle("src_tri", "src_tris", "i")}
    ${integrate_pair(K, limit, check0)}
    
    for (int iresult = 0; iresult < 81; iresult++) {
        result[i * 81 + iresult] = obs_jacobian * src_jacobian * result_temp[iresult];
    }
}
</%def>

<%def name="farfield_tris(K)">
__kernel
void farfield_tris${K.name}(__global Real* result,
    int n_quad_pts, __global Real* quad_pts, __global Real* quad_wts,
    __global Real* pts, int n_obs_tris, __global int* obs_tris, 
    int n_src_tris, __global int* src_tris, 
    __global Real* params)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    ${prim.get_triangle("obs_tri", "obs_tris", "i")}
    ${prim.get_triangle("src_tri", "src_tris", "j")}
    ${integrate_pair(K, limit = False, check0 = False)}

    % for d_obs in range(3):
    % for d_src in range(3):
    % for b_obs in range(3):
    % for b_src in range(3):
    result[
        (i * 9 + ${b_obs} * 3 + ${d_obs}) * n_src_tris * 9 +
        (j * 9 + ${b_src} * 3 + ${d_src})
        ] = obs_jacobian * src_jacobian * 
            result_temp[${prim.temp_result_idx(d_obs, d_src, b_obs, b_src)}];
    % endfor
    % endfor
    % endfor
    % endfor
}
</%def>

${prim.geometry_fncs()}

% for k_name,K in elastic_kernels.items():
${single_pairs(K, limit = True, check0 = True)}
${single_pairs(K, limit = True, check0 = False)}
${single_pairs(K, limit = False, check0 = True)}
${single_pairs(K, limit = False, check0 = False)}
${farfield_tris(K)}
% endfor
