<%
from tectosaur.kernels import kernels
K = kernels['elasticT3']
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

${cluda_preamble}

#define Real ${float_type}

CONSTANT Real quad_pts[${quad_pts.size}] = {${str(quad_pts.flatten().tolist())[1:-1]}};
CONSTANT Real quad_wts[${quad_wts.size}] = {${str(quad_wts.flatten().tolist())[1:-1]}};

<%namespace name="prim" file="integral_primitives.cl"/>

${prim.geometry_fncs()}

<%def name="farfield_tris(K)">
KERNEL
void farfield_tris${K.name}(
    GLOBAL_MEM Real* result, GLOBAL_MEM Real* input, 
    GLOBAL_MEM Real* pts, GLOBAL_MEM int* obs_tris, GLOBAL_MEM int* src_tris,
    GLOBAL_MEM Real* params, int n_obs, int n_src)
{
    <%
        dofs_per_el = K.spatial_dim * K.tensor_dim
    %>
    const int obs_tri_idx = get_global_id(0);
    const int obs_tri_rot_clicks = 0;
    int local_id = get_local_id(0);

    if (obs_tri_idx >= n_obs) {
        return;
    }

    ${K.constants_code}

    ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
    ${prim.tri_info("obs", "pts", "obs_tris", K.needs_obsn, K.surf_curl_obs)}

    Real sum[${dofs_per_el}];
    for (int k = 0; k < ${dofs_per_el}; k++) {
        sum[k] = 0.0;
    }

    for (int j = 0; j < n_src; j++) {
        const int src_tri_idx = j;
        const int src_tri_rot_clicks = 0;
        ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
        ${prim.tri_info("src", "pts", "src_tris", K.needs_srcn, K.surf_curl_src)}

        Real in[${dofs_per_el}];
        for (int k = 0; k < ${dofs_per_el}; k++) {
            in[k] = input[src_tri_idx * ${dofs_per_el} + k];
        }

        for (int iq1 = 0; iq1 < ${quad_wts.shape[0]}; iq1++) {
            Real obsxhat = quad_pts[iq1 * 2 + 0];
            Real obsyhat = quad_pts[iq1 * 2 + 1];
            for (int iq2 = 0; iq2 < ${quad_wts.shape[0]}; iq2++) {
                Real srcxhat = quad_pts[iq2 * 2 + 0];
                Real srcyhat = quad_pts[iq2 * 2 + 1];
                Real quadw = quad_wts[iq1] * quad_wts[iq2];

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

                Real factor = obs_jacobian * src_jacobian * quadw;
                % for d in range(3):
                    Real sum${dn(d)} = 0.0;
                    Real in${dn(d)} = 0.0;
                    for (int b_src = 0; b_src < 3; b_src++) {
                        in${dn(d)} += in[b_src * 3 + ${d}] * srcb[b_src];
                    }
                % endfor

                ${prim.call_vector_code(K)}

                for (int b_obs = 0; b_obs < 3; b_obs++) {
                    % for d_obs in range(3):
                    sum[b_obs * 3 + ${d_obs}] += factor * obsb[b_obs] * sum${dn(d_obs)};
                    % endfor
                }
            }
        }
    }

    for (int k = 0; k < ${dofs_per_el}; k++) {
        result[obs_tri_idx * ${dofs_per_el} + k] = sum[k];
    }
}
</%def>

% for name,K in kernels.items():
${farfield_tris(K)}
% endfor
