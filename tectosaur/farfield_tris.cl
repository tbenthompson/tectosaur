<%
from tectosaur.kernels import kernels
K = kernels['elasticT3']
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

${cluda_preamble}

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

${prim.geometry_fncs()}

<%def name="farfield_tris(K)">
KERNEL
void farfield_tris${K.name}(
    GLOBAL_MEM Real* result, GLOBAL_MEM Real* input, 
    GLOBAL_MEM Real* pts, GLOBAL_MEM int* obs_tris, GLOBAL_MEM int* src_tris,
    GLOBAL_MEM Real* quad_pts, GLOBAL_MEM Real* quad_wts, GLOBAL_MEM Real* params,
    int n_obs, int n_src, int n_quad_pts)
{
    <%
        dofs_per_el = K.spatial_dim * K.tensor_dim
    %>
    const int obs_tri_idx = get_global_id(0);
    const int obs_tri_rot_clicks = 0;
    int local_id = get_local_id(0);

    ${K.constants_code}

    ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
    Real sum[${dofs_per_el}];
    if (obs_tri_idx < n_obs) {
        ${prim.tri_info("obs", "obs_tris", K.needs_obsn, K.surf_curl_obs, force_normal)}
        for (int k = 0; k < ${dofs_per_el}; k++) {
            sum[k] = 0.0;
        }
    }

    LOCAL_MEM Real sh_quad_pts[1024]; // Limits quad order to 4 (4^5)
    LOCAL_MEM Real sh_quad_wts[256];
    for (int start_iq = 0; start_iq < n_quad_pts; start_iq += ${block_size}) {
        int iq = start_iq + local_id;
        if (iq > n_quad_pts) {
            continue;
        }
        for (int i = 0; i < 4; i++) {
            sh_quad_pts[iq * 4 + i] = quad_pts[iq * 4 + i];
        }
        sh_quad_wts[iq] = quad_wts[iq];
    }
    LOCAL_BARRIER;

    if (obs_tri_idx >= n_obs) {
        return;
    }

    for (int j = 0; j < n_src; j++) {
        const int src_tri_idx = j;
        const int src_tri_rot_clicks = 0;
        ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
        ${prim.tri_info("src", "src_tris", K.needs_srcn, K.surf_curl_src, force_normal)}

        Real in[${dofs_per_el}];
        for (int k = 0; k < ${dofs_per_el}; k++) {
            in[k] = input[src_tri_idx * ${dofs_per_el} + k];
        }

        for (int iq = 0; iq < n_quad_pts; iq++) {
            Real obsxhat = sh_quad_pts[iq * 4 + 0];
            Real obsyhat = sh_quad_pts[iq * 4 + 1];
            Real srcxhat = sh_quad_pts[iq * 4 + 2];
            Real srcyhat = sh_quad_pts[iq * 4 + 3];
            Real quadw = sh_quad_wts[iq];

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

            // TODO: There is substantially more performance possible for the elasticRH kernel.
            ${prim.call_vector_code(K)}

            for (int b_obs = 0; b_obs < 3; b_obs++) {
                % for d_obs in range(3):
                sum[b_obs * 3 + ${d_obs}] += factor * obsb[b_obs] * sum${dn(d_obs)};
                % endfor
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
