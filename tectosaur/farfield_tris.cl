<%
from tectosaur.kernels import kernels
K = kernels['elasticT3']
def dn(dim):
    return ['x', 'y', 'z'][dim]

e = [[[int((i - j) * (j - k) * (k - i) / 2) for k in range(3)]
    for j in range(3)] for i in range(3)]
%>

${cluda_preamble}

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

${prim.geometry_fncs()}

KERNEL
void farfield_triselasticRT3(
    GLOBAL_MEM Real* result, GLOBAL_MEM Real* input, 
    GLOBAL_MEM Real* pts, GLOBAL_MEM int* obs_tris, GLOBAL_MEM int* src_tris,
    GLOBAL_MEM Real* quad_pts, GLOBAL_MEM Real* quad_wts, GLOBAL_MEM Real* params,
    int n_obs, int n_src, int n_quad_pts)
{
    <%
        dofs_per_tri = K.spatial_dim * K.tensor_dim
    %>
    const int obs_tri_idx = get_global_id(0);
    const int obs_tri_rot_clicks = 0;
    int local_id = get_local_id(0);

    const Real nu = params[1];
    const Real CsRT0 = 1.0/(8.0*M_PI*(1.0-nu));

    if (obs_tri_idx < n_obs) {
        ${prim.get_triangle("obs_tri", "obs_tris")}
        ${prim.tri_info("obs", "nobs", K.needs_obsn)}
        ${prim.surf_curl("obs")}

        Real sum[${dofs_per_tri}];
        for (int k = 0; k < ${dofs_per_tri}; k++) {
            sum[k] = 0.0;
        }

        for (int j = 0; j < n_src; j++) {
            const int src_tri_idx = j;
            const int src_tri_rot_clicks = 0;
            ${prim.get_triangle("src_tri", "src_tris")}
            ${prim.tri_info("src", "nsrc", K.needs_srcn)}
            ${prim.surf_curl("src")}

            Real in[${dofs_per_tri}];
            for (int k = 0; k < ${dofs_per_tri}; k++) {
                in[k] = input[j * ${dofs_per_tri} + k];
            }

            for (int iq = 0; iq < n_quad_pts; iq++) {
                Real obsxhat = quad_pts[iq * 4 + 0];
                Real obsyhat = quad_pts[iq * 4 + 1];
                Real srcxhat = quad_pts[iq * 4 + 2];
                Real srcyhat = quad_pts[iq * 4 + 3];
                Real quadw = quad_wts[iq];

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
                Real invr = rsqrt(r2);

                % for d_obs in range(3):
                % for d_src in range(3):
                for (int b_obs = 0; b_obs < 3; b_obs++) {
                for (int b_src = 0; b_src < 3; b_src++) {
                    Real Kval = 0.0;
                    % for Ij in range(3):
                    % for Ip in range(3):
                        Kval += CsRT0 * invr * (
                            (1 - 2 * nu) * ${e[d_obs][d_src][Ij]}
                            - ${e[d_obs][Ip][Ij]} * invr * invr * 
                                D${dn(Ip)} * D${dn(d_src)}
                        ) * src_surf_curl[b_src][${Ij}];
                    % endfor
                    % endfor

                    sum[b_obs * 3 + ${d_obs}] +=
                        obs_jacobian * src_jacobian * quadw
                        * obsb[b_obs] * Kval
                        * in[b_src * 3 + ${d_src}];
                }
                }
                % endfor
                % endfor
            }
        }
        for (int k = 0; k < ${dofs_per_tri}; k++) {
            result[obs_tri_idx * ${dofs_per_tri} + k] = sum[k];
        }
    }
}
