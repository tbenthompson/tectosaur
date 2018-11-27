<%
from tectosaur.nearfield.pairs_integrator import pairs_func_name
from tectosaur.kernels import elastic_kernels, kernels

K = kernels[kernel_name]

def dn(dim):
    return ['x', 'y', 'z'][dim]
%>
${cluda_preamble}

#define Real ${float_type}

<%namespace name="prim" file="../integral_primitives.cl"/>

<%def name="setup_pair()">
    const int i = get_global_id(0);
    const int pair_idx = i + start_idx;
    if (pair_idx >= end_idx) {
        return;
    }
</%def>

<%def name="single_pairs(K, check0)">
KERNEL
void ${pairs_func_name(check0)}(GLOBAL_MEM Real* result, 
    int n_quad_pts, GLOBAL_MEM Real* quad_pts, GLOBAL_MEM Real* quad_wts,
    GLOBAL_MEM Real* pts, GLOBAL_MEM int* tris, GLOBAL_MEM int* pairs_list, 
    int start_idx, int end_idx, GLOBAL_MEM Real* params)
{
    ${setup_pair()}

    const int obs_tri_idx = pairs_list[pair_idx * 2];
    const int src_tri_idx = pairs_list[pair_idx * 2 + 1];
    const int obs_tri_rot_clicks = 0;
    const int src_tri_rot_clicks = 0;
    ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
    ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
    ${prim.tri_info("obs", "tris", K.needs_obsn, K.surf_curl_obs)}
    ${prim.tri_info("src", "tris", K.needs_srcn, K.surf_curl_src)}
    ${prim.integrate_pair(K, check0)}
    
    for (int iresult = 0; iresult < 81; iresult++) {
        result[i * 81 + iresult] = obs_jacobian * src_jacobian * result_temp[iresult];
    }
}
</%def>

<%def name="single_pairs_adj(K)">
KERNEL
void ${pairs_func_name(check0)}_adj(GLOBAL_MEM Real* result, 
    int n_quad_pts, GLOBAL_MEM Real* quad_pts, GLOBAL_MEM Real* quad_wts,
    GLOBAL_MEM Real* pts, GLOBAL_MEM int* tris, GLOBAL_MEM int* pairs_list, 
    int start_idx, int end_idx, GLOBAL_MEM Real* params)
{
    ${setup_pair()}

    const int obs_tri_idx = pairs_list[pair_idx * 5];
    const int src_tri_idx = pairs_list[pair_idx * 5 + 1];
    const int obs_tri_rot_clicks = pairs_list[pair_idx * 5 + 2];
    const int src_tri_rot_clicks = pairs_list[pair_idx * 5 + 3];
    const bool src_tri_flip = pairs_list[pair_idx * 5 + 4];
    ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
    ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
    ${prim.tri_info("obs", "tris", K.needs_obsn, K.surf_curl_obs)}
    ${prim.tri_info("src", "tris", K.needs_srcn, K.surf_curl_src, need_flip = True)}
    ${prim.integrate_pair(K, True)}

    //printf("obs1: %f %f %f\n", obs_tri[0][0], obs_tri[0][1], obs_tri[0][2]);
    //printf("obs2: %f %f %f\n", obs_tri[1][0], obs_tri[1][1], obs_tri[1][2]);
    //printf("obs3: %f %f %f\n", obs_tri[2][0], obs_tri[2][1], obs_tri[2][2]);
    //printf("src1: %f %f %f\n", src_tri[0][0], src_tri[0][1], src_tri[0][2]);
    //printf("src2: %f %f %f\n", src_tri[1][0], src_tri[1][1], src_tri[1][2]);
    //printf("src3: %f %f %f\n", src_tri[2][0], src_tri[2][1], src_tri[2][2]);
    //printf("obsjac: %f srcjac: %f \n", obs_jacobian, src_jacobian);
    
    Real derot_factor = -1;
    int obs_derot[3];
    for (size_t i = 0; i < 3; i++) {
        obs_derot[i] = positive_mod(-obs_tri_rot_clicks + i, 3);
    }
    int src_derot[3];
    for (size_t i = 0; i < 3; i++) {
        src_derot[i] = positive_mod(-src_tri_rot_clicks + i, 3);
    }
    if (src_tri_flip) { 
        int temp = src_derot[0];
        src_derot[0] = src_derot[1];
        src_derot[1] = temp;
    } 
    for (int b1 = 0; b1 < 3; b1++) {
        for (int d1 = 0; d1 < 3; d1++) {
            for (int b2 = 0; b2 < 3; b2++) {
                for (int d2 = 0; d2 < 3; d2++) {
                    int out_idx = b1 * 27 + d1 * 9 + b2 * 3 + d2;
                    int in_idx = obs_derot[b1] * 27 + d1 * 9 + src_derot[b2] * 3 + d2;
                    Real val = obs_jacobian * src_jacobian * result_temp[in_idx];
                    if (src_tri_flip) {
                        val *= -1;
                    }
                    result[i * 81 + out_idx] = val;
                }
            }
        }
    }
}
</%def>

<%def name="farfield_tris(K)">
KERNEL
void farfield_tris(GLOBAL_MEM Real* result,
    int n_quad_pts, GLOBAL_MEM Real* quad_pts, GLOBAL_MEM Real* quad_wts,
    GLOBAL_MEM Real* pts, int n_obs_tris, GLOBAL_MEM int* obs_tris, 
    int n_src_tris, GLOBAL_MEM int* src_tris, 
    GLOBAL_MEM Real* params)
{
    const int obs_tri_idx = get_global_id(0);
    const int src_tri_idx = get_global_id(1);
    const int obs_tri_rot_clicks = 0;
    const int src_tri_rot_clicks = 0;
    ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
    ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
    ${prim.tri_info("obs", "obs_tris", K.needs_obsn, K.surf_curl_obs)}
    ${prim.tri_info("src", "src_tris", K.needs_srcn, K.surf_curl_src)}

    ${prim.integrate_pair(K, check0 = False)}

    for (int b_obs = 0; b_obs < 3; b_obs++) {
    for (int d_obs = 0; d_obs < 3; d_obs++) {
    for (int b_src = 0; b_src < 3; b_src++) {
    for (int d_src = 0; d_src < 3; d_src++) {
        result[
            (obs_tri_idx * 9 + b_obs * 3 + d_obs) * n_src_tris * 9 +
            (src_tri_idx * 9 + b_src * 3 + d_src)
            ] = obs_jacobian * src_jacobian * 
                result_temp[b_obs * 27 + d_obs * 9 + b_src * 3 + d_src];
    }
    }
    }
    }
}
</%def>

${prim.geometry_fncs()}
${single_pairs(K, check0 = True)}
${single_pairs(K, check0 = False)}
${single_pairs_adj(K)}
${farfield_tris(K)}
