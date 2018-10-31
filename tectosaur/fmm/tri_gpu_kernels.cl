<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]
from tectosaur.kernels import kernels
%>
${cluda_preamble}

#define Real ${gpu_float_type}

<%namespace name="prim" file="../integral_primitives.cl"/>

${prim.geometry_fncs()}

CONSTANT Real surf_pts[${surf_pts.size}] = {${str(surf_pts.flatten().tolist())[1:-1]}};
CONSTANT int surf_tris[${surf_tris.size}] = {${str(surf_tris.flatten().tolist())[1:-1]}};
CONSTANT Real quad_pts[${quad_pts.size}] = {${str(quad_pts.flatten().tolist())[1:-1]}};
CONSTANT Real quad_wts[${quad_wts.size}] = {${str(quad_wts.flatten().tolist())[1:-1]}};

<%def name="func_def(op_name, obs_type, src_type)">
KERNEL
void ${op_name}_${K.name}(
        GLOBAL_MEM Real* out, GLOBAL_MEM Real* in,
        int n_blocks, GLOBAL_MEM Real* params,
        GLOBAL_MEM int* obs_n_idxs, GLOBAL_MEM int* obs_src_starts, GLOBAL_MEM int* src_n_idxs,
        ${params("obs", obs_type)}, ${params("src", src_type)})
</%def>

<%def name="params(name, type)">
% if type == "pts":
    GLOBAL_MEM int* ${name}_n_starts, GLOBAL_MEM int* ${name}_n_ends,
    GLOBAL_MEM Real* ${name}_pts, GLOBAL_MEM int* ${name}_tris
% else:
    GLOBAL_MEM Real* ${name}_n_center,
    GLOBAL_MEM Real* ${name}_n_R, Real ${name}_surf_r
% endif
</%def>

<%def name="get_block_idx()">
    const int global_idx = get_global_id(0); 
    const int worker_idx = get_local_id(0);
    const int block_idx = (global_idx - worker_idx) / ${n_workers_per_block};
    const int this_obs_n_idx = obs_n_idxs[block_idx];
    const int this_obs_src_start = obs_src_starts[block_idx];
    const int this_obs_src_end = obs_src_starts[block_idx + 1];
</%def>

<%def name="setup_block(name, type)">
% if type == "pts":
    int ${name}_start = ${name}_n_starts[this_${name}_n_idx];
    int ${name}_end = ${name}_n_ends[this_${name}_n_idx];
% else:
    Real ${name}_surf_radius = ${name}_n_R[this_${name}_n_idx] * ${name}_surf_r;
    % for d in range(K.spatial_dim):
        Real ${name}_center${dn(d)} = ${name}_n_center[this_${name}_n_idx * ${K.spatial_dim} + ${d}];
    % endfor
% endif
</%def>

<%def name="setup_src_local_memory(type)">
% if type == "pts":
    LOCAL_MEM Real sh_src_tris[${n_workers_per_block}][3][3];
% endif
LOCAL_MEM Real sh_input[${n_workers_per_block}][9];
</%def>

<%def name="start_block_loop()">
    for (int src_block_idx = this_obs_src_start;
         src_block_idx < this_obs_src_end;
         src_block_idx++) 
    {
        const int this_src_n_idx = src_n_idxs[src_block_idx];
</%def>

<%def name="load_tri(name, pt_array_name, tris, center, R)">
for (int c = 0; c < 3; c++) {
    int pt_idx = ${tris}[
        3 * ${name}_tri_idx + positive_mod(c + ${name}_tri_rot_clicks, 3)
    ];
    %for d in range(3):
        ${name}_tri[c][${d}] = ${R} * ${pt_array_name}[pt_idx * 3 + ${d}] + ${center[d]};
    % endfor
}
</%def>

<%def name="tri_chars(name, need_normal, need_surf_curl)">
get_unscaled_normal(${name}_tri, ${name}_unscaled_normal);
${name}_normal_length = magnitude(${name}_unscaled_normal);
${name}_jacobian = ${name}_normal_length;
% if need_normal:
    % for dim in range(3):
    n${name}${dn(dim)} = 
        ${name}_unscaled_normal[${dim}] / ${name}_normal_length;
    % endfor
% endif
% if need_surf_curl:
    ${prim.surf_curl_basis(name)}
% endif
</%def>

<%def name="tri_info(name, pt_array_name, tris, need_normal, need_surf_curl, center, R)">
${load_tri(name, pt_array_name, tris, center, R)}
${tri_chars(name, need_normal, need_surf_curl)}
</%def>

<%def name="obs_loop(obs_type)">
int obs_tri_min = ${"obs_start" if obs_type == "pts" else "0"};
int obs_tri_max = ${"obs_end" if obs_type == "pts" else str(surf_tris.shape[0])};
for (int obs_tri_start = obs_tri_min;
     obs_tri_start < obs_tri_max;
     obs_tri_start += ${n_workers_per_block})
{
    int obs_tri_idx = obs_tri_start + worker_idx;

    ${prim.decl_tri_info("obs", K.needs_obsn, K.surf_curl_obs)}
    Real sum[9];

    if (obs_tri_idx < obs_tri_max) {
        int obs_tri_rot_clicks = 0;
        % if obs_type == "pts":
            ${tri_info("obs", "obs_pts", "obs_tris", 
                K.needs_obsn, K.surf_curl_obs, (0.0, 0.0, 0.0), 1.0)}
        % else:
            ${tri_info("obs", "surf_pts", "surf_tris",
                    K.needs_obsn, K.surf_curl_obs,
                    ["obs_center" + dn(d) for d in range(3)],
                    "obs_surf_radius")}
        % endif

        % for d in range(9):
            sum[${d}] = 0.0;
        % endfor
    }
</%def>

<%def name="start_outer_src_loop(src_type)">
int src_start_idx = ${"src_start" if src_type == "pts" else "0"};
int src_end_idx = ${"src_end" if src_type == "pts" else str(surf_tris.shape[0])};
int input_offset = ${"0" if src_type == "pts" else "this_src_n_idx * " + str(surf_tris.shape[0])};
for (int chunk_start = src_start_idx;
        chunk_start < src_end_idx;
        chunk_start += ${n_workers_per_block}) 
{
    int load_idx = chunk_start + worker_idx;
    if (load_idx < src_end_idx) {
        % if src_type == "pts":
            // Traversing these arrays in a unstrided fashion does not improve performance.
            // Probably due to decent caching on newer nvidia gpus
            % for d1 in range(K.spatial_dim):
            {
                int pt_idx = src_tris[load_idx * 3 + ${d1}];
                % for d2 in range(K.spatial_dim):
                    sh_src_tris[worker_idx][${d1}][${d2}] = src_pts[pt_idx * ${K.spatial_dim} + ${d2}];
                % endfor
            }
            % endfor
        % endif

        for (int d = 0; d < 9; d++) {
            sh_input[worker_idx][d] = in[(input_offset + load_idx) * 9 + d];
        }
    }
    LOCAL_BARRIER;
</%def>

<%def name="src_inner_loop(src_type, check_r2_zero)">
if (obs_tri_idx < obs_tri_max) {
    int chunk_j_max = min(${n_workers_per_block}, src_end_idx - chunk_start);
    for (int chunk_j = 0; chunk_j < chunk_j_max; chunk_j++) {
        <% 
        %>
        int src_tri_rot_clicks = 0; 
        ${prim.decl_tri_info("src", K.needs_srcn, K.surf_curl_src)}
        % if src_type == "pts":
            for (int d1 = 0; d1 < 3; d1++) {
                for (int d2 = 0; d2 < 3; d2++) {
                    src_tri[d1][d2] = sh_src_tris[chunk_j][d1][d2];
                }
            }
            ${tri_chars("src", K.needs_srcn, K.surf_curl_src)}
        % else:
            int src_tri_idx = chunk_start + chunk_j;
            ${tri_info("src", "surf_pts", "surf_tris", K.needs_srcn, K.surf_curl_src,
                    ["src_center" + dn(d) for d in range(3)],
                    "src_surf_radius")}
        % endif

        Real in[9];
        for (int k = 0; k < 9; k++) {
            in[k] = sh_input[chunk_j][k];
        }

        for (int iq = 0; iq < ${quad_wts.shape[0]}; iq++) {
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
}
</%def>

<%def name="sum_to_global(obs_type)">
        if (obs_tri_idx < obs_tri_max) {
            int out_offset = ${0 if obs_type == "pts" else ("this_obs_n_idx * " + str(surf_tris.shape[0]))};
            for (int d = 0; d < 9; d++) {
                out[(out_offset + obs_tri_idx) * 9 + d] += sum[d];
            }
        }
    }
}
</%def>

<%def name="finish_outer_src_loop()">
    LOCAL_BARRIER;
}
</%def>

<%def name="fmm_op(op_name, obs_type, src_type, check_r2_zero)">
${func_def(op_name, obs_type, src_type)}
{
    ${get_block_idx()}
    ${K.constants_code}
    ${setup_block("obs", obs_type)}
    ${setup_src_local_memory(src_type)}

    ${start_block_loop()}
        ${setup_block("src", src_type)}

        ${obs_loop(obs_type)}
            ${start_outer_src_loop(src_type)}
                ${src_inner_loop(src_type, check_r2_zero)}
            ${finish_outer_src_loop()}
        ${sum_to_global(obs_type)}
}
</%def>


${fmm_op("p2s", "surf", "pts", False)}
${fmm_op("s2s", "surf", "surf", False)}
${fmm_op("p2p", "pts", "pts", True)}
${fmm_op("s2p", "pts", "surf", False)}


KERNEL
void c2e_kernel1(GLOBAL_MEM Real* out, GLOBAL_MEM Real* in,
        int n_nodes, int n_rows, GLOBAL_MEM int* node_idxs,
        GLOBAL_MEM Real* UT)
{
    const int idx = get_global_id(0);
    const int row_idx = get_global_id(1);
    if (row_idx >= n_rows || idx >= n_nodes) {
        return;
    }
    const int node_idx = node_idxs[idx];

    Real sum1 = 0.0;
    for (int k = 0; k < n_rows; k++) {
        Real UTv = UT[row_idx * n_rows + k];
        Real xv = in[node_idx * n_rows + k]; 
        sum1 += UTv * xv;
    }
    out[node_idx * n_rows + row_idx] = sum1;
}

KERNEL
void c2e_kernel2(GLOBAL_MEM Real* out, GLOBAL_MEM Real* in,
        int n_nodes, int n_rows, GLOBAL_MEM int* node_idxs,
        GLOBAL_MEM Real* node_R, Real alpha, GLOBAL_MEM Real* V,
        GLOBAL_MEM Real* E)
{
    const int idx = get_global_id(0);
    const int row_idx = get_global_id(1);
    if (row_idx >= n_rows || idx >= n_nodes) {
        return;
    }
    const int node_idx = node_idxs[idx];
    const Real R = node_R[node_idx];

    Real sum2 = 0.0;
    for (int j = 0; j < n_rows; j++) {
        Real Vv = V[row_idx * n_rows + j];
        Real Ev = E[j];
        Real REv = pow(R, ${K.scale_type} + 4) * Ev;
        Real invEv = REv / (REv * REv + alpha * alpha);
        sum2 += Vv * invEv * in[node_idx * n_rows + j];
    }
    out[node_idx * n_rows + row_idx] = sum2;
}
