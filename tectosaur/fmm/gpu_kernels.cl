<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]
from tectosaur.kernels import kernels
%>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define Real ${gpu_float_type}

__constant Real surf[${surf.size}] = {${str(surf.flatten().tolist())[1:-1]}};

<%def name="func_def(op_name, obs_type, src_type)">
__kernel
void ${op_name}_${K.name}(
        __global Real* out, __global Real* in,
        int n_blocks, __global Real* params,
        __global int* obs_n_idxs, __global int* obs_src_starts, __global int* src_n_idxs,
        ${params("obs", obs_type)}, ${params("src", src_type)})
</%def>

<%def name="get_block_idx()">
    const int global_idx = get_global_id(0); 
    const int worker_idx = get_local_id(0);
    const int block_idx = (global_idx - worker_idx) / ${n_workers_per_block};
    const int this_obs_n_idx = obs_n_idxs[block_idx];
    const int this_obs_src_start = obs_src_starts[block_idx];
    const int this_obs_src_end = obs_src_starts[block_idx + 1];
</%def>

<%def name="start_block_loop()">
    for (int src_block_idx = this_obs_src_start;
         src_block_idx < this_obs_src_end;
         src_block_idx++) 
    {
        const int this_src_n_idx = src_n_idxs[src_block_idx];
</%def>

<%def name="params(name, type)">
% if type == "pts":
    __global int* ${name}_n_starts, __global int* ${name}_n_ends,
    __global Real* ${name}_pts, __global Real* ${name}_ns
% else:
    __global Real* ${name}_n_center,
    __global Real* ${name}_n_R, Real ${name}_surf_r
% endif
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
    % if K.needs_srcn:
    __local Real sh_src_ns[${K.spatial_dim} * ${n_workers_per_block}];
    % endif
    __local Real sh_src_pts[${K.spatial_dim} * ${n_workers_per_block}];
    __local Real sh_input[${K.tensor_dim} * ${n_workers_per_block}];
% else:
    __local Real sh_input[${K.tensor_dim} * ${n_workers_per_block}];
% endif
</%def>

<%def name="obs_loop(obs_type)">
int obs_pt_min = ${"obs_start" if obs_type == "pts" else "0"};
int obs_pt_max = ${"obs_end" if obs_type == "pts" else str(surf.shape[0])};
for (int obs_pt_start = obs_pt_min; obs_pt_start < obs_pt_max; obs_pt_start += ${n_workers_per_block})
{
    int obs_pt_idx = obs_pt_start + worker_idx;
    % for d in range(K.spatial_dim):
        Real obs${dn(d)};
        Real nobs${dn(d)};
        Real sum${dn(d)};
    % endfor
    if (obs_pt_idx < obs_pt_max) {
        % if obs_type == "pts":
            % for d in range(K.spatial_dim):
                obs${dn(d)} = obs_pts[obs_pt_idx * ${K.spatial_dim} + ${d}];
                % if K.needs_obsn:
                    nobs${dn(d)} = obs_ns[obs_pt_idx * ${K.spatial_dim} + ${d}];
                % endif
            % endfor
        % else:
            % for d in range(K.spatial_dim):
                nobs${dn(d)} = surf[obs_pt_idx * ${K.spatial_dim} + ${d}];
                obs${dn(d)} = obs_surf_radius * nobs${dn(d)} + obs_center${dn(d)};
            % endfor
        % endif

        % for d in range(K.tensor_dim):
            sum${dn(d)} = 0.0;
        % endfor
    }
</%def>

<%def name="start_outer_src_loop(src_type)">
int src_start_idx = ${"src_start" if src_type == "pts" else "0"};
int src_end_idx = ${"src_end" if src_type == "pts" else str(surf.shape[0])};
int input_offset = ${"0" if src_type == "pts" else "this_src_n_idx * " + str(surf.shape[0])};
for (int chunk_start = src_start_idx;
        chunk_start < src_end_idx;
        chunk_start += ${n_workers_per_block}) 
{
    int load_idx = chunk_start + worker_idx;
    if (load_idx < src_end_idx) {
        % if src_type == "pts":
            // Traversing these arrays in a unstrided fashion does not improve performance.
            // Probably due to decent caching on newer nvidia gpus
            % for d in range(K.spatial_dim):
                sh_src_pts[worker_idx * ${K.spatial_dim} + ${d}] = 
                    src_pts[load_idx * ${K.spatial_dim} + ${d}];
                % if K.needs_srcn:
                    sh_src_ns[worker_idx * ${K.spatial_dim} + ${d}] = 
                        src_ns[load_idx * ${K.spatial_dim} + ${d}];
                % endif
            % endfor
        % endif

        % for d in range(K.tensor_dim):
            sh_input[worker_idx * ${K.tensor_dim} + ${d}] =
                in[(input_offset + load_idx) * ${K.tensor_dim} + ${d}];
        % endfor
    }
    barrier(CLK_LOCAL_MEM_FENCE);
</%def>

<%def name="src_inner_loop(src_type, check_r2_zero)">
if (obs_pt_idx < obs_pt_max) {
    int chunk_j_max = min(${n_workers_per_block}, src_end_idx - chunk_start);
    for (int chunk_j = 0; chunk_j < chunk_j_max; chunk_j++) {
        % if src_type == "pts":
            % for d in range(K.spatial_dim):
                Real src${dn(d)} = sh_src_pts[chunk_j * ${K.spatial_dim} + ${d}];
                % if K.needs_srcn:
                    Real nsrc${dn(d)} = sh_src_ns[chunk_j * ${K.spatial_dim} + ${d}];
                % endif
            % endfor
        % else:
            % for d in range(K.spatial_dim):
                Real nsrc${dn(d)} = surf[(chunk_start + chunk_j) * ${K.spatial_dim} + ${d}];
                Real src${dn(d)} = src_surf_radius * nsrc${dn(d)} + src_center${dn(d)};
            % endfor
        % endif
        % for d in range(K.tensor_dim):
            Real in${dn(d)} = sh_input[chunk_j * ${K.tensor_dim} + ${d}];
        % endfor

        % for d in range(K.spatial_dim):
            Real D${dn(d)} = src${dn(d)} - obs${dn(d)};
        % endfor
        Real r2 = Dx * Dx;
        % for d in range(1, K.spatial_dim):
            r2 += D${dn(d)} * D${dn(d)};
        % endfor

        % if check_r2_zero:
        if (r2 == 0) {
            continue;
        }
        % endif

        % if K.vector_code is None:
            ${K.tensor_code}
            % for d1 in range(K.tensor_dim):
                % for d2 in range(K.tensor_dim):
                    sum${dn(d1)} += K${d1}${d2} * in${dn(d2)};
                % endfor
            % endfor
        % else:
            ${K.vector_code}
        % endif
    }
}
</%def>

<%def name="sum_to_global(obs_type)">
        if (obs_pt_idx < obs_pt_max) {
            int out_offset = ${0 if obs_type == "pts" else ("this_obs_n_idx * " + str(surf.shape[0]))};
            % for d in range(K.tensor_dim):
            out[(out_offset + obs_pt_idx) * ${K.tensor_dim} + ${d}] += sum${dn(d)};
            % endfor
        }
    }
}
</%def>

<%def name="finish_outer_src_loop()">
    barrier(CLK_LOCAL_MEM_FENCE);
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

// This is essentially a weird in-place block sparse matrix-matrix multiply. 
__kernel
void c2e_kernel(__global Real* out, __global Real* in,
        int n_blocks, int n_rows, __global int* node_idx,
        int node_depth, __global Real* ops)
{
    % if type(K.scale_type) is int:
        __global Real* op_start = &ops[0];
        Real scaling = pow(2.0, node_depth * ${-K.scale_type - 4});
    % else:
        __global Real* op_start = &ops[node_depth * n_rows * n_rows];
        Real scaling = 1.0;
    % endif

    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_block_idx = get_global_id(0);
    const int global_col = get_global_id(1);

    __local Real Asub[${n_c2e_block_rows}][${n_c2e_block_rows}];
    __local Real Bsub[${n_c2e_block_rows}][${n_c2e_block_rows}];

    int global_n_idx = -1;
    if (global_block_idx < n_blocks) {
        global_n_idx = node_idx[global_block_idx];
    }

    Real sum = 0.0;
    int t = 0;
    for (;t * ${n_c2e_block_rows} < n_rows; t++) {
        const int tile_row = ${n_c2e_block_rows} * t + local_row;
        const int tile_col = ${n_c2e_block_rows} * t + local_col;
        if (tile_col < n_rows && global_n_idx != -1) {
            Asub[local_row][local_col] = in[global_n_idx * n_rows + tile_col];
        } else {
            Asub[local_row][local_col] = 0.0;
        }
        if (tile_row < n_rows && global_col < n_rows) {
            Bsub[local_row][local_col] = op_start[global_col * n_rows + tile_row];
        } else {
            Bsub[local_row][local_col] = 0.0;
        }


        barrier(CLK_LOCAL_MEM_FENCE);

        if (global_n_idx != -1 && global_col < n_rows) {
            int max_k = min(${n_c2e_block_rows}, n_rows - ${n_c2e_block_rows} * t);
            for (int k = 0; k < max_k; k++) {
                sum += Asub[local_row][k] * Bsub[k][local_col];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (global_n_idx != -1 && global_col < n_rows) {
        out[global_n_idx * n_rows + global_col] += scaling * sum;
    }
}
