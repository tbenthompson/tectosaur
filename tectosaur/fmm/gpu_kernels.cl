<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]

from tectosaur.util.build_cfg import gpu_float_type
from tectosaur.kernels import kernels
%>

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define Real ${gpu_float_type}

__constant Real surf[${surf.size}] = {${str(surf.flatten().tolist())[1:-1]}};

// Atomic floating point addition for opencl
// from: https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
float atomic_fadd(volatile __global float *addr, float val)
{
    union{
        unsigned int u32;
        float        f32;
    } next, expected, current;
    current.f32 = *addr;
    do{
        expected.f32 = current.f32;
        next.f32 = expected.f32 + val;
        current.u32 = atomic_cmpxchg(
            (volatile __global unsigned int *)addr, 
            expected.u32, next.u32
        );
    } while( current.u32 != expected.u32 );
    return current.f32;
}

<%def name="get_block_idx()">
    const int global_idx = get_global_id(0); 
    const int worker_idx = get_local_id(0);
    const int block_idx = (global_idx - worker_idx) / ${n_workers_per_block};
</%def>

<%def name="params(name, type)">
% if type == "pts":
    __global int* ${name}_n_start, __global int* ${name}_n_end,
    __global Real* ${name}_pts, __global Real* ${name}_ns
% else:
    __global int* ${name}_n_idx, __global Real* ${name}_n_center,
    __global Real* ${name}_n_width, Real ${name}_surf_r
% endif
</%def>

<%def name="setup_block(name, type)">
% if type == "pts":
    int ${name}_start = ${name}_n_start[block_idx];
    int ${name}_end = ${name}_n_end[block_idx];
    % if name == "src":
        % if K.needs_srcn:
        __local Real sh_src_ns[${K.spatial_dim} * ${n_workers_per_block}];
        % endif
        __local Real sh_src_pts[${K.spatial_dim} * ${n_workers_per_block}];
        __local Real sh_input[${K.tensor_dim} * ${n_workers_per_block}];
    % endif
% else:
    int ${name}_idx = ${name}_n_idx[block_idx];
    Real ${name}_width_mult = ${name}_surf_r * sqrt((Real)${K.spatial_dim});
    Real ${name}_surf_radius = ${name}_n_width[${name}_idx] * ${name}_width_mult;
    % for d in range(K.spatial_dim):
        Real ${name}_center${dn(d)} = ${name}_n_center[${name}_idx * ${K.spatial_dim} + ${d}];
    % endfor
    % if name == "src":
        __local Real sh_input[${K.tensor_dim} * ${n_workers_per_block}];
    % endif
% endif
</%def>

<%def name="src_end_var(src_type)">
${"src_end" if src_type == "pts" else str(surf.shape[0])}
</%def>

<%def name="start_outer_src_loop(src_type)">
<%
src_start_var = "src_start" if src_type == "pts" else "0"
input_offset = "0" if src_type == "pts" else "src_idx * " + str(surf.shape[0])
%>
for (int chunk_start = ${src_start_var};
        chunk_start < ${src_end_var(src_type)};
        chunk_start += ${n_workers_per_block}) 
{
    % if src_type == "pts":
        // Traversing these arrays in a unstrided fashion does not improve performance.
        // Probably due to decent caching on newer nvidia gpus
        % for d in range(K.spatial_dim):
            sh_src_pts[worker_idx * ${K.spatial_dim} + ${d}] = 
                src_pts[(chunk_start + worker_idx) * ${K.spatial_dim} + ${d}];
            % if K.needs_srcn:
                sh_src_ns[worker_idx * ${K.spatial_dim} + ${d}] = 
                    src_ns[(chunk_start + worker_idx) * ${K.spatial_dim} + ${d}];
            % endif
        % endfor
    % endif

    % for d in range(K.tensor_dim):
        sh_input[worker_idx * ${K.tensor_dim} + ${d}] =
            in[(${input_offset} + chunk_start + worker_idx) * ${K.tensor_dim} + ${d}];
    % endfor
    barrier(CLK_LOCAL_MEM_FENCE);
</%def>

<%def name="obs_loop(obs_type)">
<%
obs_start_var = "obs_start" if obs_type == "pts" else "0"
obs_end_var = "obs_end" if obs_type == "pts" else str(surf.shape[0])
%>
for (int i = ${obs_start_var} + worker_idx; i < ${obs_end_var}; i += ${n_workers_per_block}) {

    % if obs_type == "pts":
        % for d in range(K.spatial_dim):
            Real obs${dn(d)} = obs_pts[i * ${K.spatial_dim} + ${d}];
            % if K.needs_obsn:
                Real nobs${dn(d)} = obs_ns[i * ${K.spatial_dim} + ${d}];
            % endif
        % endfor
    % else:
        % for d in range(K.spatial_dim):
            Real nobs${dn(d)} = surf[i * ${K.spatial_dim} + ${d}];
            Real obs${dn(d)} = obs_surf_radius * nobs${dn(d)} + obs_center${dn(d)};
        % endfor
    % endif

    % for d in range(K.tensor_dim):
        Real sum${dn(d)} = 0.0;
    % endfor
</%def>

<%def name="call_kernel(K, check_r_zero)">
    % for d in range(K.spatial_dim):
        Real D${dn(d)} = src${dn(d)} - obs${dn(d)};
    % endfor
    Real r2 = Dx * Dx;
    % for d in range(1, K.spatial_dim):
        r2 += D${dn(d)} * D${dn(d)};
    % endfor

    % if check_r_zero:
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
</%def>

<%def name="src_inner_loop(src_type, check_r2_zero)">
int chunk_j_max = min(${n_workers_per_block}, ${src_end_var(src_type)} - chunk_start);
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

    ${call_kernel(K, check_r2_zero)}
}
</%def>

<%def name="output_sum(K, out_idx)">
    % for d in range(K.tensor_dim):
    {
        __global Real* dest = &out[(${out_idx}) * ${K.tensor_dim} + ${d}];
        atomic_fadd(dest, sum${dn(d)});
    }
    % endfor
</%def>

<%def name="sum_and_finish_loops(obs_type)">
        % if obs_type == "pts":
            ${output_sum(K, "i")}
        % else:
            ${output_sum(K, "obs_idx * " + str(surf.shape[0]) + " + i")}
        % endif
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}
</%def>

<%def name="fmm_op(op_name, obs_type, src_type, check_r2_zero)">
__kernel
void ${op_name}_${K.name}(
        __global Real* out, __global Real* in, int n_blocks, __global Real* params,
        ${params("obs", obs_type)}, ${params("src", src_type)})
{
    ${get_block_idx()}
    ${K.constants_code}
    ${setup_block("obs", obs_type)}
    ${setup_block("src", src_type)}

    ${start_outer_src_loop(src_type)}
    ${obs_loop(obs_type)}
    ${src_inner_loop(src_type, check_r2_zero)}
    ${sum_and_finish_loops(obs_type)}
}
</%def>

${fmm_op("p2s", "surf", "pts", False)}
${fmm_op("s2s", "surf", "surf", False)}
${fmm_op("p2p", "pts", "pts", True)}
${fmm_op("s2p", "pts", "surf", False)}

__kernel
void c2e_kernel(__global Real* out, __global Real* in,
        int n_blocks, int n_rows, __global int* node_idx, __global int* node_depth,
        __global Real* ops)
{
    ${get_block_idx()}

    int n_idx = node_idx[block_idx];
    __global Real* op_start = &ops[node_depth[n_idx] * n_rows * n_rows];

    for (int i = worker_idx; i < n_rows; i += ${n_workers_per_block}) {
        Real sum = 0.0;
        for (int j = 0; j < n_rows; j++) {
            sum += op_start[i * n_rows + j] * in[n_idx * n_rows + j];
        }
        out[n_idx * n_rows + i] += sum;
    }
}

// This is essentially a weird in-place block sparse matrix-matrix multiply. 
__kernel
void d2e_kernel(__global Real* out, __global Real* in,
        int n_blocks, int n_rows, __global int* node_idx, int node_depth,
        __global Real* ops)
{
    __global Real* op_start = &ops[node_depth * n_rows * n_rows];

    const int local_row = get_local_id(0);
    const int local_col = get_local_id(1);
    const int global_block_idx = get_global_id(0);
    const int global_col = get_global_id(1);

    __local Real Asub[${n_d2e_block_rows}][${n_d2e_block_rows}];
    __local Real Bsub[${n_d2e_block_rows}][${n_d2e_block_rows}];

    int global_n_idx = -1;
    if (global_block_idx < n_blocks) {
        global_n_idx = node_idx[global_block_idx];
    }

    Real sum = 0.0;
    for (int t = 0; t * ${n_d2e_block_rows} < n_rows; t++) {

        // Load one tile of A and B into local memory
        const int tile_row = ${n_d2e_block_rows} * t + local_row;
        const int tile_col = ${n_d2e_block_rows} * t + local_col;
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
            for (int k = 0; k < ${n_d2e_block_rows}; k++) {
                if (${n_d2e_block_rows} * t + k >= n_rows) {
                    continue;
                }
                sum += Asub[local_row][k] * Bsub[k][local_col];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_n_idx != -1 && global_col < n_rows) {
        out[global_n_idx * n_rows + global_col] += sum;
    }
}
