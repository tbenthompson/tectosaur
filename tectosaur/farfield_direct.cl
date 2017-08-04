<%
from tectosaur.kernels import elastic_kernels, kernels

def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define Real ${float_type}

<%namespace name="prim" file="integral_primitives.cl"/>

<%def name="farfield_pts(K)">
__kernel
void farfield_pts${K.name}(
    __global Real* result, __global Real* obs_pts, __global Real* obs_ns,
    __global Real* src_pts, __global Real* src_ns, __global Real* input,
    __global Real* params, int n_obs, int n_src)
{
    int i = get_global_id(0);
    int local_id = get_local_id(0);

    % for d in range(K.spatial_dim):
    Real obsp${dn(d)};
    % endfor
    if (i < n_obs) {
        % for d in range(K.spatial_dim):
        obsp${dn(d)} = obs_pts[i * ${K.spatial_dim} + ${d}];
        % endfor
    }

    % if K.needs_obsn:
    % for d in range(K.spatial_dim):
    Real nobs${dn(d)};
    % endfor
    if (i < n_obs) {
        % for d in range(K.spatial_dim):
        nobs${dn(d)} = obs_ns[i * ${K.spatial_dim} + ${d}];
        % endfor
    }
    % endif

    % if K.needs_srcn:
    __local Real sh_src_ns[${K.spatial_dim} * ${block_size}];
    % endif
    __local Real sh_src_pts[${K.spatial_dim} * ${block_size}];
    __local Real sh_input[${K.tensor_dim} * ${block_size}];

    
    ${K.constants_code}

    % for d in range(K.tensor_dim):
    Real sum${dn(d)} = 0.0;
    % endfor

    int j = 0;
    int tile = 0;
    for (; j < n_src; j += ${block_size}, tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int idx = tile * ${block_size} + local_id;
        if (idx < n_src) {
            for (int k = 0; k < ${K.spatial_dim}; k++) {
                % if K.needs_srcn:
                sh_src_ns[local_id * ${K.spatial_dim} + k] = src_ns[idx * ${K.spatial_dim} + k];
                % endif
                sh_src_pts[local_id * ${K.spatial_dim} + k] = src_pts[idx * ${K.spatial_dim} + k];
            }
            for (int k = 0; k < ${K.tensor_dim}; k++) {
                sh_input[local_id * ${K.tensor_dim} + k] = input[idx * ${K.tensor_dim} + k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i >= n_obs) {
            continue;
        }
        for (int k = 0; k < ${block_size} && k < n_src - j; k++) {
            % for d in range(K.spatial_dim):
                Real D${dn(d)} = sh_src_pts[k * ${K.spatial_dim} + ${d}] - obsp${dn(d)};
            % endfor

            Real r2 = Dx * Dx;
            % for d in range(1, K.spatial_dim):
                r2 += D${dn(d)} * D${dn(d)};
            % endfor
            if (r2 == 0.0) {
                continue;
            }
            % if K.needs_srcn:
            % for d in range(K.spatial_dim):
            Real nsrc${dn(d)} = sh_src_ns[k * ${K.spatial_dim} + ${d}];
            % endfor
            % endif

            % for d in range(K.tensor_dim):
            Real in${dn(d)} = sh_input[k * ${K.tensor_dim} + ${d}];
            % endfor

            ${prim.call_vector_code(K)}
        }
    }

    if (i < n_obs) {
        % for d in range(K.tensor_dim):
        result[i * ${K.tensor_dim} + ${d}] = sum${dn(d)};
        % endfor
    }
}
</%def>

${prim.geometry_fncs()}

% for name,K in kernels.items():
${farfield_pts(K)}
% endfor
