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
void farfield_pts${K.name}${K.spatial_dim}(
    __global Real* result, __global Real* obs_pts, __global Real* obs_ns,
    __global Real* src_pts, __global Real* src_ns, __global Real* input,
    __global Real* params, int n_obs, int n_src)
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

    % if K.needs_obsn:
    % for d in range(3):
    Real nobs${dn(d)};
    % endfor
    if (i < n_obs) {
        % for d in range(3):
        nobs${dn(d)} = obs_ns[i * 3 + ${d}];
        % endfor
    }
    % endif

    % if K.needs_srcn:
    __local Real sh_src_ns[3 * ${block_size}];
    % endif
    __local Real sh_src_pts[3 * ${block_size}];
    __local Real sh_input[3 * ${block_size}];

    
    ${K.constants_code}

    Real kahansumx = 0.0;
    Real kahansumy = 0.0;
    Real kahansumz = 0.0;
    Real kahantempx = 0.0;
    Real kahantempy = 0.0;
    Real kahantempz = 0.0;

    int j = 0;
    int tile = 0;
    for (; j < n_src; j += ${block_size}, tile++) {
        barrier(CLK_LOCAL_MEM_FENCE);
        int idx = tile * ${block_size} + local_id;
        if (idx < n_src) {
            for (int k = 0; k < 3; k++) {
                % if K.needs_srcn:
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
            % if K.needs_srcn:
            % for d in range(3):
            Real nsrc${dn(d)} = sh_src_ns[k * 3 + ${d}];
            % endfor
            % endif

            % for d in range(3):
            Real in${dn(d)} = sh_input[k * 3 + ${d}];
            % endfor

            Real sumx = 0.0;
            Real sumy = 0.0;
            Real sumz = 0.0;
            ${prim.call_vector_code(K)}
            % for d in range(3):
                { //TODO: Is kahan summation necessary here?
                    Real y = sum${dn(d)} - kahantemp${dn(d)};
                    Real t = kahansum${dn(d)} + y;
                    kahantemp${dn(d)} = (t - kahansum${dn(d)}) - y;
                    kahansum${dn(d)} = t;
                }
            % endfor
        }
    }

    if (i < n_obs) {
        % for d in range(3):
        result[i * 3 + ${d}] = kahansum${dn(d)};
        % endfor
    }
}
</%def>

${prim.geometry_fncs()}

% for K in kernels:
${farfield_pts(K)}
% endfor
