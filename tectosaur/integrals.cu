<%
import tectosaur.util.kernel_exprs
kernel_names = ['U', 'T', 'A', 'H']
kernels = tectosaur.util.kernel_exprs.get_kernels()
def dim_name(dim):
    return ['x', 'y', 'z'][dim]
%>

#include <stdio.h>

__device__
void cross(float x[3], float y[3], float out[3]) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

__device__
void sub(float x[3], float y[3], float out[3]) {
    out[0] = x[0] - y[0]; out[1] = x[1] - y[1]; out[2] = x[2] - y[2];
}

__device__
void get_unscaled_normal(float tri[3][3], float out[3]) {
    float s20[3];
    float s21[3];
    sub(tri[2], tri[0], s20);
    sub(tri[2], tri[1], s21);
    cross(s20, s21, out);
}

__device__
float magnitude(float v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

<%def name="get_triangle(name, tris, index)">
float ${name}[3][3];
for (int c = 0; c < 3; c++) {
    for (int d = 0; d < 3; d++) {
        ${name}[c][d] = pts[3 * ${tris}[3 * ${index} + c] + d];
    }
}
</%def>

<%def name="tri_info(prefix,normal_prefix)">
float ${prefix}_unscaled_normal[3];
get_unscaled_normal(${prefix}_tri, ${prefix}_unscaled_normal);
float ${prefix}_normal_length = magnitude(${prefix}_unscaled_normal);
float ${prefix}_jacobian = ${prefix}_normal_length;
% for dim in range(3):
float ${normal_prefix}${dim_name(dim)} = 
    ${prefix}_unscaled_normal[${dim}] / ${prefix}_normal_length;
% endfor
</%def>

<%def name="basis(prefix)">
float ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
float ${prefix}b1 = ${prefix}xhat;
float ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name)">
% for dim in range(3):
float ${pt_pfx}${dim_name(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dim_name(dim)} += ${basis_pfx}b${basis} * ${tri_name}[${basis}][${dim}];
% endfor
% endfor
</%def>

<%def name="temp_result_idx(d_obs, d_src, b_obs, b_src)">
${b_obs} * 27 + ${b_src} * 9 + ${d_obs} * 3 + ${d_src}
</%def>

<%def name="integrate_pair(k_name, limit)">
    ${tri_info("obs", "n")}
    ${tri_info("src", "l")}

    float result_temp[81];

    for (int iresult = 0; iresult < 81; iresult++) {
        result_temp[iresult] = 0;
    }

    for (int iq = 0; iq < n_quad_pts; iq++) {
        <% 
        qd = 4
        if limit:
            qd = 5
        %>
        float obsxhat = quad_pts[iq * ${qd} + 0];
        float obsyhat = quad_pts[iq * ${qd} + 1];
        float srcxhat = quad_pts[iq * ${qd} + 2];
        float srcyhat = quad_pts[iq * ${qd} + 3];
        % if limit:
        float eps = quad_pts[iq * ${qd} + 4];
        % endif
        float quadw = quad_wts[iq];

        ${basis("obs")}
        ${pts_from_basis("x", "obs", "obs_tri")}

        % if limit:
        % for dim in range(3):
        x${dim_name(dim)} -= eps * obs_unscaled_normal[${dim}];
        % endfor
        % endif

        ${basis("src")}
        ${pts_from_basis("y", "src", "src_tri")}

        % for d_obs in range(3):
        % for d_src in range(3):
        {
            float kernel_val = obs_jacobian * src_jacobian * quadw * 
                ${kernels[k_name]['expr'][d_obs][d_src]};
            % for b_obs in range(3):
            % for b_src in range(3):
            result_temp[${temp_result_idx(d_obs, d_src, b_obs, b_src)}] += 
                obsb${b_obs} * srcb${b_src} * kernel_val;
            % endfor
            % endfor
        }
        % endfor
        % endfor
    }
</%def>

<%def name="single_pairs(k_name, limit)">
<%
limit_label = 'N'
if limit:
    limit_label = 'S'
%>
__global__
void single_pairs${limit_label}${k_name}(float* result, 
    int n_quad_pts, float* quad_pts, float* quad_wts,
    float* pts, int* obs_tris, int* src_tris, float G, float nu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    ${get_triangle("obs_tri", "obs_tris", "i")}
    ${get_triangle("src_tri", "src_tris", "i")}
    ${integrate_pair(k_name, limit)}
    
    for (int iresult = 0; iresult < 81; iresult++) {
        result[i * 81 + iresult] = result_temp[iresult];
    }
}
</%def>

<%def name="farfield_tris(k_name)">
__global__
void farfield_tris${k_name}(float* result, int n_quad_pts, float* quad_pts,
    float* quad_wts, float* pts, int n_obs_tris, int* obs_tris, 
    int n_src_tris, int* src_tris, float G, float nu)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    ${get_triangle("obs_tri", "obs_tris", "i")}
    ${get_triangle("src_tri", "src_tris", "j")}
    ${integrate_pair(k_name, limit = False)}

    % for d_obs in range(3):
    % for d_src in range(3):
    % for b_obs in range(3):
    % for b_src in range(3):
    result[
        (i * 9 + ${d_obs} * 3 + ${b_obs}) * n_src_tris * 9 +
        (j * 9 + ${d_src} * 3 + ${b_src})
        ] = result_temp[${temp_result_idx(d_obs, d_src, b_obs, b_src)}];
    % endfor
    % endfor
    % endfor
    % endfor
}
</%def>

<%def name="farfield_pts(k_name)">
__global__ 
void farfield_pts${k_name}(float* result, float3* obs_pts, float3* obs_ns,
    float3* src_pts, float3* src_ns, float G, float nu, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;

    float xx = obs_pts[i].x;
    float xy = obs_pts[i].y;
    float xz = obs_pts[i].z;

    float nx = obs_ns[i].x;
    float ny = obs_ns[i].y;
    float nz = obs_ns[i].z;

    float yx = src_pts[j].x;
    float yy = src_pts[j].y;
    float yz = src_pts[j].z;

    float lx = src_ns[j].x;
    float ly = src_ns[j].y;
    float lz = src_ns[j].z;

    % for d1 in range(3):
    <%
    max_d_src = 3
    if kernels[k_name]['symmetric']:
        max_d_src = d1 + 1
    %>
    % for d2 in range(max_d_src):
    result[i * n * 9 + j * 9 + ${d1} * 3 + ${d2}] =
        ${kernels[k_name]['expr'][d1][d2]};
    % endfor
    % endfor

    % if kernels[k_name]['symmetric']:
    % for d1 in range(3):
    % for d2 in range(3):
    result[i * n * 9 + j * 9 + ${d2} * 3 + ${d1}] = 
        result[i * n * 9 + j * 9 + ${d1} * 3 + ${d2}];
    % endfor
    % endfor
    % endif
}
</%def>

% for k_name in kernel_names:
${farfield_pts(k_name)}
${farfield_tris(k_name)}
${single_pairs(k_name, limit = True)}
${single_pairs(k_name, limit = False)}
% endfor
