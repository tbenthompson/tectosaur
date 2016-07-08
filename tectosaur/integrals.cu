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

__global__
void farfield_ptsU(float3* result, float3* obs_pts, float3* obs_ns,
    float3* src_pts, float3* src_ns, float3* input, float G, float nu, int n)
{
    const int i = blockIdx.x * ${block_size} + threadIdx.x;

    const float3 obsp = obs_pts[i];
    const float3 obsn = obs_ns[i];

    __shared__ float3 sh_src_pts[${block_size}];
    __shared__ float3 sh_input[${block_size}];
    __shared__ float Cs[2];
    if (threadIdx.x == 0) {
        Cs[0] = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
        Cs[1] = 1.0/(G*16.0*M_PI*(1.0-nu));
    }
    float3 result_sum = {0.0, 0.0, 0.0};

    int j, tile;
    for (j = 0, tile = 0; j < n; j += ${block_size}, tile++) {
        int idx = tile * ${block_size} + threadIdx.x;
        sh_src_pts[threadIdx.x] = src_pts[idx];
        sh_input[threadIdx.x] = input[idx];
        __syncthreads();

        for (int k = 0; k < ${block_size}; k++) {
            float dx = obsp.x - sh_src_pts[k].x;
            float dy = obsp.y - sh_src_pts[k].y;
            float dz = obsp.z - sh_src_pts[k].z;
            float r2 = dx * dx;
            r2 += dy * dy;
            r2 += dz * dz;
            float invr = 1.0 / sqrt(r2);
            float Q1 = Cs[0] * invr;
            float Q2 = Cs[1] * invr / r2;
            float ddi = dx * sh_input[k].x +
                dy * sh_input[k].y + 
                dz * sh_input[k].z;
            result_sum.x += Q1 * sh_input[k].x + Q2 * dx * ddi;
            result_sum.y += Q1 * sh_input[k].y + Q2 * dy * ddi;
            result_sum.z += Q1 * sh_input[k].z + Q2 * dz * ddi;
        }
        __syncthreads();
    }

    result[i] = result_sum;
}

<%def name="farfield_pts_traction(k_name)">
<%
if k_name == 'T':
    normal_name = 'sh_src_ns[k]'
    minus_or_plus = '-'
    plus_or_minus = '+'
else:
    normal_name = 'obsn'
    minus_or_plus = '+'
    plus_or_minus = '-'
%>
__global__
void farfield_pts${k_name}(float3* result, float3* obs_pts, float3* obs_ns,
    float3* src_pts, float3* src_ns, float3* input, float G, float nu, int n)
{
    const int i = blockIdx.x * ${block_size} + threadIdx.x;

    const float3 obsp = obs_pts[i];
    % if k_name == 'A':
    const float3 obsn = obs_ns[i];
    % endif

    __shared__ float3 sh_src_pts[${block_size}];
    % if k_name == 'T':
        __shared__ float3 sh_src_ns[${block_size}];
    % endif
    __shared__ float3 sh_input[${block_size}];
    __shared__ float Cs[2];
    if (threadIdx.x == 0) {
        Cs[0] = ${plus_or_minus}(1-2.0*nu)/(8.0*M_PI*(1.0-nu));
        Cs[1] = ${minus_or_plus}3.0/(8.0*M_PI*(1.0-nu));
    }
    float3 result_sum = {0.0, 0.0, 0.0};

    int j, tile;
    for (j = 0, tile = 0; j < n; j += ${block_size}, tile++) {
        int idx = tile * ${block_size} + threadIdx.x;
        sh_src_pts[threadIdx.x] = src_pts[idx];
        % if k_name == 'T':
            sh_src_ns[threadIdx.x] = src_ns[idx];
        % endif
        sh_input[threadIdx.x] = input[idx];
        __syncthreads();

        for (int k = 0; k < ${block_size}; k++) {
            float nx = ${normal_name}.x; 
            float ny = ${normal_name}.y; 
            float nz = ${normal_name}.z; 

            float dx = sh_src_pts[k].x - obsp.x;
            float dy = sh_src_pts[k].y - obsp.y;
            float dz = sh_src_pts[k].z - obsp.z;

            float r2 = dx * dx + dy * dy + dz * dz;
            float invr = 1.0 / sqrt(r2);
            float invr2 = invr * invr;
            float invr3 = invr2 * invr;

            float rn = nx * dx + ny * dy + nz * dz;

            float A = Cs[0] * invr3;
            float C = Cs[1] * invr3 * invr2;

            float rnddi = C * rn * (dx * sh_input[k].x +
                dy * sh_input[k].y + 
                dz * sh_input[k].z);

            float nxdy = nx*dy-ny*dx;
            float nzdx = nz*dx-nx*dz;
            float nzdy = nz*dy-ny*dz;

            result_sum.x += A*(
                - rn * sh_input[k].x
                ${minus_or_plus} nxdy * sh_input[k].y
                ${plus_or_minus} nzdx * sh_input[k].z)
                + dx*rnddi;
            result_sum.y += A*(
                ${plus_or_minus} nxdy * sh_input[k].x
                - rn * sh_input[k].y
                ${plus_or_minus} nzdy * sh_input[k].z)
                + dy*rnddi;
            result_sum.z += A*(
                ${minus_or_plus} nzdx * sh_input[k].x 
                ${minus_or_plus} nzdy * sh_input[k].y 
                - rn * sh_input[k].z)
                + dz*rnddi;
        }
        __syncthreads();
    }

    result[i] = result_sum;
}
</%def>

${farfield_pts_traction("T")}
${farfield_pts_traction("A")}

<%def name="farfield_pts(k_name, need_obsn, need_srcn, n_constants, \
    constants_code, kernel_code)">
__global__
void farfield_pts${k_name}(float3* result, float3* obs_pts, float3* obs_ns,
    float3* src_pts, float3* src_ns, float3* input, float G, float nu, int n)
{
    const int i = blockIdx.x * ${block_size} + threadIdx.x;
    const float3 obsp = obs_pts[i];

    % if need_obsn:
    float mx = obs_ns[i].x; 
    float my = obs_ns[i].y; 
    float mz = obs_ns[i].z; 
    % endif

    % if need_srcn:
    __shared__ float3 sh_src_ns[${block_size}];
    % endif
    __shared__ float3 sh_src_pts[${block_size}];
    __shared__ float3 sh_input[${block_size}];

    __shared__ float Cs[${n_constants}];
    if (threadIdx.x == 0) {
        ${constants_code}
    }
    float3 result_sum = {0.0, 0.0, 0.0};

    int j, tile;
    for (j = 0, tile = 0; j < n; j += ${block_size}, tile++) {
        int idx = tile * ${block_size} + threadIdx.x;
        % if need_srcn:
        sh_src_pts[threadIdx.x] = src_pts[idx];
        % endif
        sh_src_ns[threadIdx.x] = src_ns[idx];
        sh_input[threadIdx.x] = input[idx];
        __syncthreads();

        for (int k = 0; k < ${block_size}; k++) {
            float dx = sh_src_pts[k].x - obsp.x;
            float dy = sh_src_pts[k].y - obsp.y;
            float dz = sh_src_pts[k].z - obsp.z;

            float nx = sh_src_ns[k].x; 
            float ny = sh_src_ns[k].y; 
            float nz = sh_src_ns[k].z; 
            ${kernel_code}
        }
        __syncthreads();
    }

    result[i] = result_sum;
}
</%def>

<%
H_const_code = """
Cs[0] = G / (4 * M_PI * (1 - nu));
Cs[1] = 1 - 2 * nu;
Cs[2] = -1 + 4 * nu;
Cs[3] = 3 * nu;
"""

H_code = """
float r2 = dx * dx + dy * dy + dz * dz;

float invr = 1.0 / sqrt(r2);
float invr2 = invr * invr;
float invr3 = invr2 * invr;

float rn = invr*(nx * dx + ny * dy + nz * dz);
float rm = invr*(mx * dx + my * dy + mz * dz);
float mn = mx * nx + my * ny + mz * nz;

float3 S = sh_input[k];
float sn = S.x*nx + S.y*ny + S.z*nz;
float sd = invr*(S.x*dx + S.y*dy + S.z*dz);
float sm = S.x*mx + S.y*my + S.z*mz;

float Q = Cs[0] * invr3;
float A = Q * 3 * rn;
float B = Q * Cs[1];
float C = Q * Cs[3];
float D = Q * Cs[2];

float MT = D*sn + A*Cs[1]*sd;
float NT = B*sm + C*sd*rm;
float DT = invr*(B*3*sn*rm + C*sd*mn + A*(nu*sm - 5*sd*rm));
float ST = A*nu*rm + B*mn;

result_sum.x += nx*NT + mx*MT + dx*DT + ST*S.x;
result_sum.y += ny*NT + my*MT + dy*DT + ST*S.y;
result_sum.z += nz*NT + mz*MT + dz*DT + ST*S.z;
"""
%>
${farfield_pts("H", True, True, 4, H_const_code, H_code)}

% for k_name in kernel_names:
${farfield_tris(k_name)}
${single_pairs(k_name, limit = True)}
${single_pairs(k_name, limit = False)}
% endfor
