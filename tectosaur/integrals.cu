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

        ${basis("src")}
        ${pts_from_basis("y", "src", "src_tri")}

        float dx = xx - yx;
        float dy = xy - yy;
        float dz = xz - yz;
        float r2 = dx * dx + dy * dy + dz * dz;
        if (r2 == 0.0) {
            continue;
        }

        % if limit:
        % for dim in range(3):
        x${dim_name(dim)} -= eps * obs_unscaled_normal[${dim}];
        % endfor
        % endif

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

<%def name="farfield_pts(k_name, need_obsn, need_srcn, n_constants, \
    constants_code, kernel_code)">
__global__
void farfield_pts${k_name}(float3* result, float3* obs_pts, float3* obs_ns,
    float3* src_pts, float3* src_ns, float3* input,
    float G, float nu, int n_obs, int n_src)
{
    int i = blockIdx.x * ${block_size} + threadIdx.x;

    float3 obsp;
    if (i < n_obs) {
        obsp = obs_pts[i];
    }

    % if need_obsn:
    float3 M;
    if (i < n_obs) {
        M = obs_ns[i]; 
    }
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
    float3 sum = {0.0, 0.0, 0.0};

    int j = 0;
    int tile = 0;
    for (; j < n_src; j += ${block_size}, tile++) {
        __syncthreads();
        int idx = tile * ${block_size} + threadIdx.x;
        if (idx < n_src) {
            % if need_srcn:
            sh_src_ns[threadIdx.x] = src_ns[idx];
            % endif
            sh_src_pts[threadIdx.x] = src_pts[idx];
            sh_input[threadIdx.x] = input[idx];
        }
        __syncthreads();

        if (i >= n_obs) {
            continue;
        }
        for (int k = 0; k < ${block_size} && k < n_src - j; k++) {
            float3 D = {
                sh_src_pts[k].x-obsp.x,
                sh_src_pts[k].y-obsp.y,
                sh_src_pts[k].z-obsp.z
            };

            float r2 = D.x * D.x + D.y * D.y + D.z * D.z;
            if (r2 > 0.0) {
                % if need_srcn:
                float3 N = sh_src_ns[k]; 
                % endif

                float3 S = sh_input[k];

                ${kernel_code}
            }
        }
    }

    if (i < n_obs) {
        result[i] = sum;
    }
}
</%def>

<%
U_const_code = """
Cs[0] = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
Cs[1] = 1.0/(G*16.0*M_PI*(1.0-nu));
"""

U_code = """
float invr = 1.0 / sqrt(r2);
float Q1 = Cs[0] * invr;
float Q2 = Cs[1] * invr / r2;
float ddi = D.x*S.x + D.y*S.y + D.z*S.z;
sum.x += Q1*S.x + Q2*D.x*ddi;
sum.y += Q1*S.y + Q2*D.y*ddi;
sum.z += Q1*S.z + Q2*D.z*ddi;
"""

TA_const_code = """
Cs[0] = ${plus_or_minus}(1-2.0*nu)/(8.0*M_PI*(1.0-nu));
Cs[1] = ${minus_or_plus}3.0/(8.0*M_PI*(1.0-nu));
"""

TA_code = """
float invr = 1.0 / sqrt(r2);
float invr2 = invr * invr;
float invr3 = invr2 * invr;

float rn = ${n_name}.x * D.x + ${n_name}.y * D.y + ${n_name}.z * D.z;

float A = Cs[0] * invr3;
float C = Cs[1] * invr3 * invr2;

float rnddi = C * rn * (D.x*S.x + D.y*S.y + D.z*S.z);

float nxdy = ${n_name}.x*D.y-${n_name}.y*D.x;
float nzdx = ${n_name}.z*D.x-${n_name}.x*D.z;
float nzdy = ${n_name}.z*D.y-${n_name}.y*D.z;

sum.x += A*(
    - rn * S.x
    ${minus_or_plus} nxdy * S.y
    ${plus_or_minus} nzdx * S.z)
    + D.x*rnddi;
sum.y += A*(
    ${plus_or_minus} nxdy * S.x
    - rn * S.y
    ${plus_or_minus} nzdy * S.z)
    + D.y*rnddi;
sum.z += A*(
    ${minus_or_plus} nzdx * S.x 
    ${minus_or_plus} nzdy * S.y 
    - rn * S.z)
    + D.z*rnddi;
"""

from mako.template import Template
Targs = {'plus_or_minus': '+', 'minus_or_plus': '-', 'n_name': 'N'}
T_const_code = Template(TA_const_code).render(**Targs)
T_code = Template(TA_code).render(**Targs)

Aargs = {'plus_or_minus': '-', 'minus_or_plus': '+', 'n_name': 'M'}
A_const_code = Template(TA_const_code).render(**Aargs)
A_code = Template(TA_code).render(**Aargs)

H_const_code = """
Cs[0] = G / (4 * M_PI * (1 - nu));
Cs[1] = 1 - 2 * nu;
Cs[2] = -1 + 4 * nu;
Cs[3] = 3 * nu;
"""

H_code = """
float invr = 1.0 / sqrt(r2);
float invr2 = invr * invr;
float invr3 = invr2 * invr;

float rn = invr*(N.x * D.x + N.y * D.y + N.z * D.z);
float rm = invr*(M.x * D.x + M.y * D.y + M.z * D.z);
float mn = M.x * N.x + M.y * N.y + M.z * N.z;

float sn = S.x*N.x + S.y*N.y + S.z*N.z;
float sd = invr*(S.x*D.x + S.y*D.y + S.z*D.z);
float sm = S.x*M.x + S.y*M.y + S.z*M.z;

float Q = Cs[0] * invr3;
float A = Q * 3 * rn;
float B = Q * Cs[1];
float C = Q * Cs[3];

float MT = Q*Cs[2]*sn + A*Cs[1]*sd;
float NT = B*sm + C*sd*rm;
float DT = invr*(B*3*sn*rm + C*sd*mn + A*(nu*sm - 5*sd*rm));
float ST = A*nu*rm + B*mn;

sum.x += N.x*NT + M.x*MT + D.x*DT + ST*S.x;
sum.y += N.y*NT + M.y*MT + D.y*DT + ST*S.y;
sum.z += N.z*NT + M.z*MT + D.z*DT + ST*S.z;
"""
%>
${farfield_pts("U", False, False, 2, U_const_code, U_code)}
${farfield_pts("T", False, True, 2, T_const_code, T_code)}
${farfield_pts("A", True, False, 2, A_const_code, A_code)}
${farfield_pts("H", True, True, 4, H_const_code, H_code)}

% for k_name in kernel_names:
${farfield_tris(k_name)}
${single_pairs(k_name, limit = True)}
${single_pairs(k_name, limit = False)}
% endfor
