<%
from tectosaur.nearfield_op import pairs_func_name

import tectosaur.util.kernel_exprs
kernel_names = ['U', 'T', 'A', 'H']
kernels = tectosaur.util.kernel_exprs.get_kernels('float')
def dn(dim):
    return ['x', 'y', 'z'][dim]

kronecker = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
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
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
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
float ${normal_prefix}${dn(dim)} = 
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
float ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b${basis} * ${tri_name}[${basis}][${dim}];
% endfor
% endfor
</%def>

<%def name="temp_result_idx(d_obs, d_src, b_obs, b_src)">
${b_obs} * 27 + ${d_obs} * 9 + ${b_src} * 3 + ${d_src}
</%def>

<%def name="integrate_pair(k_name, limit)">
    ${tri_info("obs", "n")}
    ${tri_info("src", "l")}

    float result_temp[81];

    for (int iresult = 0; iresult < 81; iresult++) {
        result_temp[iresult] = 0;
    }

    float CsU0 = (3.0-4.0*nu)/(G*16.0*float(M_PI)*(1.0-nu));
    float CsU1 = 1.0/(G*16.0*float(M_PI)*(1.0-nu));
    float CsT0 = 1/(8*float(M_PI)*(1-nu));
    float CsT1 = 1-2*nu;
    float CsH0 = G/(4*float(M_PI)*(1-nu));
    float CsH1 = -1+4*nu;
    float CsH2 = 3*nu;
    float CsH3 = 1-2*nu;

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

        % if limit:
            % for dim in range(3):
                x${dn(dim)} -= eps * sqrt(obs_jacobian) * n${dn(dim)};
            % endfor
        % endif

        float3 D = {yx - xx, yy - xy, yz - xz};
        float r2 = D.x * D.x + D.y * D.y + D.z * D.z;
        if (r2 == 0.0) {
            continue;
        }

        % if k_name is 'U':
            float invr = 1.0 / sqrt(r2);
            float Q1 = CsU0 * invr;
            float Q2 = CsU1 * pow(invr, 3);
            % for k in range(3):
                % for j in range(3):
                    float K${k}${j} = Q2 * D.${dn(k)} * D.${dn(j)};
                    % if k == j:
                        K${k}${j} += Q1 * ${kronecker[k][j]};
                    % endif
                % endfor
            % endfor

        % elif k_name is 'T' or k_name is 'A':
            float invr = 1.0 / sqrt(r2);
            float invr2 = invr * invr;
            float rn = lx * D.x + ly * D.y + lz * D.z;
            float A = CsT0 * invr2;
            float drdn = (D.x * lx + D.y * ly + D.z * lz) * invr;
            % for k in range(3):
                % for j in range(3):
                    float K${k}${j};
                    {
                        float term1 = (CsT1 * ${kronecker[k][j]} +
                            3 * D.${dn(k)} * D.${dn(j)} * invr2);
                        float term2 = CsT1 * (n${dn(j)} * D.${dn(k)} - l${dn(k)} * D.${dn(j)}) * invr;
                        % if k_name is 'T':
                            K${k}${j} = -A * (term1 * drdn - term2);
                        % else:
                            K${k}${j} = A * (term1 * drdn + term2);
                        % endif
                    }
                % endfor
            % endfor

        % elif k_name is 'H':
            /*float invr = 1.0 / sqrt(r2);*/
            /*float3 dr = {D.x * invr, D.y * invr, D.z * invr};*/
            /*float drdn = dr.x * lx + dr.y * ly + dr.z * lz;*/
            /*float drdm = dr.x * nx + dr.y * ny + dr.z * nz;*/
            /*float ndm = lx * nx + ly * ny + lz * nz;*/
            /*% for k in range(3):*/
            /*    % for j in range(3):*/
            /*        float K${k}${j};*/
            /*        {*/
            /*            float line1 = 3 * drdn * (*/
            /*                CsH3 * n${dn(k)} * dr.${dn(j)} */
            /*                + nu * (n${dn(j)} * dr.${dn(k)} + ${kronecker[k][j]} * drdm) */
            /*                - 5 * dr.${dn(k)} * dr.${dn(j)} * drdm*/
            /*            );*/
            /*            float line2 = CsH3 * (*/
            /*                3 * l${dn(j)} * dr.${dn(k)} * drdm */
            /*                + ${kronecker[k][j]} * ndm */
            /*                + l${dn(k)} * n${dn(j)}*/
            /*            );*/
            /*            float line3 = CsH2 * (*/
            /*                l${dn(k)} * dr.${dn(j)} * drdm +*/
            /*                ndm * dr.${dn(k)} * dr.${dn(j)}*/
            /*            );*/
            /*            float line4 = CsH1 * l${dn(j)} * n${dn(k)};*/
            /*            float C = CsH0 * invr * invr * invr;*/
            /*            K${k}${j} = C * (line1 + line2 + line3 + line4);*/
            /*        }*/
            /*    % endfor*/
            /*% endfor*/

            float invr = 1.0 / sqrt(r2);
            float invr2 = invr * invr;
            float invr3 = invr2 * invr;

            float rn = invr*(N.x * D.x + N.y * D.y + N.z * D.z);
            float rm = invr*(M.x * D.x + M.y * D.y + M.z * D.z);
            float mn = M.x * N.x + M.y * N.y + M.z * N.z;

            /*float sn = S.x*N.x + S.y*N.y + S.z*N.z;*/
            /*float sd = invr*(S.x*D.x + S.y*D.y + S.z*D.z);*/
            /*float sm = S.x*M.x + S.y*M.y + S.z*M.z;*/

            float Q = Cs[0] * invr3;
            float A = Q * 3 * rn;
            float B = Q * Cs[1];
            float C = Q * Cs[3];

            float3 MT = {
                Q*Cs[2]*N.x + A*Cs[1]*invr*D.x,
                Q*Cs[2]*N.y + A*Cs[1]*invr*D.y,
                Q*Cs[2]*N.z + A*Cs[1]*invr*D.z
            };
                /*Q*Cs[2]*sn + A*Cs[1]*sd,*/
            float NT = {
                B*M.x + C*D.x*invr*rm,
                B*M.y + C*D.y*invr*rm,
                B*M.z + C*D.z*invr*rm,
            };
            /*float NT = B*sm + C*sd*rm;*/
            float DT = {
                invr*(B*3*N.x*rm + C*invr*D.x*mn + A*(nu*M.x - 5*D.x*invr*rm)),
                invr*(B*3*N.y*rm + C*invr*D.y*mn + A*(nu*M.y - 5*D.y*invr*rm)),
                invr*(B*3*N.z*rm + C*invr*D.z*mn + A*(nu*M.z - 5*D.z*invr*rm));
            };
            /*float DT = invr*(B*3*sn*rm + C*sd*mn + A*(nu*sm - 5*sd*rm));*/
            float ST = A*nu*rm + B*mn;

            K[0][0] = N.x*NT.x + M.x*MT.x + D.x*DT.x + ST;
            K[0][1] = N.x*NT.y + M.x*MT.y + D.x*DT.y;
            K[0][2] = N.x*NT.z + M.x*MT.z + D.x*DT.z;
            K[1][0] = N.y*NT.x + M.y*MT.x + D.y*DT.x;
            K[1][1] = N.y*NT.y + M.y*MT.y + D.y*DT.y + ST;
            K[1][2] = N.y*NT.z + M.y*MT.z + D.y*DT.z;
            K[2][0] = N.z*NT.x + M.z*MT.x + D.z*DT.x;
            K[2][1] = N.z*NT.y + M.z*MT.y + D.z*DT.y;
            K[2][2] = N.z*NT.z + M.z*MT.z + D.z*DT.z + ST;

        % else:

            % for k in range(3):
                % for j in range(3):
                    float K${k}${j} = ${kernels[k_name]['expr'][k][j]};
                % endfor
            % endfor
        % endif

        % for d_obs in range(3):
            % for d_src in range(3):
                {
                    float kernel_val = obs_jacobian * src_jacobian * quadw * K${d_obs}${d_src};
                    % for b_obs in range(3):
                        % for b_src in range(3):
                            {
                                int idx = ${temp_result_idx(d_obs, d_src, b_obs, b_src)};
                                result_temp[idx] += obsb${b_obs} * srcb${b_src} * kernel_val;
                            }
                        % endfor
                    % endfor
                }
            % endfor
        % endfor
    }
</%def>

<%def name="single_pairs(k_name, limit)">
__global__
void ${pairs_func_name(limit, k_name)}(float* result, 
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
        (i * 9 + ${b_obs} * 3 + ${d_obs}) * n_src_tris * 9 +
        (j * 9 + ${b_src} * 3 + ${d_src})
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
            if (r2 == 0.0) {
                continue;
            }
            % if need_srcn:
            float3 N = sh_src_ns[k]; 
            % endif

            float3 S = sh_input[k];

            ${kernel_code}
        }
    }

    if (i < n_obs) {
        result[i] = sum;
    }
}
</%def>

<%
U_const_code = """
Cs[0] = (3.0-4.0*nu)/(G*16.0*float(M_PI)*(1.0-nu));
Cs[1] = 1.0/(G*16.0*float(M_PI)*(1.0-nu));
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
Cs[0] = ${plus_or_minus}(1-2.0*nu)/(8.0*float(M_PI)*(1.0-nu));
Cs[1] = ${minus_or_plus}3.0/(8.0*float(M_PI)*(1.0-nu));
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
Cs[0] = G / (4 * float(M_PI) * (1 - nu));
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
