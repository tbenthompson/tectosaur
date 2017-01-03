<%
from tectosaur.nearfield_op import pairs_func_name

import tectosaur.util.kernel_exprs
kernel_names = ['U', 'T', 'A', 'H']
kernels = tectosaur.util.kernel_exprs.get_kernels('float')
def dn(dim):
    return ['x', 'y', 'z'][dim]

kronecker = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
%>

void vec_cross(float x[3], float y[3], float out[3]) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

void sub(float x[3], float y[3], float out[3]) {
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
}

void get_unscaled_normal(float tri[3][3], float out[3]) {
    float s20[3];
    float s21[3];
    sub(tri[2], tri[0], s20);
    sub(tri[2], tri[1], s21);
    vec_cross(s20, s21, out);
}

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

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name,ndims)">
% for dim in range(ndims):
float ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b${basis} * ${tri_name(basis,dim)};
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

    float CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
    float CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
    float CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));
    float CsT1 = 3.0/(8.0*M_PI*(1.0-nu));
    float CsT2 = 1.0/(8*M_PI*(1-nu));
    float CsT3 = 1-2*nu;
    float CsH0 = G/(4*M_PI*(1-nu));
    float CsH1 = 1-2*nu;
    float CsH2 = -1+4*nu;
    float CsH3 = 3*nu;
    
    for (int iq = 0; iq < n_quad_pts; iq++) {
        <% 
        qd = 4
        if limit:
            qd = 5
        %>
        float obs_geom_xhat = quad_pts[iq * ${qd} + 0];
        float obs_geom_yhat = quad_pts[iq * ${qd} + 1];
        float src_geom_xhat = quad_pts[iq * ${qd} + 2];
        float src_geom_yhat = quad_pts[iq * ${qd} + 3];
        % if limit:
            float eps = quad_pts[iq * ${qd} + 4];
        % endif
        float quadw = quad_wts[iq];

        % for which, ptname in [("obs", "x"), ("src", "y")]:
            ${basis(which + "_geom_")}
            ${pts_from_basis(
                ptname, which + "_geom_",
                lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
            )}
            % for d in range(3):
                float ${which}b${d} = ${which}_geom_b${d};
            % endfor
        % endfor

        % if limit:
            % for dim in range(3):
                x${dn(dim)} -= eps * sqrt(obs_jacobian) * n${dn(dim)};
            % endfor
        % endif

        float Dx = yx - xx;
        float Dy = yy - xy; 
        float Dz = yz - xz;
        float r2 = Dx * Dx + Dy * Dy + Dz * Dz;
        if (r2 == 0.0) {
            continue;
        }

        % if k_name is 'U':
            float invr = 1.0 / sqrt(r2);
            float Q1 = CsU0 * invr;
            float Q2 = CsU1 * invr / r2;
            float K00 = Q2*Dx*Dx + Q1;
            float K01 = Q2*Dx*Dy;
            float K02 = Q2*Dx*Dz;
            float K10 = Q2*Dy*Dx;
            float K11 = Q2*Dy*Dy + Q1;
            float K12 = Q2*Dy*Dz;
            float K20 = Q2*Dz*Dx;
            float K21 = Q2*Dz*Dy;
            float K22 = Q2*Dz*Dz + Q1;
        % elif k_name is 'T' or k_name is 'A':
            <%
                minus_or_plus = '-' if k_name is 'T' else '+'
                plus_or_minus = '+' if k_name is 'T' else '-'
                n_name = 'l' if k_name is 'T' else 'n'
            %>
            float invr = 1.0 / sqrt(r2);
            float invr2 = invr * invr;
            float invr3 = invr2 * invr;

            float rn = ${n_name}x * Dx + ${n_name}y * Dy + ${n_name}z * Dz;

            float A = ${plus_or_minus}CsT0 * invr3;
            float C = ${minus_or_plus}CsT1 * invr3 * invr2;

            float nxdy = ${n_name}x*Dy-${n_name}y*Dx;
            float nzdx = ${n_name}z*Dx-${n_name}x*Dz;
            float nzdy = ${n_name}z*Dy-${n_name}y*Dz;

            float K00 = A * -rn                  + C*Dx*rn*Dx;
            float K01 = A * ${minus_or_plus}nxdy + C*Dx*rn*Dy;
            float K02 = A * ${plus_or_minus}nzdx + C*Dx*rn*Dz;
            float K10 = A * ${plus_or_minus}nxdy + C*Dy*rn*Dx;
            float K11 = A * -rn                  + C*Dy*rn*Dy;
            float K12 = A * ${plus_or_minus}nzdy + C*Dy*rn*Dz;
            float K20 = A * ${minus_or_plus}nzdx + C*Dz*rn*Dx;
            float K21 = A * ${minus_or_plus}nzdy + C*Dz*rn*Dy;
            float K22 = A * -rn                  + C*Dz*rn*Dz;
        % elif k_name is 'H':
            float invr = 1.0 / sqrt(r2);
            float invr2 = invr * invr;
            float invr3 = invr2 * invr;
            float Dorx = invr * Dx;
            float Dory = invr * Dy;
            float Dorz = invr * Dz;

            float rn = lx * Dorx + ly * Dory + lz * Dorz;
            float rm = nx * Dorx + ny * Dory + nz * Dorz;
            float mn = nx * lx + ny * ly + nz * lz;

            float Q = CsH0 * invr3;
            float A = Q * 3 * rn;
            float B = Q * CsH1;
            float C = Q * CsH3;

            float MTx = Q*CsH2*lx + A*CsH1*Dorx;
            float MTy = Q*CsH2*ly + A*CsH1*Dory;
            float MTz = Q*CsH2*lz + A*CsH1*Dorz;

            float NTx = B*nx + C*Dorx*rm;
            float NTy = B*ny + C*Dory*rm;
            float NTz = B*nz + C*Dorz*rm;

            float DTx = B*3*lx*rm + C*Dorx*mn + A*(nu*nx - 5*Dorx*rm);
            float DTy = B*3*ly*rm + C*Dory*mn + A*(nu*ny - 5*Dory*rm);
            float DTz = B*3*lz*rm + C*Dorz*mn + A*(nu*nz - 5*Dorz*rm);

            float ST = A*nu*rm + B*mn;

            float K00 = lx*NTx + nx*MTx + Dorx*DTx + ST;
            float K01 = lx*NTy + nx*MTy + Dorx*DTy;
            float K02 = lx*NTz + nx*MTz + Dorx*DTz;
            float K10 = ly*NTx + ny*MTx + Dory*DTx;
            float K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
            float K12 = ly*NTz + ny*MTz + Dory*DTz;
            float K20 = lz*NTx + nz*MTx + Dorz*DTx;
            float K21 = lz*NTy + nz*MTy + Dorz*DTy;
            float K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;
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
__kernel
void ${pairs_func_name(limit, k_name)}(__global float* result, 
    int n_quad_pts, __global float* quad_pts, __global float* quad_wts,
    __global float* pts, __global int* obs_tris, __global int* src_tris, 
    float G, float nu)
{
    const int i = get_global_id(0);

    ${get_triangle("obs_tri", "obs_tris", "i")}
    ${get_triangle("src_tri", "src_tris", "i")}
    ${integrate_pair(k_name, limit)}
    
    for (int iresult = 0; iresult < 81; iresult++) {
        result[i * 81 + iresult] = result_temp[iresult];
    }
}
</%def>


<%def name="constants()">
const float CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
const float CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
const float CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));
const float CsT1 = 3.0/(8.0*M_PI*(1.0-nu));
const float CsH0 = G/(4*M_PI*(1-nu));
const float CsH1 = 1-2*nu;
const float CsH2 = -1+4*nu;
const float CsH3 = 3*nu;
</%def>


% for k_name in kernel_names:
${single_pairs(k_name, limit = True)}
${single_pairs(k_name, limit = False)}
% endfor
