<%
def dn(dim):
    return ['x', 'y', 'z'][dim]
float_type = 'double'
%>

<%def name="geometry_fncs()">
void vec_cross(${float_type} x[3], ${float_type} y[3], ${float_type} out[3]) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

void sub(${float_type} x[3], ${float_type} y[3], ${float_type} out[3]) {
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
}

void get_unscaled_normal(${float_type} tri[3][3], ${float_type} out[3]) {
    ${float_type} s20[3];
    ${float_type} s21[3];
    sub(tri[2], tri[0], s20);
    sub(tri[2], tri[1], s21);
    vec_cross(s20, s21, out);
}

${float_type} magnitude(${float_type} v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
</%def>

<%def name="get_triangle(name, tris, index)">
${float_type} ${name}[3][3];
for (int c = 0; c < 3; c++) {
    for (int d = 0; d < 3; d++) {
        ${name}[c][d] = pts[3 * ${tris}[3 * ${index} + c] + d];
    }
}
</%def>

<%def name="tri_info(prefix,normal_prefix)">
${float_type} ${prefix}_unscaled_normal[3];
get_unscaled_normal(${prefix}_tri, ${prefix}_unscaled_normal);
${float_type} ${prefix}_normal_length = magnitude(${prefix}_unscaled_normal);
${float_type} ${prefix}_jacobian = ${prefix}_normal_length;
% for dim in range(3):
    ${float_type} ${normal_prefix}${dn(dim)} = 
        ${prefix}_unscaled_normal[${dim}] / ${prefix}_normal_length;
% endfor
</%def>

<%def name="basis(prefix)">
${float_type} ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
${float_type} ${prefix}b1 = ${prefix}xhat;
${float_type} ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name,ndims)">
% for dim in range(ndims):
${float_type} ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b${basis} * ${tri_name(basis,dim)};
% endfor
% endfor
</%def>

<%def name="temp_result_idx(d_obs, d_src, b_obs, b_src)">
${b_obs} * 27 + ${d_obs} * 9 + ${b_src} * 3 + ${d_src}
</%def>

<%def name="constants()">
const ${float_type} CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
const ${float_type} CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
const ${float_type} CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));
const ${float_type} CsT1 = 3.0/(8.0*M_PI*(1.0-nu));
const ${float_type} CsH0 = G/(4*M_PI*(1-nu));
const ${float_type} CsH1 = 1-2*nu;
const ${float_type} CsH2 = -1+4*nu;
const ${float_type} CsH3 = 3*nu;
</%def>

<%def name="tensor_kernels(k_name)">
% if k_name is 'U':
    ${float_type} invr = 1.0 / sqrt(r2);
    ${float_type} Q1 = CsU0 * invr;
    ${float_type} Q2 = CsU1 * invr / r2;
    ${float_type} K00 = Q2*Dx*Dx + Q1;
    ${float_type} K01 = Q2*Dx*Dy;
    ${float_type} K02 = Q2*Dx*Dz;
    ${float_type} K10 = Q2*Dy*Dx;
    ${float_type} K11 = Q2*Dy*Dy + Q1;
    ${float_type} K12 = Q2*Dy*Dz;
    ${float_type} K20 = Q2*Dz*Dx;
    ${float_type} K21 = Q2*Dz*Dy;
    ${float_type} K22 = Q2*Dz*Dz + Q1;
% elif k_name is 'T' or k_name is 'A':
    <%
        minus_or_plus = '-' if k_name is 'T' else '+'
        plus_or_minus = '+' if k_name is 'T' else '-'
        n_name = 'l' if k_name is 'T' else 'n'
    %>
    ${float_type} invr = 1.0 / sqrt(r2);
    ${float_type} invr2 = invr * invr;
    ${float_type} invr3 = invr2 * invr;

    ${float_type} rn = ${n_name}x * Dx + ${n_name}y * Dy + ${n_name}z * Dz;

    ${float_type} A = ${plus_or_minus}CsT0 * invr3;
    ${float_type} C = ${minus_or_plus}CsT1 * invr3 * invr2;

    ${float_type} nxdy = ${n_name}x*Dy-${n_name}y*Dx;
    ${float_type} nzdx = ${n_name}z*Dx-${n_name}x*Dz;
    ${float_type} nzdy = ${n_name}z*Dy-${n_name}y*Dz;

    ${float_type} K00 = A * -rn                  + C*Dx*rn*Dx;
    ${float_type} K01 = A * ${minus_or_plus}nxdy + C*Dx*rn*Dy;
    ${float_type} K02 = A * ${plus_or_minus}nzdx + C*Dx*rn*Dz;
    ${float_type} K10 = A * ${plus_or_minus}nxdy + C*Dy*rn*Dx;
    ${float_type} K11 = A * -rn                  + C*Dy*rn*Dy;
    ${float_type} K12 = A * ${plus_or_minus}nzdy + C*Dy*rn*Dz;
    ${float_type} K20 = A * ${minus_or_plus}nzdx + C*Dz*rn*Dx;
    ${float_type} K21 = A * ${minus_or_plus}nzdy + C*Dz*rn*Dy;
    ${float_type} K22 = A * -rn                  + C*Dz*rn*Dz;
% elif k_name is 'H':
    ${float_type} invr = 1.0 / sqrt(r2);
    ${float_type} invr2 = invr * invr;
    ${float_type} invr3 = invr2 * invr;
    ${float_type} Dorx = invr * Dx;
    ${float_type} Dory = invr * Dy;
    ${float_type} Dorz = invr * Dz;

    ${float_type} rn = lx * Dorx + ly * Dory + lz * Dorz;
    ${float_type} rm = nx * Dorx + ny * Dory + nz * Dorz;
    ${float_type} mn = nx * lx + ny * ly + nz * lz;

    ${float_type} Q = CsH0 * invr3;
    ${float_type} A = Q * 3 * rn;
    ${float_type} B = Q * CsH1;
    ${float_type} C = Q * CsH3;

    ${float_type} MTx = Q*CsH2*lx + A*CsH1*Dorx;
    ${float_type} MTy = Q*CsH2*ly + A*CsH1*Dory;
    ${float_type} MTz = Q*CsH2*lz + A*CsH1*Dorz;

    ${float_type} NTx = B*nx + C*Dorx*rm;
    ${float_type} NTy = B*ny + C*Dory*rm;
    ${float_type} NTz = B*nz + C*Dorz*rm;

    ${float_type} DTx = B*3*lx*rm + C*Dorx*mn + A*(nu*nx - 5*Dorx*rm);
    ${float_type} DTy = B*3*ly*rm + C*Dory*mn + A*(nu*ny - 5*Dory*rm);
    ${float_type} DTz = B*3*lz*rm + C*Dorz*mn + A*(nu*nz - 5*Dorz*rm);

    ${float_type} ST = A*nu*rm + B*mn;

    ${float_type} K00 = lx*NTx + nx*MTx + Dorx*DTx + ST;
    ${float_type} K01 = lx*NTy + nx*MTy + Dorx*DTy;
    ${float_type} K02 = lx*NTz + nx*MTz + Dorx*DTz;
    ${float_type} K10 = ly*NTx + ny*MTx + Dory*DTx;
    ${float_type} K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
    ${float_type} K12 = ly*NTz + ny*MTz + Dory*DTz;
    ${float_type} K20 = lz*NTx + nz*MTx + Dorz*DTx;
    ${float_type} K21 = lz*NTy + nz*MTy + Dorz*DTy;
    ${float_type} K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;
% endif
</%def>

<%def name="vector_kernels(k_name)">
% if k_name is 'U':
    ${float_type} invr = 1.0 / sqrt(r2);
    ${float_type} Q1 = CsU0 * invr;
    ${float_type} Q2 = CsU1 * invr / r2;
    ${float_type} ddi = D.x*S.x + D.y*S.y + D.z*S.z;
    sum.x += Q1*S.x + Q2*D.x*ddi;
    sum.y += Q1*S.y + Q2*D.y*ddi;
    sum.z += Q1*S.z + Q2*D.z*ddi;
% elif k_name is 'T' or k_name is 'A':
    <%
        minus_or_plus = '-' if k_name is 'T' else '+'
        plus_or_minus = '+' if k_name is 'T' else '-'
        n_name = 'l' if k_name is 'T' else 'n'
    %>
    ${float_type} invr = 1.0 / sqrt(r2);
    ${float_type} invr2 = invr * invr;
    ${float_type} invr3 = invr2 * invr;

    ${float_type} rn = ${n_name}.x * D.x + ${n_name}.y * D.y + ${n_name}.z * D.z;

    ${float_type} A = ${plus_or_minus}CsT0 * invr3;
    ${float_type} C = ${minus_or_plus}CsT1 * invr3 * invr2;

    ${float_type} rnddi = C * rn * (D.x*S.x + D.y*S.y + D.z*S.z);

    ${float_type} nxdy = ${n_name}.x*D.y-${n_name}.y*D.x;
    ${float_type} nzdx = ${n_name}.z*D.x-${n_name}.x*D.z;
    ${float_type} nzdy = ${n_name}.z*D.y-${n_name}.y*D.z;

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
% elif k_name is 'H':
    ${float_type} invr = 1.0 / sqrt(r2);
    ${float_type} invr2 = invr * invr;
    ${float_type} invr3 = invr2 * invr;

    ${float_type} rn = invr*(N.x * D.x + N.y * D.y + N.z * D.z);
    ${float_type} rm = invr*(M.x * D.x + M.y * D.y + M.z * D.z);
    ${float_type} mn = M.x * N.x + M.y * N.y + M.z * N.z;

    ${float_type} sn = S.x*N.x + S.y*N.y + S.z*N.z;
    ${float_type} sd = invr*(S.x*D.x + S.y*D.y + S.z*D.z);
    ${float_type} sm = S.x*M.x + S.y*M.y + S.z*M.z;

    ${float_type} Q = CsH0 * invr3;
    ${float_type} A = Q * 3 * rn;
    ${float_type} B = Q * CsH1;
    ${float_type} C = Q * CsH3;

    ${float_type} MT = Q*CsH2*sn + A*CsH1*sd;
    ${float_type} NT = B*sm + C*sd*rm;
    ${float_type} DT = invr*(B*3*sn*rm + C*sd*mn + A*(nu*sm - 5*sd*rm));
    ${float_type} ST = A*nu*rm + B*mn;

    sum.x += N.x*NT + M.x*MT + D.x*DT + ST*S.x;
    sum.y += N.y*NT + M.y*MT + D.y*DT + ST*S.y;
    sum.z += N.z*NT + M.z*MT + D.z*DT + ST*S.z;
%endif
</%def>



<%def name="co_theta_low(chunk)">\
% if chunk == 0:
M_PI - atan2(1 - obsyhat, obsxhat);
% elif chunk == 1:
M_PI + atan2(obsyhat, obsxhat);
% elif chunk == 2:
-atan2(obsyhat, 1 - obsxhat);
% endif
</%def>

<%def name="co_theta_high(chunk)">\
% if chunk == 0:
M_PI + atan2(obsyhat, obsxhat);
% elif chunk == 1:
2 * M_PI - atan2(obsyhat, 1 - obsxhat);
% elif chunk == 2:
M_PI - atan2(1 - obsyhat, obsxhat);
% endif
</%def>

<%def name="co_rhohigh(chunk)">\
% if chunk == 0:
-obsxhat / cos(theta);
% elif chunk == 1:
-obsyhat / sin(theta);
% elif chunk == 2:
(1 - obsyhat - obsxhat) / (cos(theta) + sin(theta));
% endif
</%def>

<%def name="adj_theta_low(chunk)">\
% if chunk == 0:
0;
% elif chunk == 1:
M_PI - atan2(1, 1 - obsxhat);
% else:
0;
% endif
</%def>

<%def name="adj_theta_high(chunk)">\
% if chunk == 0:
M_PI - atan2(1, 1 - obsxhat);
% elif chunk == 1:
M_PI;
% else:
0;
% endif
</%def>

<%def name="adj_rhohigh(chunk)">\
% if chunk == 0:
obsxhat / (costheta + sintheta);
% elif chunk == 1:
-(1 - obsxhat) / costheta;
% else:
0;
% endif
</%def>

<%def name="func_def(type, k_name)">
__kernel
void ${type}_integrals${k_name}(__global ${float_type}* result, int chunk, 
    __global ${float_type}* pts, 
    int n_rho, __global ${float_type}* rho_qx, __global ${float_type}* rho_qw, 
    int n_theta, __global ${float_type}* theta_qx, __global ${float_type}* theta_qw, 
    __global ${float_type}* in_obs_tri, __global ${float_type}* in_src_tri,
    ${float_type} eps, ${float_type} G, ${float_type} nu)
</%def>

<%def name="zero_output()">
    ${float_type} sum[81];
    ${float_type} kahanC[81];
    for (int i = 0; i < 81; i++) {
        sum[i] = 0;
        kahanC[i] = 0;
    }
</%def>

<%def name="integral_setup(obs_tri_name, src_tri_name)">
    const int cell_idx = get_global_id(0);

    ${constants()}
    ${float_type} obs_tri[3][3];
    ${float_type} src_tri[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            obs_tri[i][j] = ${obs_tri_name}[i * 3 + j];
            src_tri[i][j] = ${src_tri_name}[i * 3 + j];
        }
    }
    ${tri_info("obs", "n")}
    ${tri_info("src", "l")}
</%def>

<%def name="eval_hatvars()">
${float_type} obsxhat = pts[cell_idx * 2];
${float_type} obsyhat = pts[cell_idx * 2 + 1] * (1 - obsxhat);
</%def>

<%def name="rho_quad_eval()">
${float_type} rhohat = (rho_qx[ri] + 1) / 2.0;
${float_type} rho = rhohat * rhohigh;
${float_type} jacobian = rho_qw[ri] * rho * rhohigh * outer_jacobian;
</%def>

<%def name="setup_kernel_inputs()">
    % for which, ptname in [("obs", "x_no_offset_"), ("src", "y")]:
        ${basis(which + "")}
        ${pts_from_basis(
            ptname, which + "",
            lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
        )}
    % endfor

    % for dim in range(3):
        ${float_type} x${dn(dim)} = x_no_offset_${dn(dim)} - eps * n${dn(dim)};
    % endfor

    ${float_type} Dx = yx - xx;
    ${float_type} Dy = yy - xy; 
    ${float_type} Dz = yz - xz;
    ${float_type} r2 = Dx * Dx + Dy * Dy + Dz * Dz;
    if (r2 == 0.0) {
        continue;
    }
</%def>

<%def name="add_to_sum()">
    % for d_obs in range(3):
        % for d_src in range(3):
            {
                ${float_type} kernel_val = jacobian * K${d_obs}${d_src};
                % for b_obs in range(3):
                    % for b_src in range(3):
                        {
                            int idx = ${temp_result_idx(d_obs, d_src, b_obs, b_src)};
                            ${float_type} add_to_sum = kernel_val * obsb${b_obs} * srcb${b_src};
                            ${float_type} y = add_to_sum - kahanC[idx];
                            ${float_type} t = sum[idx] + y;
                            kahanC[idx] = (t - sum[idx]) - y;
                            sum[idx] = t;
                        }
                    % endfor
                % endfor
            }
        % endfor
    % endfor
</%def>

<%def name="coincident_integrals(k_name)">
${func_def("coincident", k_name)}
{
    ${zero_output()}
    ${integral_setup("in_obs_tri", "in_src_tri")}
    ${eval_hatvars()}

    for (int oti = 0; oti < n_theta; oti++) {
        ${float_type} thetahat = (theta_qx[oti] + 1) / 2;

        ${float_type} thetalow;
        ${float_type} thetahigh;
        if (chunk == 0) {
            thetalow = ${co_theta_low(0)}
            thetahigh = ${co_theta_high(0)}
        } else if (chunk == 1) {
            thetalow = ${co_theta_low(1)}
            thetahigh = ${co_theta_high(1)}
        } else {
            thetalow = ${co_theta_low(2)}
            thetahigh = ${co_theta_high(2)}
        }
        ${float_type} theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

        ${float_type} outer_jacobian = 
            0.5 * theta_qw[oti] * (1 - obsxhat) * (thetahigh - thetalow);
        ${float_type} costheta = cos(theta);
        ${float_type} sintheta = sin(theta);

        ${float_type} rhohigh;
        if (chunk == 0) {
            rhohigh = ${co_rhohigh(0)}
        } else if (chunk == 1) {
            rhohigh = ${co_rhohigh(1)}
        } else {
            rhohigh = ${co_rhohigh(2)}
        }

        for (int ri = 0; ri < n_rho; ri++) {
            ${rho_quad_eval()}

            ${float_type} srcxhat = obsxhat + rho * costheta;
            ${float_type} srcyhat = obsyhat + rho * sintheta;

            ${setup_kernel_inputs()}
            ${tensor_kernels(k_name)}
            ${add_to_sum()}
        }
    }
    ${float_type} const_jacobian = 0.5 * obs_jacobian * src_jacobian;
    for (int i = 0; i < 81; i++) {
        result[cell_idx * 81 + i] = const_jacobian * (sum[i] + kahanC[i]);
    }
}
</%def>

<%def name="adjacent_integrals(k_name)">
${func_def("adjacent", k_name)}
{
    ${zero_output()}
    ${integral_setup("in_obs_tri", "in_src_tri")}
    ${eval_hatvars()}

    for (int oti = 0; oti < n_theta; oti++) {
        ${float_type} thetahat = (theta_qx[oti] + 1) / 2;

        ${float_type} thetalow;
        ${float_type} thetahigh;
        if (chunk == 0) {
            thetalow = ${adj_theta_low(0)}
            thetahigh = ${adj_theta_high(0)}
        } else {
            thetalow = ${adj_theta_low(1)}
            thetahigh = ${adj_theta_high(1)}
        }
        ${float_type} theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

        ${float_type} outer_jacobian = 
            0.5 * theta_qw[oti] * (1 - obsxhat) * (thetahigh - thetalow);
        ${float_type} costheta = cos(theta);
        ${float_type} sintheta = sin(theta);

        ${float_type} rhohigh;
        if (chunk == 0) {
            rhohigh = ${adj_rhohigh(0)}
        } else {
            rhohigh = ${adj_rhohigh(1)}
        }

        for (int ri = 0; ri < n_rho; ri++) {
            ${rho_quad_eval()}

            ${float_type} srcxhat = rho * costheta + (1 - obsxhat);
            ${float_type} srcyhat = rho * sintheta;

            ${setup_kernel_inputs()}
            ${tensor_kernels(k_name)}
            ${add_to_sum()}
        }
    }

    ${float_type} const_jacobian = 0.5 * obs_jacobian * src_jacobian;
    for (int i = 0; i < 81; i++) {
        result[cell_idx * 81 + i] = const_jacobian * sum[i];
    }
}
</%def>

${geometry_fncs()}

% for k_name in ['U', 'T', 'A', 'H']:
${coincident_integrals(k_name)}
${adjacent_integrals(k_name)}
% endfor
