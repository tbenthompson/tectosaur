#include <stdio.h>
<%
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

__device__
void cross(double x[3], double y[3], double out[3]) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

__device__
void sub(double x[3], double y[3], double out[3]) {
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
}

__device__
void get_unscaled_normal(double tri[3][3], double out[3]) {
    double s20[3];
    double s21[3];
    sub(tri[2], tri[0], s20);
    sub(tri[2], tri[1], s21);
    cross(s20, s21, out);
}

__device__
double magnitude(double v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

<%def name="get_triangle(name, tris, index)">
double ${name}[3][3];
for (int c = 0; c < 3; c++) {
    for (int d = 0; d < 3; d++) {
        ${name}[c][d] = pts[3 * ${tris}[3 * ${index} + c] + d];
    }
}
</%def>

<%def name="tri_info(prefix,normal_prefix)">
double ${prefix}_unscaled_normal[3];
get_unscaled_normal(${prefix}_tri, ${prefix}_unscaled_normal);
double ${prefix}_normal_length = magnitude(${prefix}_unscaled_normal);
double ${prefix}_jacobian = ${prefix}_normal_length;
% for dim in range(3):
double ${normal_prefix}${dn(dim)} = 
    ${prefix}_unscaled_normal[${dim}] / ${prefix}_normal_length;
% endfor
</%def>

<%def name="basis(prefix)">
double ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
double ${prefix}b1 = ${prefix}xhat;
double ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name,ndims)">
% for dim in range(ndims):
double ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b${basis} * ${tri_name(basis,dim)};
% endfor
% endfor
</%def>

<%def name="co_theta_low(chunk)">\
% if chunk == 0:
M_PI - atan((1 - obsyhat) / obsxhat);
% elif chunk == 1:
M_PI + atan(obsyhat / obsxhat);
% elif chunk == 2:
-atan(obsyhat / (1 - obsxhat));
% endif
</%def>

<%def name="co_theta_high(chunk)">\
% if chunk == 0:
M_PI + atan(obsyhat / obsxhat);
% elif chunk == 1:
2 * M_PI - atan(obsyhat / (1 - obsxhat));
% elif chunk == 2:
M_PI - atan((1 - obsyhat) / obsxhat);
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

<%def name="temp_result_idx(d_obs, d_src, b_obs, b_src)">
${b_obs} * 27 + ${d_obs} * 9 + ${b_src} * 3 + ${d_src}
</%def>

<%def name="compute_integrals(chunk)">
__global__
void compute_integrals${chunk}(double* result, int n_quad_pts,
    double* quad_pts, double* quad_wts, double* mins, double* maxs,
    double* tri, double eps, double G, double nu,
    int n_rho_quad_pts, double* rho_qx, double* rho_qw)
{
    const int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double minx = mins[cell_idx * 3 + 0];
    double miny = mins[cell_idx * 3 + 1];
    double mintheta = mins[cell_idx * 3 + 2];
    double deltax = (maxs[cell_idx * 3 + 0] - minx) * 0.5;
    double deltay = (maxs[cell_idx * 3 + 1] - miny) * 0.5;
    double deltatheta = (maxs[cell_idx * 3 + 2] - mintheta) * 0.5;

    /*const float CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));*/
    /*const float CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));*/
    /*const float CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));*/
    /*const float CsT1 = 3.0/(8.0*M_PI*(1.0-nu));*/
    /*const float CsT2 = 1.0/(8*M_PI*(1-nu));*/
    /*const float CsT3 = 1-2*nu;*/
    const float CsH0 = G/(4*M_PI*(1-nu));
    const float CsH1 = 1-2*nu;
    const float CsH2 = -1+4*nu;
    const float CsH3 = 3*nu;

    double obs_tri[3][3];
    double src_tri[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            obs_tri[i][j] = tri[i * 3 + j];
            src_tri[i][j] = tri[i * 3 + j];
        }
    }
    ${tri_info("obs", "n")}
    ${tri_info("src", "l")}

    double sum[81];
    for (int i = 0; i < 81; i++) {
        sum[i] = 0;
    }
    for (int i = 0; i < n_quad_pts; i++) {
        double obsxhat = minx + deltax * (quad_pts[i * 3] + 1);
        double obsyhat = (miny + deltay * (quad_pts[i * 3 + 1] + 1)) * (1 - obsxhat);
        double thetahat = mintheta + deltatheta * (quad_pts[i * 3 + 2] + 1);
        double w = quad_wts[i] * deltax * deltay * deltatheta;

        double thetalow = ${co_theta_low(chunk)}
        double thetahigh = ${co_theta_high(chunk)}
        double theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

        double outer_jacobian = w * (1 - obsxhat) * (thetahigh - thetalow) * 
                                0.5 * obs_jacobian * src_jacobian;;
        double costheta = cos(theta);
        double sintheta = sin(theta);

        /*for (int ri = 0; ri < n_rho_quad_pts; ri++) {*/
        for (int ri = 0; ri < 1; ri++) {
            double rhohat = (rho_qx[ri] + 1) / 2.0;
            double rhohigh = ${co_rhohigh(chunk)}
            double rho = rhohat * rhohigh;
            double jacobian = rho_qw[i] * rho * rhohigh * outer_jacobian;

            double srcxhat = rho * costheta + (1 - obsxhat);
            double srcyhat = rho * sintheta;

            % for which, ptname in [("obs", "x"), ("src", "y")]:
                ${basis(which + "")}
                ${pts_from_basis(
                    ptname, which + "",
                    lambda b, d: which + "_tri[" + str(b) + "][" + str(d) + "]", 3
                )}
            % endfor

            % for dim in range(3):
                x${dn(dim)} -= eps * n${dn(dim)};
            % endfor

            double Dx = yx - xx;
            double Dy = yy - xy; 
            double Dz = yz - xz;
            double r2 = Dx * Dx + Dy * Dy + Dz * Dz;
            if (r2 == 0.0) {
                continue;
            }

            double invr = 1.0 / sqrt(r2);
            double invr2 = invr * invr;
            double invr3 = invr2 * invr;
            double Dorx = invr * Dx;
            double Dory = invr * Dy;
            double Dorz = invr * Dz;

            double rn = lx * Dorx + ly * Dory + lz * Dorz;
            double rm = nx * Dorx + ny * Dory + nz * Dorz;
            double mn = nx * lx + ny * ly + nz * lz;

            double Q = CsH0 * invr3;
            double A = Q * 3 * rn;
            double B = Q * CsH1;
            double C = Q * CsH3;

            double MTx = Q*CsH2*lx + A*CsH1*Dorx;
            double MTy = Q*CsH2*ly + A*CsH1*Dory;
            double MTz = Q*CsH2*lz + A*CsH1*Dorz;

            double NTx = B*nx + C*Dorx*rm;
            double NTy = B*ny + C*Dory*rm;
            double NTz = B*nz + C*Dorz*rm;

            double DTx = B*3*lx*rm + C*Dorx*mn + A*(nu*nx - 5*Dorx*rm);
            double DTy = B*3*ly*rm + C*Dory*mn + A*(nu*ny - 5*Dory*rm);
            double DTz = B*3*lz*rm + C*Dorz*mn + A*(nu*nz - 5*Dorz*rm);

            double ST = A*nu*rm + B*mn;

            double K00 = lx*NTx + nx*MTx + Dorx*DTx + ST;
            double K01 = lx*NTy + nx*MTy + Dorx*DTy;
            double K02 = lx*NTz + nx*MTz + Dorx*DTz;
            double K10 = ly*NTx + ny*MTx + Dory*DTx;
            double K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
            double K12 = ly*NTz + ny*MTz + Dory*DTz;
            double K20 = lz*NTx + nz*MTx + Dorz*DTx;
            double K21 = lz*NTy + nz*MTy + Dorz*DTy;
            double K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;
            % for d_obs in range(3):
                % for d_src in range(3):
                    {
                        double kernel_val = jacobian * K${d_obs}${d_src};
                        % for b_obs in range(3):
                            % for b_src in range(3):
                                {
                                    int idx = ${temp_result_idx(d_obs, d_src, b_obs, b_src)};
                                    sum[idx] += obsb${b_obs} * srcb${b_src} * kernel_val;
                                }
                            % endfor
                        % endfor
                    }
                % endfor
            % endfor
        }
    }
    for (int i = 0; i < 81; i++) {
        result[cell_idx * 81 + i] = sum[i];
    }
}
</%def>

% for chunk in range(3):
${compute_integrals(chunk)}
% endfor
/*{*/
/*    const double eps = 0.001;*/
/*    const double A = 1.5;*/
/**/
/*    const int i = blockIdx.x * blockDim.x + threadIdx.x;*/
/*    double minx = mins[i * 3 + 0];*/
/*    double miny = mins[i * 3 + 1];*/
/*    double minz = mins[i * 3 + 2];*/
/*    double deltax = (maxs[i * 3 + 0] - minx) * 0.5;*/
/*    double deltay = (maxs[i * 3 + 1] - miny) * 0.5;*/
/*    double deltaz = (maxs[i * 3 + 2] - minz) * 0.5;*/
/**/
/*    double sum = 0.0;*/
/*    for (int iq = 0; iq < n_quad_pts; iq++) {*/
/*        double x = minx + deltax * (quad_pts[iq] + 1);*/
/*        double y = miny + deltay * (quad_pts[n_quad_pts * 1 + iq] + 1);*/
/*        double z = minz + deltaz * (quad_pts[n_quad_pts * 2 + iq] + 1);*/
/*        double w = quad_wts[iq] * deltax * deltay * deltaz;*/
/*        double xd = x - z;*/
/*        double yd = y - 0.3;*/
/*        double val = powf(xd * xd + yd * yd + eps * eps, -A);*/
/*        sum += val * w;*/
/*    }*/
/*    result[i] = sum;*/
/*}*/
