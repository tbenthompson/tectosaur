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

<%def name="compute_integrals(chunk)">
__global__
void compute_integrals${chunk}(double* result, int n_quad_pts,
    double* quad_pts, double* quad_wts, double* mins, double* maxs,
    double* tri, double eps, double sm, double pr,
    int n_rho_quad_pts, double* rho_qx, double* rho_qw)
{
    const int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
    double minx = mins[cell_idx * 3 + 0];
    double miny = mins[cell_idx * 3 + 1];
    double mintheta = mins[cell_idx * 3 + 2];
    double deltax = (maxs[cell_idx * 3 + 0] - minx) * 0.5;
    double deltay = (maxs[cell_idx * 3 + 1] - miny) * 0.5;
    double deltatheta = (maxs[cell_idx * 3 + 2] - mintheta) * 0.5;

    // Add tri info.
    double unscaled_normal[3];

    double sum[81];
    for (int i = 0; i < 81; i++) {
        sum[i] = 0;
    }
    for (int i = 0; i < 1; i++) {
        double obsxhat = minx + deltax * (quad_pts[i] + 1);
        double obsyhat = miny + deltay * (quad_pts[n_quad_pts * 1 + i] + 1) * (1 - obsxhat);
        double thetahat = mintheta + deltatheta * (quad_pts[n_quad_pts * 2 + i] + 1);
        double w = quad_wts[i] * deltax * deltay * deltatheta;

        double thetalow = ${co_theta_low(chunk)}
        double thetahigh = ${co_theta_high(chunk)}
        double theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

        double outer_jacobian = (1 - obsxhat) * (thetahigh - thetalow) * 0.5;
        double costheta = cos(theta);
        double sintheta = sin(theta);
        for (size_t ri = 0; ri < 1; ri++) {
            double rhohat = (rho_qx[ri] + 1) / 2.0;
            double jacobian = rho_qw[ri] * outer_jacobian;

            double rhohigh = ${co_rhohigh(chunk)}
            double rho = rhohat * rhohigh;
            jacobian *= rho * rhohigh;

            double srcxhat = rho * costheta + (1 - obsxhat);
            double srcyhat = rho * sintheta;

            /*auto out = bem_integrand(d, obsxhat, obsyhat, srcxhat, srcyhat);*/
            /*for (int i = 0; i < 9; i++) {*/
            /*    fval[i] += jacobian * out[i];*/
            /*}*/
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
