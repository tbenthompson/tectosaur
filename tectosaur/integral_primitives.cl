<%!
def dn(dim):
    return ['x', 'y', 'z'][dim]
%>

<%def name="geometry_fncs()">
void vec_cross(Real x[3], Real y[3], Real out[3]) {
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
}

void sub(Real x[3], Real y[3], Real out[3]) {
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
}

void get_unscaled_normal(Real tri[3][3], Real out[3]) {
    Real s20[3];
    Real s21[3];
    sub(tri[2], tri[0], s20);
    sub(tri[2], tri[1], s21);
    vec_cross(s20, s21, out);
}

Real magnitude(Real v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}
</%def>

<%def name="get_triangle(name, tris, index)">
Real ${name}[3][3];
for (int c = 0; c < 3; c++) {
    for (int d = 0; d < 3; d++) {
        ${name}[c][d] = pts[3 * ${tris}[3 * ${index} + c] + d];
    }
}
</%def>

<%def name="tri_info(prefix,normal_prefix)">
Real ${prefix}_unscaled_normal[3];
get_unscaled_normal(${prefix}_tri, ${prefix}_unscaled_normal);
Real ${prefix}_normal_length = magnitude(${prefix}_unscaled_normal);
Real ${prefix}_jacobian = ${prefix}_normal_length;
% for dim in range(3):
Real ${normal_prefix}${dn(dim)} = 
    ${prefix}_unscaled_normal[${dim}] / ${prefix}_normal_length;
% endfor
</%def>

<%def name="basis(prefix)">
Real ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
Real ${prefix}b1 = ${prefix}xhat;
Real ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name,ndims)">
% for dim in range(ndims):
Real ${pt_pfx}${dn(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dn(dim)} += ${basis_pfx}b${basis} * ${tri_name(basis,dim)};
% endfor
% endfor
</%def>

<%def name="temp_result_idx(d_obs, d_src, b_obs, b_src)">
${b_obs} * 27 + ${d_obs} * 9 + ${b_src} * 3 + ${d_src}
</%def>

<%def name="constants()">
const Real CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
const Real CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
const Real CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));
const Real CsT1 = 3.0/(8.0*M_PI*(1.0-nu));
const Real CsH0 = G/(4*M_PI*(1-nu));
const Real CsH1 = 1-2*nu;
const Real CsH2 = -1+4*nu;
const Real CsH3 = 3*nu;
</%def>

<%def name="vector_kernels(k_name)">
% if k_name is 'U':
    Real invr = 1.0 / sqrt(r2);
    Real Q1 = CsU0 * invr;
    Real Q2 = CsU1 * invr / r2;
    Real ddi = Dx*Sx + Dy*Sy + Dz*Sz;
    sumx += Q1*Sx + Q2*Dx*ddi;
    sumy += Q1*Sy + Q2*Dy*ddi;
    sumz += Q1*Sz + Q2*Dz*ddi;
% elif k_name is 'T' or k_name is 'A':
    <%
        minus_or_plus = '-' if k_name is 'T' else '+'
        plus_or_minus = '+' if k_name is 'T' else '-'
        n_name = 'N' if k_name is 'T' else 'M'
    %>
    Real invr = 1.0 / sqrt(r2);
    Real invr2 = invr * invr;
    Real invr3 = invr2 * invr;

    Real rn = ${n_name}x * Dx + ${n_name}y * Dy + ${n_name}z * Dz;

    Real A = ${plus_or_minus}CsT0 * invr3;
    Real C = ${minus_or_plus}CsT1 * invr3 * invr2;

    Real rnddi = C * rn * (Dx*Sx + Dy*Sy + Dz*Sz);

    Real nxdy = ${n_name}x*Dy-${n_name}y*Dx;
    Real nzdx = ${n_name}z*Dx-${n_name}x*Dz;
    Real nzdy = ${n_name}z*Dy-${n_name}y*Dz;

    sumx += A*(
        - rn * Sx
        ${minus_or_plus} nxdy * Sy
        ${plus_or_minus} nzdx * Sz)
        + Dx*rnddi;
    sumy += A*(
        ${plus_or_minus} nxdy * Sx
        - rn * Sy
        ${plus_or_minus} nzdy * Sz)
        + Dy*rnddi;
    sumz += A*(
        ${minus_or_plus} nzdx * Sx 
        ${minus_or_plus} nzdy * Sy 
        - rn * Sz)
        + Dz*rnddi;
% elif k_name is 'H':
    Real invr = 1.0 / sqrt(r2);
    Real invr2 = invr * invr;
    Real invr3 = invr2 * invr;

    Real rn = invr*(Nx * Dx + Ny * Dy + Nz * Dz);
    Real rm = invr*(Mx * Dx + My * Dy + Mz * Dz);
    Real mn = Mx * Nx + My * Ny + Mz * Nz;

    Real sn = Sx*Nx + Sy*Ny + Sz*Nz;
    Real sd = invr*(Sx*Dx + Sy*Dy + Sz*Dz);
    Real sm = Sx*Mx + Sy*My + Sz*Mz;

    Real Q = CsH0 * invr3;
    Real A = Q * 3 * rn;
    Real B = Q * CsH1;
    Real C = Q * CsH3;

    Real MT = Q*CsH2*sn + A*CsH1*sd;
    Real NT = B*sm + C*sd*rm;
    Real DT = invr*(B*3*sn*rm + C*sd*mn + A*(nu*sm - 5*sd*rm));
    Real ST = A*nu*rm + B*mn;

    sumx += Nx*NT + Mx*MT + Dx*DT + ST*Sx;
    sumy += Ny*NT + My*MT + Dy*DT + ST*Sy;
    sumz += Nz*NT + Mz*MT + Dz*DT + ST*Sz;
%endif
</%def>

<%def name="tensor_kernels(k_name)">
% if k_name is 'U':
    Real invr = 1.0 / sqrt(r2);
    Real Q1 = CsU0 * invr;
    Real Q2 = CsU1 * invr / r2;
    Real K00 = Q2*Dx*Dx + Q1;
    Real K01 = Q2*Dx*Dy;
    Real K02 = Q2*Dx*Dz;
    Real K10 = Q2*Dy*Dx;
    Real K11 = Q2*Dy*Dy + Q1;
    Real K12 = Q2*Dy*Dz;
    Real K20 = Q2*Dz*Dx;
    Real K21 = Q2*Dz*Dy;
    Real K22 = Q2*Dz*Dz + Q1;
% elif k_name is 'T' or k_name is 'A':
    <%
        minus_or_plus = '-' if k_name is 'T' else '+'
        plus_or_minus = '+' if k_name is 'T' else '-'
        n_name = 'l' if k_name is 'T' else 'n'
    %>
    Real invr = 1.0 / sqrt(r2);
    Real invr2 = invr * invr;
    Real invr3 = invr2 * invr;

    Real rn = ${n_name}x * Dx + ${n_name}y * Dy + ${n_name}z * Dz;

    Real A = ${plus_or_minus}CsT0 * invr3;
    Real C = ${minus_or_plus}CsT1 * invr3 * invr2;

    Real nxdy = ${n_name}x*Dy-${n_name}y*Dx;
    Real nzdx = ${n_name}z*Dx-${n_name}x*Dz;
    Real nzdy = ${n_name}z*Dy-${n_name}y*Dz;

    Real K00 = A * -rn                  + C*Dx*rn*Dx;
    Real K01 = A * ${minus_or_plus}nxdy + C*Dx*rn*Dy;
    Real K02 = A * ${plus_or_minus}nzdx + C*Dx*rn*Dz;
    Real K10 = A * ${plus_or_minus}nxdy + C*Dy*rn*Dx;
    Real K11 = A * -rn                  + C*Dy*rn*Dy;
    Real K12 = A * ${plus_or_minus}nzdy + C*Dy*rn*Dz;
    Real K20 = A * ${minus_or_plus}nzdx + C*Dz*rn*Dx;
    Real K21 = A * ${minus_or_plus}nzdy + C*Dz*rn*Dy;
    Real K22 = A * -rn                  + C*Dz*rn*Dz;
% elif k_name is 'H':
    Real invr = 1.0 / sqrt(r2);
    Real invr2 = invr * invr;
    Real invr3 = invr2 * invr;
    Real Dorx = invr * Dx;
    Real Dory = invr * Dy;
    Real Dorz = invr * Dz;

    Real rn = lx * Dorx + ly * Dory + lz * Dorz;
    Real rm = nx * Dorx + ny * Dory + nz * Dorz;
    Real mn = nx * lx + ny * ly + nz * lz;

    Real Q = CsH0 * invr3;
    Real A = Q * 3 * rn;
    Real B = Q * CsH1;
    Real C = Q * CsH3;

    Real MTx = Q*CsH2*lx + A*CsH1*Dorx;
    Real MTy = Q*CsH2*ly + A*CsH1*Dory;
    Real MTz = Q*CsH2*lz + A*CsH1*Dorz;

    Real NTx = B*nx + C*Dorx*rm;
    Real NTy = B*ny + C*Dory*rm;
    Real NTz = B*nz + C*Dorz*rm;

    Real DTx = B*3*lx*rm + C*Dorx*mn + A*(nu*nx - 5*Dorx*rm);
    Real DTy = B*3*ly*rm + C*Dory*mn + A*(nu*ny - 5*Dory*rm);
    Real DTz = B*3*lz*rm + C*Dorz*mn + A*(nu*nz - 5*Dorz*rm);

    Real ST = A*nu*rm + B*mn;

    Real K00 = lx*NTx + nx*MTx + Dorx*DTx + ST;
    Real K01 = lx*NTy + nx*MTy + Dorx*DTy;
    Real K02 = lx*NTz + nx*MTz + Dorx*DTz;
    Real K10 = ly*NTx + ny*MTx + Dory*DTx;
    Real K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
    Real K12 = ly*NTz + ny*MTz + Dory*DTz;
    Real K20 = lz*NTx + nz*MTx + Dorz*DTx;
    Real K21 = lz*NTy + nz*MTy + Dorz*DTy;
    Real K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;
% endif
</%def>
