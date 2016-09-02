<%
setup_pybind11(cfg)
%>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

% for k_name in ['U', 'T', 'A', 'H']:
std::array<double,9> eval${k_name}(double xx, double xy, double xz,
        double nx, double ny, double nz,
        double yx, double yy, double yz,
        double lx, double ly, double lz,
        double G, double nu)
{
    double Dx = yx - xx;
    double Dy = yy - xy;
    double Dz = yz - xz;
    double r2 = Dx * Dx + Dy * Dy + Dz * Dz;

    % if k_name is 'U':
        double CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
        double CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
        double invr = 1.0 / sqrt(r2);
        double Q1 = CsU0 * invr;
        double Q2 = CsU1 * invr / r2;
        double K00 = Q2*Dx*Dx + Q1;
        double K01 = Q2*Dx*Dy;
        double K02 = Q2*Dx*Dz;
        double K10 = Q2*Dy*Dx;
        double K11 = Q2*Dy*Dy + Q1;
        double K12 = Q2*Dy*Dz;
        double K20 = Q2*Dz*Dx;
        double K21 = Q2*Dz*Dy;
        double K22 = Q2*Dz*Dz + Q1;
    % elif k_name is 'T' or k_name is 'A':
        double CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));
        double CsT1 = 3.0/(8.0*M_PI*(1.0-nu));
        <%
            minus_or_plus = '-' if k_name is 'T' else '+'
            plus_or_minus = '+' if k_name is 'T' else '-'
            n_name = 'l' if k_name is 'T' else 'n'
        %>
        double invr = 1.0 / sqrt(r2);
        double invr2 = invr * invr;
        double invr3 = invr2 * invr;

        double rn = ${n_name}x * Dx + ${n_name}y * Dy + ${n_name}z * Dz;

        double A = ${plus_or_minus}CsT0 * invr3;
        double C = ${minus_or_plus}CsT1 * invr3 * invr2;

        double nxdy = ${n_name}x*Dy-${n_name}y*Dx;
        double nzdx = ${n_name}z*Dx-${n_name}x*Dz;
        double nzdy = ${n_name}z*Dy-${n_name}y*Dz;

        double K00 = A * -rn                  + C*Dx*rn*Dx;
        double K01 = A * ${minus_or_plus}nxdy + C*Dx*rn*Dy;
        double K02 = A * ${plus_or_minus}nzdx + C*Dx*rn*Dz;
        double K10 = A * ${plus_or_minus}nxdy + C*Dy*rn*Dx;
        double K11 = A * -rn                  + C*Dy*rn*Dy;
        double K12 = A * ${plus_or_minus}nzdy + C*Dy*rn*Dz;
        double K20 = A * ${minus_or_plus}nzdx + C*Dz*rn*Dx;
        double K21 = A * ${minus_or_plus}nzdy + C*Dz*rn*Dy;
        double K22 = A * -rn                  + C*Dz*rn*Dz;
    % elif k_name is 'H':
        double CsH0 = G/(4*M_PI*(1-nu));
        double CsH1 = 1-2*nu;
        double CsH2 = -1+4*nu;
        double CsH3 = 3*nu;
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
    % endif
    return {K00,K01,K02,K10,K11,K12,K20,K21,K22}; 
}
% endfor


PYBIND11_PLUGIN(ext) {
    pybind11::module m("ext", "");

    % for k_name in ['U', 'T', 'A', 'H']:
    m.def("eval${k_name}", &eval${k_name});
    % endfor

    return m.ptr();
}
