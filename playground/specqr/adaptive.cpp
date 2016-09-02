#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include "cubature/cubature.h"

double thetalim0(double x, double y) { return M_PI - atan((1 - y) / x); }
double thetalim1(double x, double y) { return M_PI + atan(y / x); }
double thetalim2a(double x, double y) { return 2 * M_PI - atan(y / (1 - x)); }
double thetalim2b(double x, double y) { return - atan(y / (1 - x)); }
double rholim0(double x, double y, double t) { return (1 - y - x) / (cos(t) + sin(t)); }
double rholim1(double x, double y, double t) { return -x / cos(t); }
double rholim2(double x, double y, double t) { return -y / sin(t); }

using Tri = std::array<std::array<double,3>,3>;

struct Data {
    int evals;
    std::function<double(double,double)> theta_low;
    std::function<double(double,double)> theta_high;
    std::function<double(double,double,double)> rhohigh;
    Tri tri;
    double eps;
    double G;
    double nu;
    double CsU0;
    double CsU1;
    double CsH0;
    double CsH1;
    double CsH2;
    double CsH3;

    Data(std::array<std::array<double,3>,3> tri, int piece, double eps, double G, double nu):
        tri(tri),
        eps(eps),
        G(G),
        nu(nu),
        CsU0((3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu))),
        CsU1(1.0/(G*16.0*M_PI*(1.0-nu))),
        CsH0(G/(4*M_PI*(1-nu))),
        CsH1(1-2*nu),
        CsH2(-1+4*nu),
        CsH3(3*nu)
    {
        if (piece == 0) {
            theta_low = thetalim0; theta_high = thetalim1; rhohigh = rholim1;
        } else if (piece == 1) {
            theta_low = thetalim1; theta_high = thetalim2a; rhohigh = rholim2;
        } else if (piece == 2) {
            theta_low = thetalim2b; theta_high = thetalim0; rhohigh = rholim0;
        }
    }
};

double basiseval(double xhat, double yhat, std::array<double,3> vals) {
    return (1.0 - xhat - yhat) * vals[0] + xhat * vals[1] + yhat * vals[2];
}

std::array<std::array<double,3>,3> Ukernel(Data& d, double Dx, double Dy, double Dz,
    double nx, double ny, double nz, double lx, double ly, double lz)
{
    double r2 = Dx * Dx + Dy * Dy + Dz * Dz;
    double invr = 1.0 / sqrt(r2);
    double Q1 = d.CsU0 * invr;
    double Q2 = d.CsU1 * pow(invr,3);
    double K00 = Q2*Dx*Dx + Q1;
    double K01 = Q2*Dx*Dy;
    double K02 = Q2*Dx*Dz;
    double K10 = Q2*Dy*Dx;
    double K11 = Q2*Dy*Dy + Q1;
    double K12 = Q2*Dy*Dz;
    double K20 = Q2*Dz*Dx;
    double K21 = Q2*Dz*Dy;
    double K22 = Q2*Dz*Dz + Q1;
    return {{{K00,K01,K02},{K10,K11,K12},{K20,K21,K22}}};
}

std::array<std::array<double,3>,3> Hkernel(Data& d, double Dx, double Dy, double Dz,
    double nx, double ny, double nz, double lx, double ly, double lz)
{
    double r2 = Dx * Dx + Dy * Dy + Dz * Dz;
    double invr = 1.0 / sqrt(r2);
    double invr2 = invr * invr;
    double invr3 = invr2 * invr;
    double Dorx = invr * Dx;
    double Dory = invr * Dy;
    double Dorz = invr * Dz;

    double rn = lx * Dorx + ly * Dory + lz * Dorz;
    double rm = nx * Dorx + ny * Dory + nz * Dorz;
    double mn = nx * lx + ny * ly + nz * lz;

    double Q = d.CsH0 * invr3;
    double A = Q * 3 * rn;
    double B = Q * d.CsH1;
    double C = Q * d.CsH3;

    double MTx = Q*d.CsH2*lx + A*d.CsH1*Dorx;
    double MTy = Q*d.CsH2*ly + A*d.CsH1*Dory;
    double MTz = Q*d.CsH2*lz + A*d.CsH1*Dorz;

    double NTx = B*nx + C*Dorx*rm;
    double NTy = B*ny + C*Dory*rm;
    double NTz = B*nz + C*Dorz*rm;

    double DTx = B*3*lx*rm + C*Dorx*mn + A*(d.nu*nx - 5*Dorx*rm);
    double DTy = B*3*ly*rm + C*Dory*mn + A*(d.nu*ny - 5*Dory*rm);
    double DTz = B*3*lz*rm + C*Dorz*mn + A*(d.nu*nz - 5*Dorz*rm);

    double ST = A*d.nu*rm + B*mn;

    double K00 = lx*NTx + nx*MTx + Dorx*DTx + ST;
    double K01 = lx*NTy + nx*MTy + Dorx*DTy;
    double K02 = lx*NTz + nx*MTz + Dorx*DTz;
    double K10 = ly*NTx + ny*MTx + Dory*DTx;
    double K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
    double K12 = ly*NTz + ny*MTz + Dory*DTz;
    double K20 = lz*NTx + nz*MTx + Dorz*DTx;
    double K21 = lz*NTy + nz*MTy + Dorz*DTy;
    double K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;
    return {{{K00,K01,K02},{K10,K11,K12},{K20,K21,K22}}};
}

int f(unsigned ndim, long unsigned npts, const double *x,
    void *fdata, unsigned fdim, double *fval) 
{
    Data& d = *reinterpret_cast<Data*>(fdata);
    d.evals += npts;

    // std::cout << d.evals << " " << npts << std::endl;

#pragma omp parallel for
    for (unsigned i = 0; i < npts; i++) {
        double obsxhat = x[i * 4 + 0];
        double obsyhat = x[i * 4 + 1] * (1 - x[i * 4 + 0]);
        double jacobian = (1 - x[i * 4 + 0]);

        double thetahat = x[i * 4 + 2];
        double thetalow = d.theta_low(obsxhat, obsyhat);
        double thetahigh = d.theta_high(obsxhat, obsyhat);
        double theta = (1 - thetahat) * thetalow + thetahat * thetahigh;
        jacobian *= thetahigh - thetalow;

        // tried a sinh transform here and it doesn't work well
        // double s = 2.0 * x[i * 4 + 3] - 1.0;
        // double a = -1.0;
        // double b = eps;
        // double mu0 = 0.5 * (asinh((1.0 + a) / b) + asinh((1.0 - a) / b));
        // double eta0 = 0.5 * (asinh((1.0 + a) / b) - asinh((1.0 - a) / b));
        // double rhohat = ((a + b * sinh(mu0 * s - eta0)) + 1.0) / 2.0;
        // jacobian *= b * mu0 * cosh(mu0 * s - eta0);
        double rhohat = x[i * 4 + 3];

        double rhohigh = d.rhohigh(obsxhat, obsyhat, theta);
        double rho = rhohat * rhohigh;
        jacobian *= rho * rhohigh;

        double srcxhat = obsxhat + rho * cos(theta);
        double srcyhat = obsyhat + rho * sin(theta);
        // double srcxhat = x[i * 4 + 2];
        // double srcyhat = x[i * 4 + 3] * (1 - x[i * 4 + 2]);
        // double xx = basiseval(obsxhat, obsyhat, {d.tri[0][0], d.tri[1][0], d.tri[2][0]});
        // double xy = basiseval(obsxhat, obsyhat, {d.tri[0][1], d.tri[1][1], d.tri[2][1]});
        double xx = obsxhat;
        double xy = obsyhat;
        double xz = d.eps;

        // double yx = ref2realx(srcxhat, srcyhat);
        // double yy = ref2realy(srcxhat, srcyhat);
        double yx = srcxhat;
        double yy = srcyhat;
        double yz = 0;

        double nx = 0;
        double ny = 0;
        double nz = 1;
        double lx = 0;
        double ly = 0;
        double lz = 1;


        double Dx = yx - xx;
        double Dy = yy - xy;
        double Dz = yz - xz;

        auto K = Ukernel(d, Dx, Dy, Dz, nx, ny, nz, lx, ly, lz);

        fval[i] = K[0][0] * jacobian;
        // fval[i] = K00 * (1 - x[i * 4 + 2]) * (1 - x[i * 4 + 0]);
    }
    return 0; 
}

double integrate(Data& d) {
    d.evals = 0;
    const double xmin[4] = {0,0,0,0};
    const double xmax[4] = {1,1,1,1};
    double val, err;
    hcubature_v(1, f, &d, 4, xmin, xmax, 0, 0, 1e-5, ERROR_INDIVIDUAL, &val, &err);
    return val;
}

/* Chebyshev nodes in the [a,b] interval. Good for interpolation. Avoids Runge
 * phenomenon. 
 */
std::vector<double> chebab(double a, double b, int n) {
    std::vector<double> out(n);
    for (int i = 0; i < n; i++) {
        out[i] = 0.5 * (a + b) + 0.5 * (b - a) * cos(((2 * i + 1) * M_PI) / (2 * n));
    }
    return out;
}

void testHinterp() {
    // Left right symmetry means that I only need to go from A in [0.0, 0.5]
    int nA = 16;
    int nB = 31;
    auto minlegalA = 0.100201758858;
    auto minlegalB = 0.24132345693;
    auto maxlegalB = 0.860758641203;
    auto Avals = chebab(minlegalA, 0.5, nA); 
    auto Bvals = chebab(minlegalB, maxlegalB, nB); 
    for (int i = 0; i < nA; i++) {
        double A = Avals[i];
        for (int j = 0; j < nB; j++) {
            double B = Bvals[j];
            double sum = 0;
            for (int piece = 0; piece < 3; piece++) {

                Data d(Tri{{{0,0,0},{1,0,0},{A,B,0}}}, piece, 0.1, 1.0, 0.25);
                sum += integrate(d);
                // std::cout << val << " " << err << std::endl;
                // std::cout << "With evals = " << d.evals << std::endl;
            }
            std::cout << i << ", " << j << ", " 
                << A << ", " << B << ", " << sum << ", " << std::endl;
        }
    }
}

int main() {
    double sum = 0.0;
    for (int piece = 0; piece < 3; piece++) {
        Data d(Tri{{{0,0,0},{1,0,0},{0,1,0}}}, piece, 0.1, 1.0, 0.25);
        sum += integrate(d);
    }
    std::cout << sum << std::endl;
}
