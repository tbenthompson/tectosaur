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

struct Data {
    int evals = 0;
    std::function<double(double,double)> theta_low;
    std::function<double(double,double)> theta_high;
    std::function<double(double,double,double)> rhohigh;
    double A;
    double B;
    double eps;
    double G;
    double nu;
    double CsU0;
    double CsU1;
    double CsH0;
    double CsH1;
    double CsH2;
    double CsH3;

    Data(double A, double B, int piece, double eps, double G, double nu):
        A(A),
        B(B),
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

int f(unsigned ndim, long unsigned npts, const double *x,
    void *fdata, unsigned fdim, double *fval) 
{
    Data& d = *reinterpret_cast<Data*>(fdata);
    d.evals += npts;
    auto ref2realx = [&] (double xhat, double yhat) { return xhat + yhat * d.A; };
    auto ref2realy = [&] (double xhat, double yhat) { return yhat * d.B; };

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
        double xx = ref2realx(obsxhat, obsyhat);
        double xy = ref2realy(obsxhat, obsyhat);
        double xz = d.eps;
        jacobian *= d.B;

        double yx = ref2realx(srcxhat, srcyhat);
        double yy = ref2realy(srcxhat, srcyhat);
        double yz = 0;
        jacobian *= d.B;

        double nx = 0;
        double ny = 0;
        double nz = 1;
        double lx = 0;
        double ly = 0;
        double lz = 1;


        double Dx = yx - xx;
        double Dy = yy - xy;
        double Dz = yz - xz;
        double r2 = Dx * Dx + Dy * Dy + Dz * Dz;

        // U kernel
        // double invr = 1.0 / sqrt(r2);
        // double Q1 = d.CsU0 * invr;
        // double Q2 = d.CsU1 * pow(invr,3);
        // double K00 = Q2*Dx*Dx + Q1;
        // double K01 = Q2*Dx*Dy;
        // double K02 = Q2*Dx*Dz;
        // double K10 = Q2*Dy*Dx;
        // double K11 = Q2*Dy*Dy + Q1;
        // double K12 = Q2*Dy*Dz;
        // double K20 = Q2*Dz*Dx;
        // double K21 = Q2*Dz*Dy;
        // double K22 = Q2*Dz*Dz + Q1;

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
        // double K01 = lx*NTy + nx*MTy + Dorx*DTy;
        // double K02 = lx*NTz + nx*MTz + Dorx*DTz;
        // double K10 = ly*NTx + ny*MTx + Dory*DTx;
        // double K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
        // double K12 = ly*NTz + ny*MTz + Dory*DTz;
        // double K20 = lz*NTx + nz*MTx + Dorz*DTx;
        // double K21 = lz*NTy + nz*MTy + Dorz*DTy;
        // double K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;

        fval[i] = K00 * jacobian;
        // fval[i] = K00 * (1 - x[i * 4 + 2]) * (1 - x[i * 4 + 0]);
    }
    return 0; 
}

/* Chebyshev nodes in the [a,b] interval. Good for interpolation. Avoid Runge
 * phenomenon. 
 */
std::vector<double> chebab(double a, double b, int n) {
    std::vector<double> out(n);
    for (int i = 0; i < n; i++) {
        out[i] = 0.5 * (a + b) + 0.5 * (b - a) * cos(((2 * i + 1) * M_PI) / (2 * n));
    }
    return out;
}

int main() {
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

            double xmin[4] = {0,0,0,0};
            double xmax[4] = {1,1,1,1};
            double sum = 0;
            double val, err;
            for (int piece = 0; piece < 3; piece++) {

                Data d(A, B, piece, 0.1, 1.0, 0.25);
                hcubature_v(1, f, &d, 4, xmin, xmax, 0, 0, 1e-5, ERROR_INDIVIDUAL, &val, &err);
                sum += val;
                // std::cout << val << " " << err << std::endl;
                // std::cout << "With evals = " << d.evals << std::endl;
            }
            std::cout << i << ", " << j << ", " 
                << A << ", " << B << ", " << sum << ", " << std::endl;
        }
    }
}
