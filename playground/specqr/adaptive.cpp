#include <iostream>
#include <cmath>
#include <vector>
#include <functional>
#include <chrono>
#include "cubature/cubature.h"

// Multidimensional adaptive quadrature
// http://ab-initio.mit.edu/wiki/index.php/Cubature <-- using this
// http://mint.sbg.ac.at/HIntLib/
// http://www.feynarts.de/cuba/
//
//
// TODO:
// is hcubature or pcubature better?
//
// should i used vector integrands or is it better to split up the integral into each
// individual component so that they don't all need to take as long as the most expensive one
//
// write the table lookup procedure
//
// test the lookup by selecting random legal triangles and comparing the interpolation
//
// problem! running out of memory for higher accuracy integrals.. how to fix? can i split up
// the domain of integration? how do i reduce the accuracy requirements after splitting a domain?
// i suspect the answer is buried in the cubature code...

struct Timer {
    typedef std::chrono::high_resolution_clock::time_point Time;
    Time t_start;
    int time_us = 0;

    void start() {
        t_start = std::chrono::high_resolution_clock::now();
    }

    void stop() {
        time_us += std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - t_start
        ).count();
    }

    int get_time() {
        return time_us;
    }
};

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
    Timer timer;

    Data(std::array<std::array<double,3>,3> tri, double eps, double G, double nu):
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
        set_piece(0);
    }

    void set_piece(int piece) {
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

std::array<double,3> basis(double xhat, double yhat) {
     return {1 - xhat - yhat, xhat, yhat};
}

void all_zeros(double* fval) {
    for (int idx = 0; idx < 81; idx++) {
        fval[idx] = 0.0;
    }
}

int f(unsigned ndim, const double *x,
    void *fdata, unsigned fdim, double *fval) 
{
    Data& d = *reinterpret_cast<Data*>(fdata);
    d.evals += 1;

    // std::cout << d.evals << " " << npts << std::endl;
    double obsxhat = x[0];
    double obsyhat = x[1] * (1 - obsxhat);
    double jacobian = 1 - obsxhat;

    double thetahat = x[2];
    double thetalow = d.theta_low(obsxhat, obsyhat);
    double thetahigh = d.theta_high(obsxhat, obsyhat);
    double theta = (1 - thetahat) * thetalow + thetahat * thetahigh;
    if (std::isnan(theta)) { 
        // this is necessary to handle the interval endpoints used by pcubature
        all_zeros(fval);
        return 0; 
    }
    jacobian *= thetahigh - thetalow;

    // tried a sinh transform here and it doesn't work well
    double rhohat = x[3];

    double rhohigh = d.rhohigh(obsxhat, obsyhat, theta);
    double rho = rhohat * rhohigh;
    jacobian *= rho * rhohigh;

    double srcxhat = obsxhat + rho * cos(theta);
    double srcyhat = obsyhat + rho * sin(theta);
    // double xx = basiseval(obsxhat, obsyhat, {d.tri[0][0], d.tri[1][0], d.tri[2][0]});
    // double xy = basiseval(obsxhat, obsyhat, {d.tri[0][1], d.tri[1][1], d.tri[2][1]});
    double xx = obsxhat;
    double xy = obsyhat;
    double xz = -d.eps;

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

    auto K = Hkernel(d, Dx, Dy, Dz, nx, ny, nz, lx, ly, lz);

    auto obsbasis = basis(obsxhat, obsyhat);
    auto srcbasis = basis(srcxhat, srcyhat);

    for (int b1 = 0; b1 < 3; b1++) {
        for (int b2 = 0; b2 < 3; b2++) {
            auto basisprod = obsbasis[b1] * srcbasis[b2];
            for (int d1 = 0; d1 < 3; d1++) {
                for (int d2 = 0; d2 < 3; d2++) {
                    auto idx = b1 * 27 + d1 * 9 + b2 * 3 + d2;
                    fval[idx] = K[d1][d2] * basisprod * jacobian;
                }
            }
        }
    }
    return 0; 
}

std::array<double,81> integrate(Data& d) {
    d.evals = 0;
    std::array<double,81> sum{};
    for (int piece = 0; piece < 3; piece++) {
        d.set_piece(piece);
        const double xmin[4] = {0,0,0,0};
        const double xmax[4] = {1,1,1,1};
        double val[81], err[81];
        d.timer.start();
        pcubature(81, f, &d, 4, xmin, xmax, 0, 0, 1e-1, ERROR_INDIVIDUAL, val, err);
        d.timer.stop();
        for (int i = 0; i < 81; i++) {
            sum[i] += val[i];
        }
        std::cout << "internal: " << d.timer.get_time() << std::endl;
    }
    return sum;
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

    // These numbers are from ablims.py
    auto minlegalA = 0.100201758858;
    auto minlegalB = 0.24132345693;
    auto maxlegalB = 0.860758641203;
    auto Avals = chebab(minlegalA, 0.5, nA); 
    auto Bvals = chebab(minlegalB, maxlegalB, nB); 
    for (int i = 0; i < nA; i++) {
        double A = Avals[i];
        for (int j = 0; j < nB; j++) {
            double B = Bvals[j];
            Data d(Tri{{{0,0,0},{1,0,0},{A,B,0}}}, 0.1, 1.0, 0.25);
            auto sum = integrate(d);
            std::cout << i << ", " << j << ", " 
                << A << ", " << B << ", " << sum[0] << ", " << std::endl;
        }
    }
}

int main() {
    Data d(Tri{{{0,0,0},{1,0,0},{0,1,0}}}, 0.001, 1.0, 0.25);
    auto sum = integrate(d);
    for(int i = 0; i < 81; i++) {
        std::cout << sum[i] << ", " << std::endl;
    }
}

// correct result for H
// -0.0681271, 0.00374717, -2.92972e-08, -0.0479374, 0.000289359, -0.0310311, -0.0273078, 0.000289343, -0.0157399, 0.00374717, -0.0681271, 1.33654e-10, 0.000289359, -0.0273078, -0.0157399, 0.000289343, -0.0479374, -0.0310311, -2.92972e-08, 1.33654e-10, -0.253279, -0.0310311, -0.0157399, -0.0988838, -0.0157399, -0.0310311, -0.0988839, -0.0479374, 0.00028935, 0.0310311, -0.0773209, 0.0222243, 6.62088e-07, -0.0347615, 0.0145934, 0.0148439, 0.00028935, -0.0273078, 0.01574, 0.0222243, -0.0499281, -1.0635e-06, 0.0145934, -0.0347614, -0.014844, 0.0310311, 0.01574, -0.0988838, 6.62088e-07, -1.0635e-06, -0.25622, 0.0148439, -0.014844, -0.108641, -0.0273078, 0.000289349, 0.0157399, -0.0347616, 0.0145936, -0.0148446, -0.049928, 0.0222245, -1.30437e-06, 0.000289349, -0.0479374, 0.0310311, 0.0145936, -0.0347616, 0.0148445, 0.0222245, -0.0773207, 1.04316e-06, 0.0157399, 0.0310311, -0.0988839, -0.0148446, 0.0148445, -0.108641, -1.30437e-06, 1.04316e-06, -0.256222, 

// correct result for U
// 0.00627743, -0.00010931, 8.61569e-13, 0.00537344, -8.73445e-05, 0.000245357, 0.00502987, -8.73445e-05, 6.91567e-05, -0.00010931, 0.00627743, 6.53942e-13, -8.73445e-05, 0.00502987, 6.91567e-05, -8.73445e-05, 0.00537344, 0.000245357, 8.61569e-13, 6.53942e-13, 0.00602593, 0.000245357, 6.91567e-05, 0.00478726, 6.91567e-05, 0.000245357, 0.00478726, 0.00537344, -8.73445e-05, -0.000245357, 0.00617234, -0.000251347, -5.2341e-13, 0.00484596, -0.000324509, -0.000149677, -8.73445e-05, 0.00502987, -6.91567e-05, -0.000251347, 0.00596753, 2.48253e-13, -0.000324509, 0.00484596, 0.000149677, -0.000245357, -6.91567e-05, 0.00478726, -5.2341e-13, 2.48253e-13, 0.00581716, -0.000149677, 0.000149677, 0.00445606, 0.00502987, -8.73445e-05, -6.91567e-05, 0.00484596, -0.000324509, 0.000149677, 0.00596753, -0.000251347, 7.19921e-14, -8.73445e-05, 0.00537344, -0.000245357, -0.000324509, 0.00484596, -0.000149677, -0.000251347, 0.00617234, -2.4481e-13, -6.91567e-05, -0.000245357, 0.00478726, 0.000149677, -0.000149677, 0.00445606, 7.19921e-14, -2.4481e-13, 0.00581716,
    
