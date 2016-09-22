#include <cmath>
#include <map>
#include <vector>
#include <functional>
#include <chrono>
#include "cubature/cubature.h"

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

struct Data;
using Kernel = std::function<
    std::array<std::array<double,3>,3>(
        Data&,double,double,double,double,double,double,double,double,double
    )>;

struct Data {
    double tol;
    bool p;

    Kernel K;
    int b1; 
    int b2;

    int evals;
    std::function<double(double,double)> theta_low;
    std::function<double(double,double)> theta_high;
    std::function<double(double,double,double)> rhohigh;
    Tri obs_tri;
    Tri src_tri;
    double eps;
    double G;
    double nu;
    double CsU0;
    double CsU1;
    double CsT0;
    double CsT1;
    double CsH0;
    double CsH1;
    double CsH2;
    double CsH3;
    Timer timer;

    Data(double tol, bool p, Kernel K, 
            std::array<std::array<double,3>,3> tri, double eps, double G, double nu):
        tol(tol),
        p(p),
        K(K),
        obs_tri(tri),
        src_tri(tri),
        eps(eps),
        G(G),
        nu(nu),
        CsU0((3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu))),
        CsU1(1.0/(G*16.0*M_PI*(1.0-nu))),
        CsT0((1-2.0*nu)/(8.0*M_PI*(1.0-nu))),
        CsT1(3.0/(8.0*M_PI*(1.0-nu))),
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

    void set_basis(int b1, int b2) {
        this->b1 = b1;
        this->b2 = b2;
    }
};

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


std::array<std::array<double,3>,3> Tkernel(Data& d, double Dx, double Dy, double Dz,
    double nx, double ny, double nz, double lx, double ly, double lz)
{
    double r2 = Dx * Dx + Dy * Dy + Dz * Dz;
    float invr = 1.0 / sqrt(r2);
    float invr2 = invr * invr;
    float invr3 = invr2 * invr;

    float rn = lx * Dx + ly * Dy + lz * Dz;

    float A = +d.CsT0 * invr3;
    float C = -d.CsT1 * invr3 * invr2;

    float nxdy = lx*Dy-ly*Dx;
    float nzdx = lz*Dx-lx*Dz;
    float nzdy = lz*Dy-ly*Dz;

    float K00 = A * -rn                  + C*Dx*rn*Dx;
    float K01 = A * -nxdy + C*Dx*rn*Dy;
    float K02 = A * +nzdx + C*Dx*rn*Dz;
    float K10 = A * +nxdy + C*Dy*rn*Dx;
    float K11 = A * -rn                  + C*Dy*rn*Dy;
    float K12 = A * +nzdy + C*Dy*rn*Dz;
    float K20 = A * -nzdx + C*Dz*rn*Dx;
    float K21 = A * -nzdy + C*Dz*rn*Dy;
    float K22 = A * -rn                  + C*Dz*rn*Dz;
    return {{{K00,K01,K02},{K10,K11,K12},{K20,K21,K22}}};
}

std::array<std::array<double,3>,3> Akernel(Data& d, double Dx, double Dy, double Dz,
    double nx, double ny, double nz, double lx, double ly, double lz)
{
    double r2 = Dx * Dx + Dy * Dy + Dz * Dz;
    float invr = 1.0 / sqrt(r2);
    float invr2 = invr * invr;
    float invr3 = invr2 * invr;

    float rn = nx * Dx + ny * Dy + nz * Dz;

    float A = -d.CsT0 * invr3;
    float C = +d.CsT1 * invr3 * invr2;

    float nxdy = nx*Dy-ny*Dx;
    float nzdx = nz*Dx-nx*Dz;
    float nzdy = nz*Dy-ny*Dz;

    float K00 = A * -rn                  + C*Dx*rn*Dx;
    float K01 = A * +nxdy + C*Dx*rn*Dy;
    float K02 = A * -nzdx + C*Dx*rn*Dz;
    float K10 = A * -nxdy + C*Dy*rn*Dx;
    float K11 = A * -rn                  + C*Dy*rn*Dy;
    float K12 = A * -nzdy + C*Dy*rn*Dz;
    float K20 = A * +nzdx + C*Dz*rn*Dx;
    float K21 = A * +nzdy + C*Dz*rn*Dy;
    float K22 = A * -rn                  + C*Dz*rn*Dz;
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

double basiseval(double xhat, double yhat, std::array<double,3> vals) {
    return (1.0 - xhat - yhat) * vals[0] + xhat * vals[1] + yhat * vals[2];
}

std::array<double,3> basis(double xhat, double yhat) {
     return {1 - xhat - yhat, xhat, yhat};
}

std::array<double,3> cross(std::array<double,3> x, std::array<double,3> y) {
    return {
        x[1] * y[2] - x[2] * y[1],
        x[2] * y[0] - x[0] * y[2],
        x[0] * y[1] - x[1] * y[0]
    };
}

std::array<double,3> sub(std::array<double,3> x, std::array<double,3> y) {
    return {x[0] - y[0], x[1] - y[1], x[2] - y[2]};
}

double dot(std::array<double,3> x, std::array<double,3> y) {
    return x[0] * y[0] + x[1] * y[1] + x[2] * y[2];
}

std::array<double,3> get_unscaled_normal(std::array<std::array<double,3>,3> tri) {
    return cross(sub(tri[2], tri[0]), sub(tri[2], tri[1]));
}

double magnitude(std::array<double,3> v) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

std::array<double,3> ref_to_real(
    double xhat, double yhat, std::array<std::array<double,3>,3> tri) 
{
    return {
        dot(basis(xhat, yhat), {tri[0][0], tri[1][0], tri[2][0]}),
        dot(basis(xhat, yhat), {tri[0][1], tri[1][1], tri[2][1]}),
        dot(basis(xhat, yhat), {tri[0][2], tri[1][2], tri[2][2]})
    };
}

void all_zeros(double* fval) {
    for (int idx = 0; idx < 81; idx++) {
        fval[idx] = 0.0;
    }
}

int f(unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) 
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

    auto obspt = ref_to_real(obsxhat, obsyhat, d.obs_tri);
    double xx = obspt[0];
    double xy = obspt[1];
    double xz = obspt[2];

    auto obsn = get_unscaled_normal(d.obs_tri);
    auto obsnL = magnitude(obsn);
    jacobian *= obsnL;
    auto inv_obsnL = 1.0 / obsnL;
    double nx = obsn[0] * inv_obsnL;
    double ny = obsn[1] * inv_obsnL;
    double nz = obsn[2] * inv_obsnL;

    auto sqrt_obsnL = sqrt(obsnL);
    xx -= d.eps * nx * sqrt_obsnL;
    xy -= d.eps * ny * sqrt_obsnL;
    xz -= d.eps * nz * sqrt_obsnL;

    auto srcpt = ref_to_real(srcxhat, srcyhat, d.src_tri);
    double yx = srcpt[0];
    double yy = srcpt[1];
    double yz = srcpt[2];

    auto srcn = get_unscaled_normal(d.src_tri);
    auto srcnL = magnitude(srcn);
    jacobian *= srcnL;
    double lx = srcn[0] / srcnL;
    double ly = srcn[1] / srcnL;
    double lz = srcn[2] / srcnL;


    double Dx = yx - xx;
    double Dy = yy - xy;
    double Dz = yz - xz;

    auto K = d.K(d, Dx, Dy, Dz, nx, ny, nz, lx, ly, lz);

    auto obsbasis = basis(obsxhat, obsyhat);
    auto srcbasis = basis(srcxhat, srcyhat);

    auto basisprod = obsbasis[d.b1] * srcbasis[d.b2];
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            fval[d1 * 3 + d2] = K[d1][d2] * basisprod * jacobian;
        }
    }
    return 0; 
}

// void try_integrate_log() {
//     double val2[1],err2[1];
//     auto f2 = [] (unsigned ndim, const double *x, void *fdata, unsigned fdim, double *fval) {
//         fval[0] = std::log(x[0]);
//         return 0;
//     };
//     const double xmin2[1] = {0};
//     const double xmax2[1] = {1};
//     hcubature(1, f2, nullptr, 1, xmin2, xmax2, 0, 0, 1e-4, ERROR_INDIVIDUAL, val2, err2);
//     std::cout << val2[0] << " " << err2[0] << std::endl;
// }

std::array<double,81> integrate(Data& d) {
    d.evals = 0;
    std::array<double,81> sum{};


    for (int b1 = 0; b1 < 3; b1++) {
        for (int b2 = 0; b2 < 3; b2++) {
            for (int piece = 0; piece < 3; piece++) {
                // std::cout << piece << " " << b1 << " " << b2 << std::endl;
                d.set_piece(piece);
                d.set_basis(b1, b2);
                const double xmin[4] = {0,0,0,0};
                const double xmax[4] = {1,1,1,1};
                double val[81], err[81];
                d.timer.start();
                decltype(&pcubature) integrator;
                if (d.p) {
                    integrator = &pcubature;
                } else {
                    integrator = &hcubature;
                }
                integrator(9, f, &d, 4, xmin, xmax, 0, 0, d.tol, ERROR_INDIVIDUAL, val, err);
                d.timer.stop();
                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        sum[b1 * 27 + d1 * 9 + b2 * 3 + d2] += val[d1 * 3 + d2];
                    }
                }
                // std::cout << "piece " << piece << ": " << d.timer.get_time() << std::endl;
            }
        }
    }

    return sum;
}

