<%
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-Wall'])
cfg['sources'] = ['cubature/hcubature.c', 'cubature/pcubature.c']
%>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cmath>
#include <map>
#include <vector>
#include <functional>
#include <chrono>
#include "cubature/cubature.h"

double co_thetalim0(double x, double y) { return M_PI - atan((1 - y) / x); }
double co_thetalim1(double x, double y) { return M_PI + atan(y / x); }
double co_thetalim2a(double x, double y) { return 2 * M_PI - atan(y / (1 - x)); }
double co_thetalim2b(double x, double y) { return - atan(y / (1 - x)); }
double co_rholim0(double x, double y, double t) { return (1 - y - x) / (cos(t) + sin(t)); }
double co_rholim1(double x, double y, double t) { return -x / cos(t); }
double co_rholim2(double x, double y, double t) { return -y / sin(t); }

double adj_thetalim0(double x) { return 0; }
double adj_thetalim1(double x) { return M_PI - atan(1 / (1 - x)); }
double adj_thetalim2(double x) { return M_PI; }
double adj_rholim0(double x, double t) { return x / (cos(t) + sin(t)); }
double adj_rholim1(double x, double t) { return -(1 - x) / cos(t); }

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

struct Tri {
    std::array<std::array<double,3>,3> pts;
    std::array<double,3> normal;
    double size;

    explicit Tri(std::array<std::array<double,3>,3> pts):
        pts(pts)
    {
        auto unscaled_normal = get_unscaled_normal(pts); 
        size = magnitude(unscaled_normal);
        for (int d = 0; d < 3; d++) {
            normal[d] = unscaled_normal[d] / size; 
        }
    }
};

struct Material {
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

    Material(double G, double nu):
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
    {}
};

using Kernel = std::function<
    std::array<std::array<double,3>,3>(
        Material&,double,double,double,double,double,double,double,double,double
    )>;

struct Data {
    double tol;
    decltype(&pcubature) integrator;

    Kernel K;
    int b1; 
    int b2;

    std::vector<double> rho_hats;
    std::vector<double> rho_wts;

    int evals;

    int piece;
    std::function<double(double,double)> co_theta_low;
    std::function<double(double,double)> co_theta_high;
    std::function<double(double,double,double)> co_rhohigh;

    std::function<double(double)> adj_theta_low;
    std::function<double(double)> adj_theta_high;
    std::function<double(double,double)> adj_rhohigh;

    Tri obs_tri;
    Tri src_tri;

    double eps;

    Material material;

    Data(double tol, bool p, Kernel K, 
            Tri obs_tri, Tri src_tri, 
            double eps, double G, double nu,
            std::vector<double> rho_hats, std::vector<double> rho_wts):
        tol(tol),
        integrator((p) ? &pcubature : &hcubature),
        K(K),
        rho_hats(rho_hats),
        rho_wts(rho_wts),
        obs_tri(obs_tri),
        src_tri(src_tri),
        eps(eps),
        material(G, nu)
    {
        set_piece(0);
    }

    void set_piece(int piece) {
        this->piece = piece;
        if (piece == 0) {
            co_theta_low = co_thetalim0; co_theta_high = co_thetalim1; co_rhohigh = co_rholim1;
            adj_theta_low = adj_thetalim0;
            adj_theta_high = adj_thetalim1;
            adj_rhohigh = adj_rholim0;
        } else if (piece == 1) {
            co_theta_low = co_thetalim1; co_theta_high = co_thetalim2a; co_rhohigh = co_rholim2;
            adj_theta_low = adj_thetalim1;
            adj_theta_high = adj_thetalim2;
            adj_rhohigh = adj_rholim1;
        } else if (piece == 2) {
            co_theta_low = co_thetalim2b; co_theta_high = co_thetalim0; co_rhohigh = co_rholim0;
        }
    }

    void set_basis(int b1, int b2) {
        this->b1 = b1;
        this->b2 = b2;
    }
};

std::array<std::array<double,3>,3> Ukernel(Material& d, double Dx, double Dy, double Dz,
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


std::array<std::array<double,3>,3> Tkernel(Material& d, double Dx, double Dy, double Dz,
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

std::array<std::array<double,3>,3> Akernel(Material& d, double Dx, double Dy, double Dz,
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

std::array<std::array<double,3>,3> Hkernel(Material& d, double Dx, double Dy, double Dz,
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


std::array<double,3> center(const Tri& t) {
    return {
        (t.pts[0][0] + t.pts[1][0] + t.pts[2][0]) / 3.0,
        (t.pts[0][1] + t.pts[1][1] + t.pts[2][1]) / 3.0,
        (t.pts[0][2] + t.pts[1][2] + t.pts[2][2]) / 3.0
    };
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

std::array<double,9> bem_integrand(Data& d, double obsxhat, double obsyhat,
    double srcxhat, double srcyhat) 
{
    double jacobian = 1.0;
    auto obspt = ref_to_real(obsxhat, obsyhat, d.obs_tri.pts);
    double xx = obspt[0];
    double xy = obspt[1];
    double xz = obspt[2];

    jacobian *= d.obs_tri.size;
    double nx = d.obs_tri.normal[0];
    double ny = d.obs_tri.normal[1];
    double nz = d.obs_tri.normal[2];

    auto sqrt_obsnL = sqrt(d.obs_tri.size);
    xx -= d.eps * nx * sqrt_obsnL;
    xy -= d.eps * ny * sqrt_obsnL;
    xz -= d.eps * nz * sqrt_obsnL;

    auto srcpt = ref_to_real(srcxhat, srcyhat, d.src_tri.pts);
    double yx = srcpt[0];
    double yy = srcpt[1];
    double yz = srcpt[2];

    jacobian *= d.src_tri.size;
    double lx = d.src_tri.normal[0];
    double ly = d.src_tri.normal[1];
    double lz = d.src_tri.normal[2];


    double Dx = yx - xx;
    double Dy = yy - xy;
    double Dz = yz - xz;

    auto K = d.K(d.material, Dx, Dy, Dz, nx, ny, nz, lx, ly, lz);

    auto obsbasis = basis(obsxhat, obsyhat);
    auto srcbasis = basis(srcxhat, srcyhat);

    auto basisprod = obsbasis[d.b1] * srcbasis[d.b2];
    std::array<double,9> out;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            out[d1 * 3 + d2] = K[d1][d2] * basisprod * jacobian;
        }
    }
    return out;
}

int f_coincident(unsigned ndim, const double* x, void* fdata, unsigned fdim, double* fval) 
{
    Data& d = *reinterpret_cast<Data*>(fdata);
    d.evals += 1;
    
    for (int idx = 0; idx < 9; idx++) {
        fval[idx] = 0.0;
    }

    // std::cout << d.evals << " " << npts << std::endl;
    double obsxhat = x[0];
    double obsyhat = x[1] * (1 - obsxhat);

    double thetahat = x[2];
    double thetalow = d.co_theta_low(obsxhat, obsyhat);
    double thetahigh = d.co_theta_high(obsxhat, obsyhat);
    double theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

    double outer_jacobian = (1 - obsxhat) * (thetahigh - thetalow) * 0.5;
    double costheta = cos(theta);
    double sintheta = sin(theta);

    for (size_t ri = 0; ri < d.rho_hats.size(); ri++) {
        double rhohat = (d.rho_hats[ri] + 1) / 2.0;
        double jacobian = d.rho_wts[ri] * outer_jacobian;

        double rhohigh = d.co_rhohigh(obsxhat, obsyhat, theta);
        double rho = rhohat * rhohigh;
        jacobian *= rho * rhohigh;

        double srcxhat = obsxhat + rho * costheta;
        double srcyhat = obsyhat + rho * sintheta;

        auto out = bem_integrand(d, obsxhat, obsyhat, srcxhat, srcyhat);
        for (int i = 0; i < 9; i++) {
            fval[i] += jacobian * out[i];
        }
    }

    return 0; 
}

int f_adjacent(unsigned ndim, const double* x, void* fdata, unsigned fdim, double* fval) {
    Data& d = *reinterpret_cast<Data*>(fdata);
    d.evals += 1;

    for (int idx = 0; idx < 9; idx++) {
        fval[idx] = 0.0;
    }

    if (d.piece > 1) {
        return 0;
    }

    double obsxhat = x[0];
    double obsyhat = x[1] * (1 - obsxhat);

    double thetahat = x[2];
    double thetalow = d.adj_theta_low(obsxhat);
    double thetahigh = d.adj_theta_high(obsxhat);
    double theta = (1 - thetahat) * thetalow + thetahat * thetahigh;

    double outer_jacobian = (1 - obsxhat) * (thetahigh - thetalow) * 0.5;
    double costheta = cos(theta);
    double sintheta = sin(theta);

    for (size_t ri = 0; ri < d.rho_hats.size(); ri++) {
        double rhohat = (d.rho_hats[ri] + 1) / 2.0;
        double jacobian = d.rho_wts[ri] * outer_jacobian;

        double rhohigh = d.adj_rhohigh(obsxhat, theta);
        double rho = rhohat * rhohigh;
        jacobian *= rho * rhohigh;

        double srcxhat = rho * costheta + (1 - obsxhat);
        double srcyhat = rho * sintheta;

        auto out = bem_integrand(d, obsxhat, obsyhat, srcxhat, srcyhat);
        for (int i = 0; i < 9; i++) {
            fval[i] += jacobian * out[i];
        }
    }
    return 0;
}

struct InteriorData {
    std::array<double,3> obspt;
    std::array<double,3> obsn;
    Tri src_tri;
    double tol;
    Kernel K;
    Material m;
};

int f_interior(unsigned ndim, const double* x, void* fdata, unsigned fdim, double* fval) {
    InteriorData& d = *reinterpret_cast<InteriorData*>(fdata);

    auto srcxhat = x[0];
    auto srcyhat = x[1] * (1 - srcxhat);
    auto jacobian = 1 - srcxhat;

    auto srcpt = ref_to_real(srcxhat, srcyhat, d.src_tri.pts);
    double yx = srcpt[0];
    double yy = srcpt[1];
    double yz = srcpt[2];

    jacobian *= d.src_tri.size;
    double lx = d.src_tri.normal[0];
    double ly = d.src_tri.normal[1];
    double lz = d.src_tri.normal[2];

    double Dx = yx - d.obspt[0];
    double Dy = yy - d.obspt[1];
    double Dz = yz - d.obspt[2];

    auto K = d.K(d.m, Dx, Dy, Dz, d.obsn[0], d.obsn[1], d.obsn[2], lx, ly, lz);

    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            fval[d1 * 3 + d2] = jacobian * K[d1][d2];
        }
    }
    return 0;
}

enum class Adjacency {
    Coincident,
    EdgeAdjacent,
    VertAdjacent,
    Separated
};

std::array<double,81> integrate(Data& d, Adjacency adj) {
    d.evals = 0;
    std::array<double,81> sum{};

    for (int b1 = 0; b1 < 3; b1++) {
        for (int b2 = 0; b2 < 3; b2++) {
            for (int piece = 0; piece < 3; piece++) {
                // std::cout << b1 << " " << b2 << " " << piece << std::endl;
                d.set_piece(piece);
                d.set_basis(b1, b2);
                double val[81], err[81];

                if (adj == Adjacency::Coincident) {
                    const double xmin[3] = {0,0,0};
                    const double xmax[3] = {1,1,1};
                    d.integrator(
                        9, f_coincident, &d, 3, xmin, xmax, 0, 0,
                        d.tol, ERROR_INDIVIDUAL, val, err
                    );
                } else if (adj == Adjacency::EdgeAdjacent) {
                    const double xmin[3] = {0,0,0};
                    const double xmax[3] = {1,1,1};
                    d.integrator(
                        9, f_adjacent, &d, 3, xmin, xmax, 0, 0,
                        d.tol, ERROR_INDIVIDUAL, val, err
                    );
                }

                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        sum[b1 * 27 + d1 * 9 + b2 * 3 + d2] += val[d1 * 3 + d2];
                    }
                }
            }
        }
    }

    return sum;
}

namespace py = pybind11;

Kernel get_kernel(std::string k_name) {
    std::map<std::string,Kernel> Ks;
    Ks["U"] = Ukernel;
    Ks["T"] = Tkernel;
    Ks["A"] = Akernel;
    Ks["H"] = Hkernel;
    return Ks[k_name];
}

PYBIND11_PLUGIN(adaptive_integrate) {
    py::module m("adaptive_integrate");

    m.def("integrate_coincident",
        [] (std::string k_name, std::array<std::array<double,3>,3> tri,
            double tol, double eps, double sm, double pr,
            std::vector<double> rho_hats, std::vector<double> rho_wts) 
        {
            Data d(
                tol, false, get_kernel(k_name), Tri(tri),
                Tri(tri), eps, sm, pr, rho_hats, rho_wts
            );
            auto result = integrate(d, Adjacency::Coincident);
            return result;
        }
    );

    m.def("integrate_adjacent",
        [] (std::string k_name, 
            std::array<std::array<double,3>,3> obs_tri,
            std::array<std::array<double,3>,3> src_tri,
            double tol, double eps, double sm, double pr,
            std::vector<double> rho_hats, std::vector<double> rho_wts) 
        {
            Data d(
                tol, false, get_kernel(k_name), 
                Tri(obs_tri), Tri(src_tri), 
                eps, sm, pr, rho_hats, rho_wts
            );
            auto result = integrate(d, Adjacency::EdgeAdjacent);
            return result;
        }
    );

    m.def("integrate_interior",
        [] (std::string k_name,
            std::array<double,3> obs_pt, std::array<double,3> obs_n,
            std::array<std::array<double,3>,3> src_tri,
            double tol, double sm, double pr)
        {
            std::array<double,9> val;
            std::array<double,9> err;
            double xmin[2] = {0,0};
            double xmax[2] = {1,1};
            InteriorData d{
                obs_pt, obs_n, Tri(src_tri), tol, get_kernel(k_name), Material(sm, pr)
            };
            hcubature(
                9, f_interior, &d, 2, xmin, xmax, 0, 0, tol,
                ERROR_INDIVIDUAL, val.data(), err.data()
            );
            return val;
        }
    );

    return m.ptr();
}
