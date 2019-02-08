/*cppimport
<%
import os
import tectosaur
setup_pybind11(cfg)
cfg['compiler_args'] += ['-std=c++14', '-O3', '-fopenmp']
cfg['linker_args'] += ['-fopenmp']
cfg['include_dirs'] += [os.path.join(tectosaur.source_dir, os.pardir)]
%>
*/

#include <utility>
#include <cmath>
#include <functional>
#include <iostream>
#include <cassert>

template <typename F, typename Fp>
std::pair<double,bool> newton(const F& f, const Fp& fp, double x0, double tol, int maxiter) {
    for (int i = 0; i < maxiter; i++) {
        double y = f(x0);
        double yp = fp(x0);
        double x1 = x0 - y / yp;
        // std::cout << x0 << " " << x1 << " " << y << " " << yp << x0 - x1 << std::endl;
        if (std::fabs(x1 - x0) <= tol * std::fabs(x0)) {
            return {x1, true};
        }
        x0 = x1;
    }
    return {x0, false}; 
}

double F(double V, double sigma_n, double state, double a, double V0, double C) {
    return a * sigma_n * std::asinh(V / (2 * V0) * std::exp(state / a)) - C;
}

//https://www.wolframalpha.com/input/?i=d%5Ba*S*arcsinh(x+%2F+(2*y)+*+exp(s%2Fa))%5D%2Fdx
double dFdV(double V, double sigma_n, double state, double a, double V0) {
    double expsa = std::exp(state / a);
    double Q = (V * expsa) / (2 * V0);
    return a * expsa * sigma_n / (2 * V0 * std::sqrt(1 + (Q * Q)));
}

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include "tectosaur/include/pybind11_nparray.hpp"
#include "tectosaur/include/vec_tensor.hpp"

namespace py = pybind11;

auto newton_py(std::function<double(double)> f,
        std::function<double(double)> fp,
        double x0, double tol, int maxiter) 
{
    return newton(f, fp, x0, tol, maxiter);
}

auto newton_rs(double tau_qs, double eta, double sigma_n,
        double state, double a, double V0, double C,
        double V_guess, double tol, int maxiter) 
{
    auto rsf = [&] (double V) { 
        return tau_qs - eta * V - F(V, sigma_n, state, a, V0, C); 
    };
    auto rsf_deriv = [&] (double V) { 
        return -eta - dFdV(V, sigma_n, state, a, V0); 
    };
    auto out = newton(rsf, rsf_deriv, V_guess, tol, maxiter);
    auto left = rsf(out.first * (1 - tol));
    auto right = rsf(out.first * (1 + tol));
    assert(left > out && right > out);
    return out;
}


Vec3 solve_for_dof_mag(const Vec3& traction_vec, double state, const Vec3& normal,
    const std::function<double(double, double)> rs_solver) 
{
    const double eps = 1e-14;
    auto normal_stress_vec = projection(traction_vec, normal);
    auto shear_traction_vec = sub(traction_vec, normal_stress_vec);

    double normal_mag = length(normal_stress_vec);
    double shear_mag = length(shear_traction_vec);
    if (shear_mag < eps) {
        return {0,0,0};
    }

    double V_mag = rs_solver(shear_mag, normal_mag);

    auto shear_dir = div(shear_traction_vec, shear_mag);
    return mult(shear_dir, V_mag);
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

Vec3 solve_for_dof_separate_dims(const Vec3& traction_vec,
        double state, const Vec3& normal,
        const std::function<double(double, double)> rs_solver) 
{
    Vec3 out;
    double normal_mag = 0.0;//length(normal_stress_vec);
    for (int d = 0; d < 3; d++) {
        auto t = traction_vec[d];
        out[d] = rs_solver(std::fabs(t), normal_mag) * sgn(t);
    }
    return out;
}

void rate_state_solver(NPArray<double> tri_normals, NPArray<double> traction,
        NPArray<double> state, NPArray<double> velocity, NPArray<double> a,
        double eta, double V0, double C,
        double additional_normal_stress,
        double tol, double maxiter, int basis_dim, bool separate_dims)
{
    auto* tri_normals_ptr = as_ptr<Vec3>(tri_normals);
    auto* state_ptr = as_ptr<double>(state);
    auto* velocity_ptr = as_ptr<Vec3>(velocity);
    auto* traction_ptr = as_ptr<Vec3>(traction);
    auto* a_ptr = as_ptr<double>(a);

    size_t n_tris = tri_normals.request().shape[0];

    #pragma omp parallel for
    for (size_t i = 0; i < n_tris; i++) {
        auto normal = tri_normals_ptr[i];
        for (int d = 0; d < basis_dim; d++) {

            size_t dof = i * basis_dim + d;
            auto traction_vec = traction_ptr[dof];
            auto state = state_ptr[dof];
            
            auto rs_solver_fnc = [&] (double shear_mag, double normal_mag) {
                auto solve_result = newton_rs(
                    shear_mag, eta, normal_mag + additional_normal_stress, 
                    state, a_ptr[dof], V0, C, 0.0, tol, maxiter
                );
                assert(solve_result.second);
                return solve_result.first;
            };

            Vec3 vel;
            if (separate_dims) {
                vel = solve_for_dof_separate_dims(
                    traction_vec, state, normal, rs_solver_fnc
                );
            } else {
                vel = solve_for_dof_mag(
                    traction_vec, state, normal, rs_solver_fnc
                );
            }
            for (int d2 = 0; d2 < 3; d2++) {
                velocity_ptr[dof][d2] = vel[d2];
            }
        }
    }
}

//TODO: should this be here?
void to_pts(size_t n_pts, NPArray<long> tris, NPArray<double> vec_dofs, 
    NPArray<double> vec_verts)
{
    size_t n_tris = tris.request().shape[0];
    auto* tris_ptr = as_ptr<long>(tris);
    auto* vec_dofs_ptr = as_ptr<Vec3>(vec_dofs);
    auto* vec_verts_ptr = as_ptr<Vec3>(vec_verts);
    
    for (size_t i = 0; i < n_tris; i++) {
        for (int v = 0; v < 3; v++) {
            size_t dof_idx = i * 3 + v;
            size_t vert_idx = tris_ptr[dof_idx];
            vec_verts_ptr[vert_idx] = add(vec_verts_ptr[vert_idx], vec_dofs_ptr[dof_idx]);
        }
    }
}

PYBIND11_MODULE(newton,m) {
    m.def("newton", &newton_py);
    m.def("newton_rs", &newton_rs);
    m.def("F", F);
    m.def("dFdV", dFdV);
    m.def("rate_state_solver", rate_state_solver);
    m.def("to_pts", to_pts);
}
