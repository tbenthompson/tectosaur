<% 
setup_pybind11(cfg)
cfg['dependencies'] = ['lib/pybind11_nparray.hpp']

import tectosaur.util.kernel_exprs
kernel_names = ['U', 'T', 'A', 'H']
kernels = tectosaur.util.kernel_exprs.get_kernels('float')
def dim_name(dim):
    return ['x', 'y', 'z'][dim]
%>
#include <pybind11/pybind11.h>
#include "lib/pybind11_nparray.hpp"

using FloatT2d = std::array<std::array<float,3>,3>;
using FloatT4d = std::array<std::array<std::array<std::array<float,3>,3>,3>,3>;

std::array<float,3> cross(std::array<float,3> x, std::array<float,3> y) {
    std::array<float,3> out;
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
    return out;
}

std::array<float,3> sub(std::array<float,3> x, std::array<float,3> y) {
    std::array<float,3> out;
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
    return out;
}

std::array<float,3> get_unscaled_normal(FloatT2d tri, float out[3]) {
    auto s20 = sub(tri[2], tri[0]);
    auto s21 = sub(tri[2], tri[1]);
    return cross(s20, s21);
}

float magnitude(float v[3]) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

<%def name="get_triangle(name, tris, index)">
FloatT2d ${name}{};
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
float ${normal_prefix}${dim_name(dim)} = 
    ${prefix}_unscaled_normal[${dim}] / ${prefix}_normal_length;
% endfor
</%def>

<%def name="basis(prefix)">
float ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
float ${prefix}b1 = ${prefix}xhat;
float ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name)">
% for dim in range(3):
float ${pt_pfx}${dim_name(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dim_name(dim)} += ${basis_pfx}b${basis} * ${tri_name}[${basis}][${dim}];
% endfor
% endfor
</%def>

FloatT4d taylor_integralsH_pair(int n_obs_quad, float* obs_quad_pts,
    float* obs_quad_wts, float* obs_dir, 
    int n_src_quad, float* src_quad_pts, float* src_quad_wts,
    FloatT2d obs_tri, FloatT2d src_tri, float G, float nu) 
{
    ${tri_info("obs", "n")}
    ${tri_info("src", "l")}

    FloatT4d out{};
    for (int obs_idx = 0; obs_idx < n_obs_quad; obs_idx++) {
        float obsxhat = obs_quad_pts[obs_idx * 2 + 0];
        float obsyhat = obs_quad_pts[obs_idx * 2 + 1];
        float obsw = obs_quad_wts[obs_idx];

        ${basis("obs")}
        ${pts_from_basis("x", "obs", "obs_tri")}

        //TODO: offset points!

        for (int src_idx = 0; src_idx < n_src_quad; src_idx++) {
            float srcxhat = src_quad_pts[src_idx * 2 + 0];
            float srcyhat = src_quad_pts[src_idx * 2 + 1];
            float srcw = src_quad_wts[src_idx];

            ${basis("src")}
            ${pts_from_basis("y", "src", "src_tri")}

            //TODO: Check for singularity and return zero.
            % for d_obs in range(3):
            % for d_src in range(3):
            {
                float kernel_val = obs_jacobian * src_jacobian * obsw * srcw * 
                    (${kernels['H']['expr'][d_obs][d_src]});
                % for b_obs in range(3):
                % for b_src in range(3):
                out[${b_obs}][${d_obs}][${b_src}][${d_src}] =
                    obsb${b_obs} * srcb${b_src} * kernel_val;
                % endfor
                % endfor
            }
            % endfor
            % endfor
        }
    }
    return out;
}

void taylor_integralsH(float* result,
    int n_obs_quad, float* obs_quad_pts, float* obs_quad_wts, float* obs_dir,
    int n_src_quad, float* src_quad_pts, float* src_quad_wts, 
    int n_tri_pairs, float* pts, int* tris, int* obs_tri_idxs, int* src_tri_idxs,
    float G, float nu) 
{
    for (int i = 0; i < n_tri_pairs; i++) {

        auto obs_t_idx = obs_tri_idxs[i];
        auto src_t_idx = src_tri_idxs[i];

        ${get_triangle("obs_tri", "tris", "obs_t_idx")}
        ${get_triangle("src_tri", "tris", "src_t_idx")}

        auto temp_result = taylor_integralsH_pair(
            n_obs_quad, obs_quad_pts, obs_quad_wts,
            &obs_dir[obs_t_idx * n_obs_quad * 3],
            n_src_quad, src_quad_pts, src_quad_wts,
            obs_tri, src_tri, G, nu
        );
        float* temp_result_ptr = reinterpret_cast<float*>(&temp_result);
        for (int j = 0; j < 81; j++) {
            result[i * 81 + j] = temp_result_ptr[j];
        }
    }
}

PYBIND11_PLUGIN(taylor_integrals) {
    pybind11::module m("taylor_integrals");

    m.def("taylor_integralsH", 
        [] (
            NPArrayF obs_quad_pts, NPArrayF obs_quad_wts, NPArrayF obs_dir,
            NPArrayF src_quad_pts, NPArrayF src_quad_wts, 
            NPArrayF pts, NPArrayI tris, NPArrayI obs_tris, NPArrayI src_tris,
            float G, float nu) 
        {
            auto result = make_array<float>({obs_tris.request().shape[0] * 81});
            taylor_integralsH(
                as_ptr<float>(result),
                obs_quad_pts.request().shape[0],
                as_ptr<float>(obs_quad_pts),
                as_ptr<float>(obs_quad_wts),
                as_ptr<float>(obs_dir),
                src_quad_pts.request().shape[0],
                as_ptr<float>(src_quad_pts),
                as_ptr<float>(src_quad_wts),
                obs_tris.request().shape[0],
                as_ptr<float>(pts),
                as_ptr<int>(tris),
                as_ptr<int>(obs_tris),
                as_ptr<int>(src_tris),
                G, nu
            );
            return result;
        });
    return m.ptr();
}