<% 
setup_pybind11(cfg)
cfg['dependencies'] = ['lib/pybind11_nparray.hpp']

import tectosaur.util.kernel_exprs
def dim_name(dim):
    return ['x', 'y', 'z'][dim]
%>
#include <pybind11/pybind11.h>
#include "lib/pybind11_nparray.hpp"
#include "taylor.hpp"

using RealT = float;
using RealT2d = std::array<std::array<RealT,3>,3>;
using RealT4d = std::array<std::array<std::array<std::array<RealT,3>,3>,3>,3>;

std::array<RealT,3> cross(std::array<RealT,3> x, std::array<RealT,3> y) {
    std::array<RealT,3> out;
    out[0] = x[1] * y[2] - x[2] * y[1];
    out[1] = x[2] * y[0] - x[0] * y[2];
    out[2] = x[0] * y[1] - x[1] * y[0];
    return out;
}

std::array<RealT,3> sub(std::array<RealT,3> x, std::array<RealT,3> y) {
    std::array<RealT,3> out;
    % for d in range(3):
    out[${d}] = x[${d}] - y[${d}];
    % endfor
    return out;
}

std::array<RealT,3> get_unscaled_normal(RealT2d tri) {
    auto s20 = sub(tri[2], tri[0]);
    auto s21 = sub(tri[2], tri[1]);
    return cross(s20, s21);
}

RealT magnitude(std::array<RealT,3> v) {
    return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

<%def name="get_triangle(name, tris, index)">
RealT2d ${name}{};
for (int c = 0; c < 3; c++) {
    for (int d = 0; d < 3; d++) {
        ${name}[c][d] = pts[3 * ${tris}[3 * ${index} + c] + d];
    }
}
</%def>

<%def name="tri_info(prefix,normal_prefix)">
auto ${prefix}_unscaled_normal = get_unscaled_normal(${prefix}_tri);
RealT ${prefix}_normal_length = magnitude(${prefix}_unscaled_normal);
RealT ${prefix}_jacobian = ${prefix}_normal_length;
% for dim in range(3):
RealT ${normal_prefix}${dim_name(dim)} = 
    ${prefix}_unscaled_normal[${dim}] / ${prefix}_normal_length;
% endfor
</%def>

<%def name="basis(prefix)">
RealT ${prefix}b0 = 1 - ${prefix}xhat - ${prefix}yhat;
RealT ${prefix}b1 = ${prefix}xhat;
RealT ${prefix}b2 = ${prefix}yhat;
</%def>

<%def name="pts_from_basis(pt_pfx,basis_pfx,tri_name)">
% for dim in range(3):
RealT ${pt_pfx}${dim_name(dim)} = 0;
% for basis in range(3):
${pt_pfx}${dim_name(dim)} += ${basis_pfx}b${basis} * ${tri_name}[${basis}][${dim}];
% endfor
% endfor
</%def>

<%def name="taylor_integrals(k_name)">
RealT4d taylor_integrals${k_name}_pair(int n_obs_quad, RealT* obs_quad_pts,
    RealT* obs_quad_wts, RealT* obs_dir, RealT offset,
    int n_src_quad, RealT* src_quad_pts, RealT* src_quad_wts,
    RealT2d obs_tri, RealT2d src_tri, RealT G, RealT nu) 
{
    ${tri_info("obs", "n")}
    ${tri_info("src", "l")}

    const static int taylor_order = 5;

    auto taylor_var = Tf<taylor_order>::var(1.0);

    RealT CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
    RealT CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
    RealT CsT0 = 1/(8*M_PI*(1-nu));
    RealT CsT1 = 1-2*nu;
    RealT CsH0 = G/(4*M_PI*(1-nu));
    RealT CsH1 = 1-2*nu;
    RealT CsH2 = -1+4*nu;
    RealT CsH3 = 3*nu;

    RealT4d out{};
    RealT4d kahanc{};
    for (int obs_idx = 0; obs_idx < n_obs_quad; obs_idx++) {
        RealT obsxhat = obs_quad_pts[obs_idx * 2 + 0];
        RealT obsyhat = obs_quad_pts[obs_idx * 2 + 1];
        RealT obsw = obs_quad_wts[obs_idx];

        ${basis("obs")}
        ${pts_from_basis("x_exp", "obs", "obs_tri")}

        % for dim in range(3):
        auto x${dim_name(dim)} = x_exp${dim_name(dim)} +
            obs_dir[obs_idx * 3 + ${dim}] * offset * taylor_var;
        % endfor

        for (int src_idx = 0; src_idx < n_src_quad; src_idx++) {
            RealT srcxhat = src_quad_pts[src_idx * 2 + 0];
            RealT srcyhat = src_quad_pts[src_idx * 2 + 1];
            RealT srcw = src_quad_wts[src_idx];

            ${basis("src")}
            ${pts_from_basis("y", "src", "src_tri")}

            auto Dx = yx - xx;
            auto Dy = yy - xy;
            auto Dz = yz - xz;
            auto r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            % if k_name is 'U':
                float invr = 1.0 / sqrt(r2);
                float Q1 = CsU0 * invr;
                float Q2 = CsU1 * invr / r2;
                float K00 = Q2*Dx*Dx + Q1;
                float K01 = Q2*Dx*Dy;
                float K02 = Q2*Dx*Dz;
                float K10 = Q2*Dy*Dx;
                float K11 = Q2*Dy*Dy + Q1;
                float K12 = Q2*Dy*Dz;
                float K20 = Q2*Dz*Dx;
                float K21 = Q2*Dz*Dy;
                float K22 = Q2*Dz*Dz + Q1;
            % elif k_name is 'T' or k_name is 'A':
                float invr = 1.0 / sqrt(r2);
                float invr2 = invr * invr;
                float rn = lx * Dx + ly * Dy + lz * Dz;
                float A = CsT0 * invr2;
                float drdn = (Dx * lx + Dy * ly + Dz * lz) * invr;
                % for k in range(3):
                    % for j in range(3):
                        float K${k}${j};
                        {
                            float term1 = (CsT1 * ${kronecker[k][j]} +
                                3 * D${dn(k)} * D${dn(j)} * invr2);
                            float term2 = CsT1 * (n${dn(j)} * D${dn(k)} - l${dn(k)} * D${dn(j)}) * invr;
                            % if k_name is 'T':
                                K${k}${j} = -A * (term1 * drdn - term2);
                            % else:
                                K${k}${j} = A * (term1 * drdn + term2);
                            % endif
                        }
                    % endfor
                % endfor

            % elif k_name is 'H':
                float invr = 1.0 / sqrt(r2);
                float invr2 = invr * invr;
                float invr3 = invr2 * invr;
                float Dorx = invr * Dx;
                float Dory = invr * Dy;
                float Dorz = invr * Dz;

                float rn = lx * Dorx + ly * Dory + lz * Dorz;
                float rm = nx * Dorx + ny * Dory + nz * Dorz;
                float mn = nx * lx + ny * ly + nz * lz;

                float Q = CsH0 * invr3;
                float A = Q * 3 * rn;
                float B = Q * CsH1;
                float C = Q * CsH3;

                float MTx = Q*CsH2*lx + A*CsH1*Dorx;
                float MTy = Q*CsH2*ly + A*CsH1*Dory;
                float MTz = Q*CsH2*lz + A*CsH1*Dorz;

                float NTx = B*nx + C*Dorx*rm;
                float NTy = B*ny + C*Dory*rm;
                float NTz = B*nz + C*Dorz*rm;

                float DTx = B*3*lx*rm + C*Dorx*mn + A*(nu*nx - 5*Dorx*rm);
                float DTy = B*3*ly*rm + C*Dory*mn + A*(nu*ny - 5*Dory*rm);
                float DTz = B*3*lz*rm + C*Dorz*mn + A*(nu*nz - 5*Dorz*rm);

                float ST = A*nu*rm + B*mn;

                float K00 = lx*NTx + nx*MTx + Dorx*DTx + ST;
                float K01 = lx*NTy + nx*MTy + Dorx*DTy;
                float K02 = lx*NTz + nx*MTz + Dorx*DTz;
                float K10 = ly*NTx + ny*MTx + Dory*DTx;
                float K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
                float K12 = ly*NTz + ny*MTz + Dory*DTz;
                float K20 = lz*NTx + nz*MTx + Dorz*DTx;
                float K21 = lz*NTy + nz*MTy + Dorz*DTy;
                float K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;
            % endif

            % for d_obs in range(3):
                % for d_src in range(3):
                    {
                        auto kernel_val = obs_jacobian * src_jacobian * obsw * srcw * 
                            K${d_obs}${d_src}.eval(-1.0);
                        % for b_obs in range(3):
                        % for b_src in range(3):
                        {
                        auto val = obsb${b_obs} * srcb${b_src} * kernel_val;
                        auto kahany = val - kahanc[${b_obs}][${d_obs}][${b_src}][${d_src}];
                        auto kahant = out[${b_obs}][${d_obs}][${b_src}][${d_src}] + kahany;
                        kahanc[${b_obs}][${d_obs}][${b_src}][${d_src}] =
                            (kahant - out[${b_obs}][${d_obs}][${b_src}][${d_src}]) - kahany;
                        out[${b_obs}][${d_obs}][${b_src}][${d_src}] = kahant;
                        }
                        % endfor
                        % endfor
                    }
                % endfor
            % endfor
        }
    }
    return out;
}

void taylor_integrals${k_name}(RealT* result,
    int n_obs_quad, RealT* obs_quad_pts, RealT* obs_quad_wts,
    RealT* obs_dir, RealT offset,
    int n_src_quad, RealT* src_quad_pts, RealT* src_quad_wts, 
    int n_tri_pairs, RealT* pts, int* tris, int* obs_tri_idxs, int* src_tri_idxs,
    RealT G, RealT nu) 
{
    for (int i = 0; i < n_tri_pairs; i++) {

        auto obs_t_idx = obs_tri_idxs[i];
        auto src_t_idx = src_tri_idxs[i];

        ${get_triangle("obs_tri", "tris", "obs_t_idx")}
        ${get_triangle("src_tri", "tris", "src_t_idx")}

        auto temp_result = taylor_integrals${k_name}_pair(
            n_obs_quad, obs_quad_pts, obs_quad_wts,
            &obs_dir[obs_t_idx * n_obs_quad * 3], offset,
            n_src_quad, src_quad_pts, src_quad_wts,
            obs_tri, src_tri, G, nu
        );
        RealT* temp_result_ptr = reinterpret_cast<RealT*>(&temp_result);
        for (int j = 0; j < 81; j++) {
            result[i * 81 + j] = temp_result_ptr[j];
        }
    }
}
<%/def>

% for k in ['U', 'T', 'A', 'H']:
${taylor_integrals(k)}
% endfor

PYBIND11_PLUGIN(taylor_integrals) {
    pybind11::module m("taylor_integrals");

    % for k in ['U', 'T', 'A', 'H']:
    m.def("taylor_integrals${k}", 
        [] (NPArray<RealT> obs_quad_pts, NPArray<RealT> obs_quad_wts,
            NPArray<RealT> obs_dir, RealT offset,
            NPArray<RealT> src_quad_pts, NPArray<RealT> src_quad_wts, 
            NPArray<RealT> pts, NPArrayI tris, NPArrayI obs_tris, NPArrayI src_tris,
            RealT G, RealT nu) 
        {
            auto result = make_array<RealT>({obs_tris.request().shape[0] * 81});
            taylor_integrals${k}(
                as_ptr<RealT>(result),
                obs_quad_pts.request().shape[0],
                as_ptr<RealT>(obs_quad_pts),
                as_ptr<RealT>(obs_quad_wts),
                as_ptr<RealT>(obs_dir),
                offset,
                src_quad_pts.request().shape[0],
                as_ptr<RealT>(src_quad_pts),
                as_ptr<RealT>(src_quad_wts),
                obs_tris.request().shape[0],
                as_ptr<RealT>(pts),
                as_ptr<int>(tris),
                as_ptr<int>(obs_tris),
                as_ptr<int>(src_tris),
                G, nu
            );
            return result;
        });
    % endfor
    return m.ptr();
}
