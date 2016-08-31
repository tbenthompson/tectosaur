<% 
setup_pybind11(cfg)
cfg['dependencies'] = ['lib/pybind11_nparray.hpp', 'taylor.hpp']
cfg['compiler_args'] += ['-Wno-unused-variable']

import tectosaur.util.kernel_exprs
def dim_name(dim):
    return ['x', 'y', 'z'][dim]
%>
#include <pybind11/pybind11.h>
#include "lib/pybind11_nparray.hpp"
#include "taylor.hpp"

using RealT = double;
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
RealT4d taylor_integrals${k_name}_pair(int n_quad, RealT* quad_pts, RealT* quad_wts,
    RealT offset, RealT2d obs_tri, RealT2d src_tri, RealT G, RealT nu) 
{
    ${tri_info("obs", "n")}
    ${tri_info("src", "l")}

    const static int taylor_order = 10;

    auto taylor_var = Taylor<RealT,taylor_order>::var(1.0);

    RealT CsU0 = (3.0-4.0*nu)/(G*16.0*M_PI*(1.0-nu));
    RealT CsU1 = 1.0/(G*16.0*M_PI*(1.0-nu));
    RealT CsT0 = (1-2.0*nu)/(8.0*M_PI*(1.0-nu));
    RealT CsT1 = 3.0/(8.0*M_PI*(1.0-nu));
    RealT CsH0 = G/(4*M_PI*(1-nu));
    RealT CsH1 = 1-2*nu;
    RealT CsH2 = -1+4*nu;
    RealT CsH3 = 3*nu;

    RealT4d out{};
    RealT4d kahanc{};
    for (int q_idx = 0; q_idx < n_quad; q_idx++) {
        RealT obsxhat = quad_pts[q_idx * 4 + 0];
        RealT obsyhat = quad_pts[q_idx * 4 + 1];
        RealT srcxhat = quad_pts[q_idx * 4 + 2];
        RealT srcyhat = quad_pts[q_idx * 4 + 3];
        RealT qw = obs_jacobian * src_jacobian * quad_wts[q_idx];

        ${basis("obs")}
        ${pts_from_basis("x_exp", "obs", "obs_tri")}

        ${basis("src")}
        ${pts_from_basis("y", "src", "src_tri")}

        % for dim in range(3):
        auto x${dim_name(dim)} = x_exp${dim_name(dim)} -
            offset * sqrt(obs_jacobian) * n${dim_name(dim)} * taylor_var;
        % endfor

        auto Dx = yx - xx;
        auto Dy = yy - xy;
        auto Dz = yz - xz;
        auto r2 = Dx * Dx + Dy * Dy + Dz * Dz;

        % if k_name is 'U':
            auto invr = 1.0 / sqrt(r2);
            auto Q1 = CsU0 * invr;
            auto Q2 = CsU1 * invr / r2;
            auto K00 = Q2*Dx*Dx + Q1;
            auto K01 = Q2*Dx*Dy;
            auto K02 = Q2*Dx*Dz;
            auto K10 = Q2*Dy*Dx;
            auto K11 = Q2*Dy*Dy + Q1;
            auto K12 = Q2*Dy*Dz;
            auto K20 = Q2*Dz*Dx;
            auto K21 = Q2*Dz*Dy;
            auto K22 = Q2*Dz*Dz + Q1;
        % elif k_name is 'T' or k_name is 'A':
            <%
                minus_or_plus = '-' if k_name is 'T' else '+'
                plus_or_minus = '+' if k_name is 'T' else '-'
                n_name = 'l' if k_name is 'T' else 'n'
            %>
            auto invr = 1.0 / sqrt(r2);
            auto invr2 = invr * invr;
            auto invr3 = invr2 * invr;

            auto rn = ${n_name}x * Dx + ${n_name}y * Dy + ${n_name}z * Dz;

            auto A = ${plus_or_minus}CsT0 * invr3;
            auto C = ${minus_or_plus}CsT1 * invr3 * invr2;

            auto nxdy = ${n_name}x*Dy-${n_name}y*Dx;
            auto nzdx = ${n_name}z*Dx-${n_name}x*Dz;
            auto nzdy = ${n_name}z*Dy-${n_name}y*Dz;

            auto K00 = A * -rn                  + C*Dx*rn*Dx;
            auto K01 = A * ${minus_or_plus}nxdy + C*Dx*rn*Dy;
            auto K02 = A * ${plus_or_minus}nzdx + C*Dx*rn*Dz;
            auto K10 = A * ${plus_or_minus}nxdy + C*Dy*rn*Dx;
            auto K11 = A * -rn                  + C*Dy*rn*Dy;
            auto K12 = A * ${plus_or_minus}nzdy + C*Dy*rn*Dz;
            auto K20 = A * ${minus_or_plus}nzdx + C*Dz*rn*Dx;
            auto K21 = A * ${minus_or_plus}nzdy + C*Dz*rn*Dy;
            auto K22 = A * -rn                  + C*Dz*rn*Dz;
        % elif k_name is 'H':
            auto invr = 1.0 / sqrt(r2);
            auto invr2 = invr * invr;
            auto invr3 = invr2 * invr;
            auto Dorx = invr * Dx;
            auto Dory = invr * Dy;
            auto Dorz = invr * Dz;

            auto rn = lx * Dorx + ly * Dory + lz * Dorz;
            auto rm = nx * Dorx + ny * Dory + nz * Dorz;
            auto mn = nx * lx + ny * ly + nz * lz;

            auto Q = CsH0 * invr3;
            auto A = Q * 3 * rn;
            auto B = Q * CsH1;
            auto C = Q * CsH3;

            auto MTx = Q*CsH2*lx + A*CsH1*Dorx;
            auto MTy = Q*CsH2*ly + A*CsH1*Dory;
            auto MTz = Q*CsH2*lz + A*CsH1*Dorz;

            auto NTx = B*nx + C*Dorx*rm;
            auto NTy = B*ny + C*Dory*rm;
            auto NTz = B*nz + C*Dorz*rm;

            auto DTx = B*3*lx*rm + C*Dorx*mn + A*(nu*nx - 5*Dorx*rm);
            auto DTy = B*3*ly*rm + C*Dory*mn + A*(nu*ny - 5*Dory*rm);
            auto DTz = B*3*lz*rm + C*Dorz*mn + A*(nu*nz - 5*Dorz*rm);

            auto ST = A*nu*rm + B*mn;

            auto K00 = lx*NTx + nx*MTx + Dorx*DTx + ST;
            auto K01 = lx*NTy + nx*MTy + Dorx*DTy;
            auto K02 = lx*NTz + nx*MTz + Dorx*DTz;
            auto K10 = ly*NTx + ny*MTx + Dory*DTx;
            auto K11 = ly*NTy + ny*MTy + Dory*DTy + ST;
            auto K12 = ly*NTz + ny*MTz + Dory*DTz;
            auto K20 = lz*NTx + nz*MTx + Dorz*DTx;
            auto K21 = lz*NTy + nz*MTy + Dorz*DTy;
            auto K22 = lz*NTz + nz*MTz + Dorz*DTz + ST;
        % endif

        % for d_obs in range(3):
            % for d_src in range(3):
                {
                    auto kernel_val = qw * K${d_obs}${d_src}.eval(-1.0);
                    % for b_obs in range(3):
                    % for b_src in range(3):
                    {
                    auto val = obsb${b_obs} * srcb${b_src} * kernel_val;
                    auto& this_out = out[${b_obs}][${d_obs}][${b_src}][${d_src}];
                    auto& this_kahanc = kahanc[${b_obs}][${d_obs}][${b_src}][${d_src}];
                    auto kahany = val - this_kahanc;
                    auto kahant = this_out + kahany;
                    this_kahanc = (kahant - this_out) - kahany;
                    this_out = kahant;
                    }
                    % endfor
                    % endfor
                }
            % endfor
        % endfor
    }
    return out;
}

void taylor_integrals${k_name}(RealT* result,
    int n_quad, RealT* quad_pts, RealT* quad_wts, RealT offset,
    int n_tri_pairs, RealT* pts, int* tris, int* obs_tri_idxs, int* src_tri_idxs,
    RealT G, RealT nu) 
{
    for (int i = 0; i < n_tri_pairs; i++) {

        auto obs_t_idx = obs_tri_idxs[i];
        auto src_t_idx = src_tri_idxs[i];

        ${get_triangle("obs_tri", "tris", "obs_t_idx")}
        ${get_triangle("src_tri", "tris", "src_t_idx")}

        auto temp_result = taylor_integrals${k_name}_pair(
            n_quad, quad_pts, quad_wts, offset, obs_tri, src_tri, G, nu
        );
        RealT* temp_result_ptr = reinterpret_cast<RealT*>(&temp_result);
        for (int j = 0; j < 81; j++) {
            result[i * 81 + j] = temp_result_ptr[j];
        }
    }
}
</%def>

% for k in ['U', 'T', 'A', 'H']:
${taylor_integrals(k)}
% endfor

PYBIND11_PLUGIN(taylor_integrals) {
    pybind11::module m("taylor_integrals");

    % for k in ['U', 'T', 'A', 'H']:
    m.def("taylor_integrals${k}", 
        [] (NPArray<RealT> quad_pts, NPArray<RealT> quad_wts, RealT offset,
            NPArray<RealT> pts, NPArrayI tris, NPArrayI obs_tris, NPArrayI src_tris,
            RealT G, RealT nu) 
        {
            auto result = make_array<RealT>({obs_tris.request().shape[0] * 81});
            taylor_integrals${k}(
                as_ptr<RealT>(result),
                quad_pts.request().shape[0],
                as_ptr<RealT>(quad_pts),
                as_ptr<RealT>(quad_wts),
                offset,
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
