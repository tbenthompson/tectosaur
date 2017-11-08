<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'].extend([
    '../include/pybind11_nparray.hpp',
    '../include/vec_tensor.hpp',
    '../include/math_tools.hpp',
    '_standardize.hpp',
    'edge_adj_separate.hpp',
])

from tectosaur.nearfield.table_params import table_min_internal_angle,\
         minlegalA, minlegalB, maxlegalA, maxlegalB, min_intersect_angle

from tectosaur.kernels import kernels
%>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/pybind11_nparray.hpp"
#include "include/vec_tensor.hpp"
#include "include/math_tools.hpp"
#include "_standardize.hpp"
#include "edge_adj_separate.hpp"

//TODO: In general, this is some of the most disgusting code in tectosaur. This
// whole mess needs to be treated as some legacy crap and cleaned up. That said,
// it works... So don't break anything. Lots of golden master tests and slow 
// refactoring. Maybe not anytime soon...

namespace py = pybind11;

py::tuple coincident_lookup_pts(NPArray<double> tri_pts, double pr) {
    auto tri_pts_buf = tri_pts.request();
    auto* tri_pts_ptr = reinterpret_cast<double*>(tri_pts_buf.ptr);
    size_t n_tris = tri_pts_buf.shape[0];

    auto out = make_array<double>({n_tris, 3});
    auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);
    std::vector<StandardizeResult> standard_tris(n_tris);

#pragma omp parallel for
    for (size_t i = 0; i < n_tris; i++) {
        Tensor3 tri;
        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                tri[d1][d2] = tri_pts_ptr[i * 9 + d1 * 3 + d2];
            }
        }

        auto standard_tri_info = standardize(tri, ${table_min_internal_angle}, true);
        standard_tris[i] = standard_tri_info;

        double A = standard_tri_info.tri[2][0];
        double B = standard_tri_info.tri[2][1];

        double Ahat = from_interval(${minlegalA}, ${maxlegalA}, A);
        double Bhat = from_interval(${minlegalB}, ${maxlegalB}, B);
        double prhat = from_interval(0.0, 0.5, pr);

        out_ptr[i * 3] = Ahat;
        out_ptr[i * 3 + 1] = Bhat;
        out_ptr[i * 3 + 2] = prhat;
    }

    return py::make_tuple(out, standard_tris);
}

NPArray<double> coincident_lookup_from_standard(
    std::vector<StandardizeResult> standard_tris, 
    NPArray<double> interp_vals, NPArray<double> log_coeffs, std::string kernel, double sm)
{
    auto n_tris = standard_tris.size();
    auto out = make_array<double>({n_tris, 81});
    auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);

    auto* interp_vals_ptr = reinterpret_cast<double*>(interp_vals.request().ptr);
    auto* log_coeffs_ptr = reinterpret_cast<double*>(log_coeffs.request().ptr);

    auto kernel_props = get_kernel_props(kernel);

#pragma omp parallel for
    for (size_t i = 0; i < n_tris; i++) {
        auto log_standard_scale = log(sqrt(length(tri_normal(standard_tris[i].tri))));
        
        std::array<double,81> interp_vals_array;
        for (int j = 0; j < 81; j++) {
            interp_vals_array[j] = 
                interp_vals_ptr[i * 81 + j] + log_standard_scale * log_coeffs_ptr[i * 81 + j];
        }

        auto transformed = transform_from_standard(
            interp_vals_array, kernel_props, sm,
            standard_tris[i].labels, standard_tris[i].translation,
            standard_tris[i].R, standard_tris[i].scale
        );

        for (int j = 0; j < 81; j++) {
            out_ptr[i * 81 + j] = transformed[j];
        }
    }
    return out;
}

double get_adjacent_phi(const Tensor3& obs_tri, const Tensor3& src_tri) {
    auto p = sub(obs_tri[1], obs_tri[0]);
    auto L1 = sub(obs_tri[2], obs_tri[0]);
    auto L2 = sub(src_tri[2], src_tri[0]);
    auto T1 = sub(L1, projection(L1, p));
    auto T2 = sub(L2, projection(L2, p));

    auto n1 = tri_normal(obs_tri);
    n1 = div(n1, length(n1));

    auto samedir = dot(n1, sub(T2, T1)) > 0;
    auto phi = vec_angle(T1, T2);

    if (samedir) {
        return phi;
    } else {
        return 2 * M_PI - phi;
    }
}

std::array<std::array<int,3>,2> find_va_rotations(const std::array<int,3>& ot,
    const std::array<int,3>& st) 
{
    int ot_clicks = 0;
    int st_clicks = 0;
    bool matching_vert = false;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            matching_vert = st[d1] == ot[d2];
            if (matching_vert) {
                st_clicks = d1;
                ot_clicks = d2;
                break;
            }
        }
        if (matching_vert) {
            break;
        }
    }

    return {rotation_idxs<3>(ot_clicks), rotation_idxs<3>(st_clicks)};
}

struct VertexAdjacentSubTris {
    std::vector<std::array<double,3>> pts;
    std::vector<int> original_pair_idx;
    std::vector<std::array<size_t,3>> tris;
    std::vector<std::array<size_t,4>> pairs;
    std::vector<std::array<std::array<double,2>,3>> obs_basis;
    std::vector<std::array<std::array<double,2>,3>> src_basis;
};

struct EdgeAdjacentLookupTris {
    std::vector<std::array<double,2>> pts;
    std::vector<Tensor3> obs_tris;
    std::vector<int> obs_clicks;
    std::vector<int> src_clicks;
    std::vector<bool> src_flips;
    std::vector<std::array<std::array<double,2>,3>> obs_basis;
    std::vector<std::array<std::array<double,2>,3>> src_basis;

    EdgeAdjacentLookupTris(size_t n_tris):
        pts(n_tris),
        obs_tris(n_tris),
        obs_clicks(n_tris),
        src_clicks(n_tris),
        src_flips(n_tris),
        obs_basis(n_tris),
        src_basis(n_tris)
    {}
};

std::tuple<int,Tensor3,int,bool,Tensor3> orient_adj_tris(double* pts_ptr,
        long* tris_ptr, int tri_idx1, int tri_idx2)
{
    std::pair<int,int> pair1 = {-1,-1};
    std::pair<int,int> pair2 = {-1,-1};
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            if (tris_ptr[tri_idx1 * 3 + d1] != tris_ptr[tri_idx2 * 3 + d2]) {
                continue;
            }
            if (pair1.first == -1) {
                pair1 = {d1, d2};
            } else {
                pair2 = {d1, d2};
            }
        }
    }

    assert(pair1.first != -1);
    assert(pair1.second != -1);
    assert(pair2.first != -1);
    assert(pair2.second != -1);

    int obs_clicks = -1;
    if (positive_mod(pair1.first + 1, 3) == pair2.first) {
        obs_clicks = pair1.first;
    } else {
        obs_clicks = pair2.first;
        std::swap(pair1, pair2);
    }
    assert(obs_clicks != -1);

    int src_clicks = -1;
    bool src_flip = false;
    if (positive_mod(pair1.second + 1, 3) == pair2.second) {
        src_flip = true;
        src_clicks = pair1.second;
    } else {
        src_clicks = pair2.second;
    }
    assert(src_clicks != -1);

    auto obs_rot = rotation_idxs<3>(obs_clicks);
    auto src_rot = rotation_idxs<3>(src_clicks);
    if (src_flip) {
        std::swap(src_rot[0], src_rot[1]);
    }

    Tensor3 obs_tri;
    Tensor3 src_tri;
    for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            auto obs_v_idx = obs_rot[d1];
            auto src_v_idx = src_rot[d1];
            obs_tri[d1][d2] = pts_ptr[tris_ptr[tri_idx1 * 3 + obs_v_idx] * 3 + d2];
            src_tri[d1][d2] = pts_ptr[tris_ptr[tri_idx2 * 3 + src_v_idx] * 3 + d2];
        }
    }

    for (int d = 0; d < 3; d++) {
        assert(obs_tri[0][d] == src_tri[1][d]);
        assert(obs_tri[1][d] == src_tri[0][d]);
    }

    return std::make_tuple(obs_clicks, obs_tri, src_clicks, src_flip, src_tri);
}

py::tuple orient_adj_tris_shim(NPArray<double> pts, NPArray<long> tris,
    int tri_idx1, int tri_idx2)
{
    auto* pts_ptr = as_ptr<double>(pts);
    auto* tris_ptr = as_ptr<long>(tris);
    auto result = orient_adj_tris(pts_ptr, tris_ptr, tri_idx1, tri_idx2);
    return py::make_tuple(
        std::get<0>(result), std::get<1>(result),
        std::get<2>(result), std::get<3>(result),
        std::get<4>(result)
    );
}

py::tuple adjacent_lookup_pts(NPArray<double> pts, NPArray<long> tris,
    NPArray<long> ea_tri_indices, double pr, bool flip_symmetry) 
{
    size_t n_tris = ea_tri_indices.request().shape[0];

    VertexAdjacentSubTris va; 
    EdgeAdjacentLookupTris ea(n_tris);

    auto* pts_ptr = as_ptr<double>(pts);
    auto* tris_ptr = as_ptr<long>(tris);
    auto* ea_tri_indices_ptr = as_ptr<long>(ea_tri_indices);
    
    for (size_t i = 0; i < n_tris; i++) {
        auto tri_idx1 = ea_tri_indices_ptr[i * 2];
        auto tri_idx2 = ea_tri_indices_ptr[i * 2 + 1];
        auto oriented = orient_adj_tris(pts_ptr, tris_ptr, tri_idx1, tri_idx2);
        ea.obs_clicks[i] = std::get<0>(oriented);
        ea.obs_tris[i] = std::get<1>(oriented);
        ea.src_clicks[i] = std::get<2>(oriented);
        ea.src_flips[i] = std::get<3>(oriented);
        auto src_tri = std::get<4>(oriented);

        auto sep_res = separate_tris(ea.obs_tris[i], src_tri);

        ea.obs_basis[i] = sep_res.obs_basis_tri[0];
        ea.src_basis[i] = sep_res.src_basis_tri[0];

        for (size_t j = 0; j < sep_res.pts.size(); j++) {
            va.pts.push_back(sep_res.pts[j]);
        }
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                if (j == 0 && k == 0) {
                    continue;
                }

                auto otA = length(tri_normal({
                    sep_res.pts[sep_res.obs_tri[j][0]],
                    sep_res.pts[sep_res.obs_tri[j][1]],
                    sep_res.pts[sep_res.obs_tri[j][2]]
                }));

                auto stA = length(tri_normal({
                    sep_res.pts[sep_res.src_tri[k][0]],
                    sep_res.pts[sep_res.src_tri[k][1]],
                    sep_res.pts[sep_res.src_tri[k][2]]
                }));
                if (otA * stA < 1e-10) {
                    continue;
                }

                auto ot = sep_res.obs_tri[j];
                auto st = sep_res.src_tri[k];
                auto rot = find_va_rotations(ot, st);
                auto ot_rot = rot[0];
                auto st_rot = rot[1];

                va.original_pair_idx.push_back(i);
                va.pairs.push_back({va.tris.size(), va.tris.size() + 1, 0, 0});
                va.tris.push_back({
                    ot[ot_rot[0]] + 6 * i, ot[ot_rot[1]] + 6 * i, ot[ot_rot[2]] + 6 * i,
                });
                va.tris.push_back({
                    st[st_rot[0]] + 6 * i, st[st_rot[1]] + 6 * i, st[st_rot[2]] + 6 * i,
                });
                va.obs_basis.push_back({
                    sep_res.obs_basis_tri[j][ot_rot[0]],
                    sep_res.obs_basis_tri[j][ot_rot[1]],
                    sep_res.obs_basis_tri[j][ot_rot[2]],
                });
                va.src_basis.push_back({
                    sep_res.src_basis_tri[k][st_rot[0]],
                    sep_res.src_basis_tri[k][st_rot[1]],
                    sep_res.src_basis_tri[k][st_rot[2]],
                });
            }
        }

        auto phi = get_adjacent_phi(ea.obs_tris[i], src_tri);
        double phi_max = M_PI;
        if (flip_symmetry) {
            if (phi > M_PI) {
                phi = 2 * M_PI - phi;
            }
            assert(${min_intersect_angle} <= phi && phi <= M_PI);
        } else {
            phi_max = 2 * M_PI - ${min_intersect_angle};
            assert(${min_intersect_angle} <= phi && phi <= 2 * M_PI - ${min_intersect_angle});
        }

        auto phihat = from_interval(${min_intersect_angle}, phi_max, phi);
        auto prhat = from_interval(0, 0.5, pr); 
        ea.pts[i][0] = phihat;
        ea.pts[i][1] = prhat;
    }
    return py::make_tuple(va, ea);
}

double basis_fnc(double x, double y, int idx) {
    if (idx == 0) {
        return 1 - x - y;
    } else if (idx == 1) {
        return x;
    } else {
        return y;
    }
}

std::array<double,81> sub_basis(const std::array<double,81>& I,
        const std::array<std::array<double,2>,3>& obs_basis_tri,
        const std::array<std::array<double,2>,3>& src_basis_tri)
{
    std::array<double,81> out{};
    for (int ob1 = 0; ob1 < 3; ob1++) {
    for (int sb1 = 0; sb1 < 3; sb1++) {
    for (int ob2 = 0; ob2 < 3; ob2++) {
    for (int sb2 = 0; sb2 < 3; sb2++) {
        auto x = obs_basis_tri[ob2][0];
        auto y = obs_basis_tri[ob2][1];
        auto obv = basis_fnc(x, y, ob1);

        x = src_basis_tri[sb2][0];
        y = src_basis_tri[sb2][1];
        auto sbv = basis_fnc(x, y, sb1);

        (void)x;(void)y;

        for (int d1 = 0; d1 < 3; d1++) {
        for (int d2 = 0; d2 < 3; d2++) {
            auto out_idx = ob1 * 27 + d1 * 9 + sb1 * 3 + d2;
            auto in_idx = ob2 * 27 + d1 * 9 + sb2 * 3 + d2;
            out[out_idx] += I[in_idx] * obv * sbv;
        }
        }
    }
    }
    }
    }
    return out;
}

// TODO: Duplicated with derotate_adj_mat in fast_assembly.cpp, refactor, fix.
void derotate(int obs_clicks, int src_clicks, bool src_flip, double* out_start, double* in_start) {
    auto obs_derot = rotation_idxs<3>(-obs_clicks);
    auto src_derot = rotation_idxs<3>(-src_clicks);
    if (src_flip) {
        std::swap(src_derot[0], src_derot[1]);
    }
    for (int b1 = 0; b1 < 3; b1++) {
        for (int b2 = 0; b2 < 3; b2++) {
            for (int d1 = 0; d1 < 3; d1++) {
                for (int d2 = 0; d2 < 3; d2++) {
                    auto out_idx = b1 * 27 + d1 * 9 + b2 * 3 + d2;
                    auto in_idx = obs_derot[b1] * 27 + d1 * 9 + src_derot[b2] * 3 + d2;
                    auto val = in_start[in_idx];
                    if (src_flip) {
                        val *= -1;
                    }
#pragma omp atomic
                    out_start[out_idx] += val;
                }
            }
        }
    }
}

void vert_adj_subbasis(NPArray<double> out, NPArray<double> Iv,
    const VertexAdjacentSubTris& va, const EdgeAdjacentLookupTris& ea) 
{
    size_t n_integrals = Iv.request().shape[0];    
    auto Iv_ptr = as_ptr<double>(Iv);
    auto out_ptr = as_ptr<double>(out);
#pragma omp parallel for
    for (size_t i = 0; i < n_integrals; i++) {
        std::array<double,81> this_integral;
        for (int j = 0; j < 81; j++) {
            this_integral[j] = Iv_ptr[i * 81 + j];
        }
        auto res = sub_basis(this_integral, va.obs_basis[i], va.src_basis[i]);

        int out_idx = va.original_pair_idx[i];
        derotate(
            ea.obs_clicks[out_idx], ea.src_clicks[out_idx], ea.src_flips[out_idx],
            &out_ptr[out_idx * 81], res.data()
        );
    }
}

NPArray<double> adjacent_lookup_from_standard(
    NPArray<double> interp_vals, NPArray<double> log_coeffs,
    EdgeAdjacentLookupTris ea, std::string kernel, double sm)
{
    auto n_tris = ea.obs_tris.size();
    auto out = make_array<double>({n_tris, 81});
    auto* out_ptr = reinterpret_cast<double*>(out.request().ptr);

    auto* interp_vals_ptr = reinterpret_cast<double*>(interp_vals.request().ptr);
    auto* log_coeffs_ptr = reinterpret_cast<double*>(log_coeffs.request().ptr);

    auto kernel_props = get_kernel_props(kernel);

#pragma omp parallel for
    for (size_t i = 0; i < n_tris; i++) {
        for (size_t j = 0; j < 81; j++) {
            out_ptr[i * 81 + j] = 0.0;
        }

        Tensor3 tri = ea.obs_tris[i];
        auto standardized_res = standardize(tri, ${table_min_internal_angle}, false);

        auto log_standard_scale = log(sqrt(length(tri_normal(standardized_res.tri))));
        
        std::array<double,81> interp_vals_array;
        for (int j = 0; j < 81; j++) {
            interp_vals_array[j] = 
                interp_vals_ptr[i * 81 + j] + log_standard_scale * log_coeffs_ptr[i * 81 + j];
        }

        auto transformed = transform_from_standard(
            interp_vals_array, kernel_props, sm,
            standardized_res.labels, standardized_res.translation,
            standardized_res.R, standardized_res.scale
        );

        auto chunk = sub_basis(transformed, ea.obs_basis[i], ea.src_basis[i]);

        derotate(ea.obs_clicks[i], ea.src_clicks[i], ea.src_flips[i], &out_ptr[i * 81], chunk.data());
    }
    return out;
}

PYBIND11_MODULE(fast_lookup, m) {
    m.def("coincident_lookup_pts", coincident_lookup_pts);
    m.def("coincident_lookup_from_standard", coincident_lookup_from_standard);
    m.def("adjacent_lookup_from_standard", adjacent_lookup_from_standard);

    m.def("triangle_internal_angles", triangle_internal_angles);
    m.def("get_adjacent_phi", get_adjacent_phi);
    m.def("sub_basis", sub_basis);
    m.def("find_va_rotations", find_va_rotations);

    py::class_<VertexAdjacentSubTris>(m, "VertexAdjacentSubTris")
        .NPARRAYPROP(VertexAdjacentSubTris, pts)
        .NPARRAYPROP(VertexAdjacentSubTris, tris)
        .NPARRAYPROP(VertexAdjacentSubTris, pairs);

    py::class_<EdgeAdjacentLookupTris>(m, "EdgeAdjacentLookupTris")
        .NPARRAYPROP(EdgeAdjacentLookupTris, pts);

    m.def("orient_adj_tris", orient_adj_tris_shim);
    m.def("adjacent_lookup_pts", adjacent_lookup_pts);
    m.def("vert_adj_subbasis", vert_adj_subbasis);
}
