<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'].extend([
    '../include/pybind11_nparray.hpp',
    '../include/vec_tensor.hpp',
    '../include/math_tools.hpp',
    'edge_adj_setup.hpp',
])

from tectosaur.nearfield.table_params import min_angle_isoceles_height, min_intersect_angle
%>

#include "include/math_tools.hpp"
#include "include/pybind11_nparray.hpp"
#include "edge_adj_setup.hpp"
#include <pybind11/stl.h>

namespace py = pybind11;

Vec3 get_split_pt(const Tensor3& tri) {
    auto base_vec = sub(tri[1], tri[0]);
    auto midpt = add(mult(base_vec, 0.5), tri[0]);
    auto to2 = sub(tri[2], midpt);
    auto V = sub(to2, projection(to2, base_vec));
    auto Vlen = length(V);
    auto base_len = length(base_vec);
    auto V_scaled = mult(V, base_len * ${min_angle_isoceles_height} / Vlen);
    return add(midpt, V_scaled);
}

Vec2 cramers_rule(const Vec2& a, const Vec2& b, const Vec2& c) {
    auto denom = a[0] * b[1] - b[0] * a[1];
    return {
        (c[0] * b[1] - b[0] * c[1]) / denom,
        (c[1] * a[0] - a[1] * c[0]) / denom
    };
}

// Solve least squares for a 3x2 system using cramers rule and the normal eqtns.
Vec2 least_squares_helper(const Vec3& a, const Vec3& b, const Vec3& c) {
    auto ada = dot(a, a);
    auto adb = dot(a, b);
    auto bdb = dot(b, b);

    auto adc = dot(a, c);
    auto bdc = dot(b, c);

    return cramers_rule({ada, adb}, {adb, bdb}, {adc, bdc});
}

// TODO: Move this to a central geometry module
Vec2 xyhat_from_pt(const Vec3& pt, const Tensor3& tri) {
    auto v1 = sub(tri[1], tri[0]);
    auto v2 = sub(tri[2], tri[0]);
    auto pt_trans = sub(pt, tri[0]);
        
    // Use cramer's rule to solve for xhat, yhat
    return least_squares_helper(v1, v2, pt_trans);
}

bool check_xyhat(const Vec2& xyhat) {
    return (
        (xyhat[0] + xyhat[1] <= 1.0 + 1e-15) &&
        (xyhat[0] >= -1e-15) && 
        (xyhat[1] >= -1e-15)
    );
}

SeparateTriResult separate_tris(const Tensor3& obs_tri, const Tensor3& src_tri) {
    auto obs_split_pt = get_split_pt(obs_tri);
    auto obs_split_pt_xyhat = xyhat_from_pt(obs_split_pt, obs_tri);
    assert(check_xyhat(obs_split_pt_xyhat));

    auto src_split_pt = get_split_pt(src_tri);
    auto src_split_pt_xyhat = xyhat_from_pt(src_split_pt, src_tri);
    assert(check_xyhat(src_split_pt_xyhat));

    std::array<Vec3,6> pts = {
        obs_tri[0], obs_tri[1], obs_tri[2], src_tri[2], obs_split_pt, src_split_pt
    };
    std::array<std::array<int,3>,3> obs_tris = {{{0,1,4}, {4,1,2}, {0,4,2}}};
    std::array<std::array<int,3>,3> src_tris = {{{1,0,5}, {5,0,3}, {1,5,3}}};
    std::array<std::array<Vec2,3>,3> obs_basis_tris = {{
        {{{0,0},{1,0},obs_split_pt_xyhat}},
        {{obs_split_pt_xyhat, {1,0},{0,1}}},
        {{{0,0},obs_split_pt_xyhat,{0,1}}}
    }};
    std::array<std::array<Vec2,3>,3> src_basis_tris = {{
        {{{0,0},{1,0},src_split_pt_xyhat}},
        {{src_split_pt_xyhat, {1,0},{0,1}}},
        {{{0,0},src_split_pt_xyhat,{0,1}}}
    }};

    return {pts, obs_tris, src_tris, obs_basis_tris, src_basis_tris};
}

py::tuple separate_tris_pyshim(const Tensor3& obs_tri, const Tensor3& src_tri) {
    auto out = separate_tris(obs_tri, src_tri);
    return py::make_tuple(
        out.pts, out.obs_tri, out.src_tri, out.obs_basis_tri, out.src_basis_tri
    );
}

double calc_adjacent_phi(const Tensor3& obs_tri, const Tensor3& src_tri) {
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

double calc_adjacent_phihat(double phi, bool flip_symmetry) {
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

    return from_interval(${min_intersect_angle}, phi_max, phi);
}

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

PYBIND11_MODULE(edge_adj_setup, m) {
    m.def("get_split_pt", get_split_pt);
    m.def("xyhat_from_pt", xyhat_from_pt);
    m.def("check_xyhat", check_xyhat);
    m.def("separate_tris", separate_tris_pyshim);
    m.def("calc_adjacent_phi", calc_adjacent_phi);
    m.def("calc_adjacent_phihat", calc_adjacent_phihat);
    m.def("orient_adj_tris", orient_adj_tris_shim);
}
