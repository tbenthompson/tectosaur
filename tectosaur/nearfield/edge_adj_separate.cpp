<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'].extend([
    '../include/pybind11_nparray.hpp',
    '../include/vec_tensor.hpp',
    'edge_adj_separate.hpp',
])

from tectosaur.nearfield.table_params import min_angle_isoceles_height
%>

#include "edge_adj_separate.hpp"
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

PYBIND11_MODULE(edge_adj_separate, m) {
    m.def("get_split_pt", get_split_pt);
    m.def("xyhat_from_pt", xyhat_from_pt);
    m.def("separate_tris", separate_tris_pyshim);
}
