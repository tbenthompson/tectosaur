#pragma once
#include <pybind11/pybind11.h>
#include "../include/vec_tensor.hpp"

Vec3 get_split_pt(const Tensor3& tri, double min_angle_isoceles_height);
Vec2 cramers_rule(const Vec2& a, const Vec2& b, const Vec2& c);
Vec2 least_squares_helper(const Vec3& a, const Vec3& b, const Vec3& c);
Vec2 xyhat_from_pt(const Vec3& pt, const Tensor3& tri);
bool check_xyhat(const Vec2& xyhat);
struct SeparateTriResult {
    std::array<Vec3,6> pts;
    std::array<std::array<int,3>,3> obs_tri;
    std::array<std::array<int,3>,3> src_tri;
    std::array<std::array<Vec2,3>,3> obs_basis_tri;
    std::array<std::array<Vec2,3>,3> src_basis_tri;
};
SeparateTriResult separate_tris(const Tensor3& obs_tri, const Tensor3& src_tri,
        double min_angle_isoceles_height);
pybind11::tuple separate_tris_pyshim(const Tensor3& obs_tri, const Tensor3& src_tri,
        double min_angle_isoceles_height);
