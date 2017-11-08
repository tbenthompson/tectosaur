#pragma once
#include <pybind11/pybind11.h>
#include "../include/vec_tensor.hpp"

Vec3 get_split_pt(const Tensor3& tri);
Vec2 xyhat_from_pt(const Vec3& pt, const Tensor3& tri);
bool check_xyhat(const Vec2& xyhat);
struct SeparateTriResult {
    std::array<Vec3,6> pts;
    std::array<std::array<int,3>,3> obs_tri;
    std::array<std::array<int,3>,3> src_tri;
    std::array<std::array<Vec2,3>,3> obs_basis_tri;
    std::array<std::array<Vec2,3>,3> src_basis_tri;
};
SeparateTriResult separate_tris(const Tensor3& obs_tri, const Tensor3& src_tri);
