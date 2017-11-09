#pragma once
#include <pybind11/pybind11.h>
#include "../include/vec_tensor.hpp"

struct SeparateTriResult {
    std::array<Vec3,6> pts;
    std::array<std::array<int,3>,3> obs_tri;
    std::array<std::array<int,3>,3> src_tri;
    std::array<std::array<Vec2,3>,3> obs_basis_tri;
    std::array<std::array<Vec2,3>,3> src_basis_tri;
};
SeparateTriResult separate_tris(const Tensor3& obs_tri, const Tensor3& src_tri);

double calc_adjacent_phi(const Tensor3& obs_tri, const Tensor3& src_tri);
double calc_adjacent_phihat(double phi, bool flip_symmetry);
std::tuple<int,Tensor3,int,bool,Tensor3> orient_adj_tris(double* pts_ptr,
        long* tris_ptr, int tri_idx1, int tri_idx2);
