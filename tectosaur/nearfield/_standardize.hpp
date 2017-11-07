#pragma once
#include <pybind11/pybind11.h>
#include "include/vec_tensor.hpp"

Vec3 get_edge_lens(const Tensor3& tri);
int get_longest_edge(const Vec3& lens);
int get_origin_vertex(const Vec3& lens);
std::pair<Tensor3,std::array<int,3>> relabel(
    const Tensor3& tri, int ov, int longest_edge);
pybind11::tuple relabel_shim(const Tensor3& tri, int ov, int longest_edge);

double lawcos(double a, double b, double c);
double rad2deg(double radians);
int check_bad_tri(const Tensor3& tri, double angle_lim);

struct StandardizeResult {
    int bad_tri_code;
    Tensor3 tri;
    std::array<int,3> labels;
    Vec3 translation;
    Tensor3 R;
    double scale;
};
StandardizeResult standardize(const Tensor3& tri, double angle_lim, bool should_relabel);


struct KernelProps {
    int scale_power;
    int sm_power;
    bool flip_negate;
};
std::array<double,81> transform_from_standard(const std::array<double,81>& I,
    const KernelProps& k_props, double sm, const std::array<int,3>& labels,
    const Vec3& translation, const Tensor3& R, double scale);
KernelProps get_kernel_props(std::string K);

void expose_standardize(pybind11::module& m);
