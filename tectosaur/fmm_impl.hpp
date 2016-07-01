#pragma once

#include "octree.hpp"
#include <cmath>
#include <functional>
#include <memory>

namespace tectosaur {

struct Kernel {
    using FType = std::function<double(const Vec3&,const Vec3&)>;
    FType f;

    void direct_nbody(const Vec3* obs_pts, const Vec3* src_pts,
        size_t n_obs, size_t n_src, double* out) const;
};

double one(const Vec3&,const Vec3&);
double inv_r(const Vec3&,const Vec3&);

struct FMMConfig {
    double mac;
    std::vector<Vec3> surf;
    Kernel kernel;
    double equiv_r = 0.3 * std::sqrt(3);
    double check_r = 2.9 * std::sqrt(3);
};

struct SparseMat {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
};

struct FMMMat {
    SparseMat p2p;
    SparseMat p2m;
    SparseMat m2m;
    SparseMat m2p;
};

FMMMat fmmmmmmm(const KDTree& obs_tree, const KDTree& src_tree, const FMMConfig& cfg);

} //end namespace tectosaur
