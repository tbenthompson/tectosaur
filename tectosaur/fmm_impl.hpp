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
    // I think the MAC needs to < (1.0 / (check_r - 1)) so that farfield
    // approximations aren't used when the check surface intersects the 
    // target box.
    double mac;
    double equiv_r;
    double check_r;
    std::vector<Vec3> surf;
    Kernel kernel;
};

struct SparseMat {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;

    std::vector<double> matvec(double* vec, size_t out_size);
    size_t get_nnz() { return rows.size(); }
};

struct FMMMat {
    SparseMat p2p;
    SparseMat p2m;
    // SparseMat p2l;
    SparseMat m2m;
    SparseMat m2p;
    // SparseMat m2l;
    // SparseMat l2l;
    // SparseMat l2p;
};

FMMMat fmmmmmmm(const KDTree& obs_tree, const KDTree& src_tree, const FMMConfig& cfg);

} //end namespace tectosaur
