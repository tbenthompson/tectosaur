#pragma once

#include <cmath>
#include <functional>
#include <memory>
#include "fmm_kernels.hpp"
#include "kdtree.hpp"

namespace tectosaur {

struct FMMConfig {
    // The MAC needs to < (1.0 / (check_r - 1)) so that farfield
    // approximations aren't used when the check surface intersects the
    // target box. How about we just use that as our MAC, since errors
    // further from the check surface should flatten out!
    double inner_r;
    double outer_r;
    size_t order;
    Kernel kernel;
    std::vector<double> params;

    int tensor_dim() const { return kernel.tensor_dim; }
};

struct Block {
    size_t row_start;
    size_t col_start;
    int n_rows;
    int n_cols;
    size_t data_start;
};

struct BlockSparseMat {
    std::vector<Block> blocks;
    std::vector<double> vals;

    std::vector<double> matvec(double* vec, size_t out_size);
    size_t get_nnz() { return vals.size(); }
};

struct MatrixFreeBlock {
    size_t obs_n_idx;
    size_t src_n_idx;
};

struct MatrixFreeOp {
    std::vector<MatrixFreeBlock> blocks;
};

struct NewMatrixFreeOp {
    std::vector<int> obs_n_start;
    std::vector<int> obs_n_end;
    std::vector<int> obs_n_idx;
    std::vector<int> src_n_start;
    std::vector<int> src_n_end;
    std::vector<int> src_n_idx;

    void insert(const KDNode& obs_n, const KDNode& src_n);
};

struct FMMMat {
    KDTree obs_tree;
    KDTree src_tree;
    FMMConfig cfg;
    std::vector<Vec3> surf;
    int translation_surface_order;

    FMMMat(KDTree obs_tree, KDTree src_tree, FMMConfig cfg,
        std::vector<Vec3> surf);

    std::vector<Vec3> get_surf(const KDNode& src_n, double r);
    
    int tensor_dim() const { return cfg.tensor_dim(); }

    void p2m_matvec(double* out, double* in);
    void m2m_matvec(double* out, double* in, int level);
    void p2l_matvec(double* out, double* in);
    void m2l_matvec(double* out, double* in);
    void l2l_matvec(double* out, double* in, int level);
    void p2p_matvec(double* out, double* in);
    void m2p_matvec(double* out, double* in);
    void l2p_matvec(double* out, double* in);

    std::vector<double> p2p_eval(double* in);
    std::vector<double> eval(double* in);

    NewMatrixFreeOp p2p_new;
    MatrixFreeOp p2p;
    MatrixFreeOp p2m;
    MatrixFreeOp p2l;
    MatrixFreeOp m2p;
    std::vector<MatrixFreeOp> m2m;
    MatrixFreeOp m2l;
    MatrixFreeOp l2p;
    std::vector<MatrixFreeOp> l2l;

    std::vector<BlockSparseMat> uc2e;
    std::vector<BlockSparseMat> dc2e;
};

FMMMat fmmmmmmm(const KDTree& obs_tree, const KDTree& src_tree, const FMMConfig& cfg);

}  // end namespace tectosaur
