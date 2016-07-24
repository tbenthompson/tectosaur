#include "fmm_impl.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include "lib/doctest.h"
#include "blas_wrapper.hpp"
#include "test_helpers.hpp"

namespace tectosaur {

std::vector<Vec3> inscribe_surf(const Sphere& b, double scaling,
                                const std::vector<Vec3>& fmm_surf) {
    std::vector<Vec3> out(fmm_surf.size());
    for (size_t i = 0; i < fmm_surf.size(); i++) {
        for (size_t d = 0; d < 3; d++) {
            out[i][d] = fmm_surf[i][d] * b.r * scaling + b.center[d];
        }
    }
    return out;
}

std::vector<Vec3> surrounding_surface_sphere(size_t order)
{
    std::vector<Vec3> pts;
    double a = 4 * M_PI / order;
    double d = std::sqrt(a);
    auto M_theta = static_cast<size_t>(std::round(M_PI / d));
    double d_theta = M_PI / M_theta;
    double d_phi = a / d_theta;
    for (size_t m = 0; m < M_theta; m++) {
        double theta = M_PI * (m + 0.5) / M_theta;
        auto M_phi = static_cast<size_t>(
            std::round(2 * M_PI * std::sin(theta) / d_phi)
        );
        for (size_t n = 0; n < M_phi; n++) {
            double phi = 2 * M_PI * n / M_phi;
            double x = std::sin(theta) * std::cos(phi);
            double y = std::sin(theta) * std::sin(phi);
            double z = std::cos(theta);
            pts.push_back({x, y, z});
        }
    }

    return pts;
}

TEST_CASE("inscribe") {
    auto s =
        inscribe_surf({{1, 1, 1}, 2}, 0.5, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    REQUIRE(s.size() == size_t(3));
    REQUIRE_ARRAY_EQUAL(s[0], Vec3{2, 1, 1}, 3);
    REQUIRE_ARRAY_EQUAL(s[1], Vec3{1, 2, 1}, 3);
    REQUIRE_ARRAY_EQUAL(s[2], Vec3{1, 1, 2}, 3);
}

struct Workspace {
    FMMMat& result;
    const KDTree& obs_tree;
    const KDTree& src_tree;
    std::vector<Vec3> surf;
    const FMMConfig& cfg;

    std::vector<Vec3> get_surf(const KDNode& src_n, double r) const {
        return inscribe_surf(src_n.bounds, r, surf);
    }
};

extern "C" void dgemv_(char* TRANS, int* M, int* N, double* ALPHA, double* A,
                       int* LDA, double* X, int* INCX, double* BETA, double* Y,
                       int* INCY);
std::vector<double> BlockSparseMat::matvec(double* vec, size_t out_size) {
    char transpose = 'T';
    double alpha = 1;
    double beta = 1;
    int inc = 1;
    std::vector<double> out(out_size, 0.0);
    for (size_t b_idx = 0; b_idx < blocks.size(); b_idx++) {
        auto& b = blocks[b_idx];
        dgemv_(
            &transpose, &b.n_cols, &b.n_rows, &alpha, &vals[b.data_start],
            &b.n_cols, &vec[b.col_start], &inc, &beta, &out[b.row_start], &inc
        );
    }
    return out;
}

TEST_CASE("matvec") {
    BlockSparseMat m{{{1, 1, 2, 2, 0}}, {0, 2, 1, 3}};
    std::vector<double> in = {0, -1, 1};
    auto out = m.matvec(in.data(), 3);
    REQUIRE(out[0] == 0.0);
    REQUIRE(out[1] == 2.0);
    REQUIRE(out[2] == 2.0);
}

void to_pts(const Workspace& ws, BlockSparseMat& mat,
            const Vec3* obs_pts, const Vec3* obs_ns,
            size_t n_obs, size_t obs_pt_start,
            const Vec3* src_pts, const Vec3* src_ns,
            size_t n_src, size_t src_pt_start) {
    auto tdim = ws.cfg.kernel.tensor_dim;
    auto n_rows = n_obs * tdim;
    auto n_cols = n_src * tdim;
    mat.blocks.push_back({obs_pt_start * tdim, src_pt_start * tdim, int(n_rows),
                          int(n_cols), mat.vals.size()});
    auto old_n_vals = mat.vals.size();
    mat.vals.resize(old_n_vals + n_rows * n_cols);
    ws.cfg.kernel.f(
        NBodyProblem{obs_pts, obs_ns, src_pts, src_ns, n_obs, n_src, ws.cfg.params.data()},
        &mat.vals[old_n_vals]);
}

void p2p(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    to_pts(ws, ws.result.p2p, 
        &ws.obs_tree.pts[obs_n.start], &ws.obs_tree.normals[obs_n.start],
        obs_n.end - obs_n.start, obs_n.start,
        &ws.src_tree.pts[src_n.start], &ws.src_tree.normals[src_n.start],
        src_n.end - src_n.start, src_n.start);
}

void m2p(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto equiv = ws.get_surf(src_n, ws.cfg.inner_r);
    to_pts(ws, ws.result.m2p,
        &ws.obs_tree.pts[obs_n.start], &ws.obs_tree.normals[obs_n.start],
        obs_n.end - obs_n.start, obs_n.start,
        equiv.data(), ws.surf.data(),
        ws.surf.size(), src_n.idx * ws.surf.size());
}

void l2p(const Workspace& ws, const KDNode& obs_n) {
    auto equiv = ws.get_surf(obs_n, ws.cfg.outer_r);
    to_pts(ws, ws.result.l2p, 
        &ws.obs_tree.pts[obs_n.start], &ws.obs_tree.normals[obs_n.start],
        obs_n.end - obs_n.start, obs_n.start,
        equiv.data(), ws.surf.data(),
        ws.surf.size(), obs_n.idx * ws.surf.size());
}

void p2m(const Workspace& ws, const KDNode& src_n) {
    auto check = ws.get_surf(src_n, ws.cfg.outer_r);
    to_pts(ws, ws.result.p2m, 
        check.data(), ws.surf.data(), 
        ws.surf.size(), src_n.idx * ws.surf.size(),
        &ws.src_tree.pts[src_n.start], &ws.src_tree.normals[src_n.start],
        src_n.end - src_n.start, src_n.start);
}

void m2m(const Workspace& ws, const KDNode& parent_n, const KDNode& child_n) {
    auto check = ws.get_surf(parent_n, ws.cfg.outer_r);
    auto equiv = ws.get_surf(child_n, ws.cfg.inner_r);
    to_pts(ws, ws.result.m2m[parent_n.height], 
        check.data(), ws.surf.data(), 
        ws.surf.size(), parent_n.idx * ws.surf.size(),
        equiv.data(), ws.surf.data(), 
        ws.surf.size(), child_n.idx * ws.surf.size());
}

void l2l(const Workspace& ws, const KDNode& parent_n, const KDNode& child_n) {
    auto check = ws.get_surf(child_n, ws.cfg.inner_r);
    auto equiv = ws.get_surf(parent_n, ws.cfg.outer_r);
    to_pts(ws, ws.result.l2l[child_n.depth],
        check.data(), ws.surf.data(), 
        ws.surf.size(), child_n.idx * ws.surf.size(),
        equiv.data(), ws.surf.data(), 
        ws.surf.size(), parent_n.idx * ws.surf.size());
}

void p2l(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto check = ws.get_surf(obs_n, ws.cfg.inner_r);
    to_pts(ws, ws.result.p2l, 
        check.data(), ws.surf.data(), 
        ws.surf.size(), obs_n.idx * ws.surf.size(),
        &ws.src_tree.pts[src_n.start], &ws.src_tree.normals[src_n.start],
        src_n.end - src_n.start, src_n.start);
}

void m2l(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto check = ws.get_surf(obs_n, ws.cfg.inner_r);
    auto equiv = ws.get_surf(src_n, ws.cfg.inner_r);
    to_pts(ws, ws.result.m2l,
        check.data(), ws.surf.data(), 
        ws.surf.size(), obs_n.idx * ws.surf.size(),
        equiv.data(), ws.surf.data(), 
        ws.surf.size(), src_n.idx * ws.surf.size());
}

void traverse(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto r_src = src_n.bounds.r;
    auto r_obs = obs_n.bounds.r;
    auto sep = hypot(sub(obs_n.bounds.center, src_n.bounds.center));

    // If outer_r * r_src + inner_r * r_obs is less than the separation, then
    // the relevant check surfaces for the two interacting cells don't
    // intersect.
    // That means it should be safe to perform approximate interactions. I add
    // a small safety factor just in case!
    double safety_factor = 0.98;
    if (ws.cfg.outer_r * r_src + ws.cfg.inner_r * r_obs < safety_factor * sep) {
        // If there aren't enough src or obs to justify using the approximation,
        // then just do a p2p direct calculation between the nodes.
        bool small_src = src_n.end - src_n.start < ws.surf.size();
        bool small_obs = obs_n.end - obs_n.start < ws.surf.size();

        if (small_src && small_obs) {
            p2p(ws, obs_n, src_n);
        } else if (small_obs) {
            m2p(ws, obs_n, src_n);
        } else if (small_src) {
            p2l(ws, obs_n, src_n);
        } else {
            m2l(ws, obs_n, src_n);
        }
        return;
    }

    if (src_n.is_leaf && obs_n.is_leaf) {
        p2p(ws, obs_n, src_n);
        return;
    }

    bool split_src = ((r_obs < r_src) && !src_n.is_leaf) || obs_n.is_leaf;
    if (split_src) {
        for (int i = 0; i < 2; i++) {
            traverse(ws, obs_n, ws.src_tree.nodes[src_n.children[i]]);
        }
    } else {
        for (int i = 0; i < 2; i++) {
            traverse(ws, ws.obs_tree.nodes[obs_n.children[i]], src_n);
        }
    }
}

extern "C" void dgelsy_(int* M, int* N, int* NRHS, double* A, int* LDA,
                        double* B, int* LDB, int* JPVT, double* RCOND,
                        int* RANK, double* WORK, int* LWORK, int* INFO);

std::vector<double> qr_pseudoinverse(double* matrix, int n) {
    std::vector<double> rhs(n * n, 0.0);
    for (int i = 0; i < n; i++) {
        rhs[i * n + i] = 1.0;
    }

    std::vector<int> jpvt(n, 0);
    int lwork = 4 * n + 1;
    std::vector<double> work(lwork);
    int rank;
    int info;
    double rcond = 1e-15;
    dgelsy_(&n, &n, &n, matrix, &n, rhs.data(), &n, jpvt.data(), &rcond, &rank,
            work.data(), &lwork, &info);
    return rhs;
}

// invert equiv to check operator
// In some cases, the equivalent surface to check surface operator
// is poorly conditioned. In this case, truncate the singular values
// to solve a regularized least squares version of the problem.
//
// TODO: There is quite a bit of numerical error incurred by storing this
// fully inverted and truncated.
//
// Can I just store it in factored form? Increases complexity.
// Without doing this, the error is can't be any lower than ~10^-10. Including
// this, the error can get down to 10^-15.
// I don't expect to need anything better than 10^-10. But in single precision,
// the number may be much lower. In which case, I may need to go to double
// precision sooner than I'd prefer.
// So, I see two ways to design this. I can store the check to equiv matrix
// along with each block that needs it. Or, I can separate the P2M, M2M, M2L,
// P2L, L2L into two steps each: P2M, M2M, M2L, P2L, L2L and UC2E and DC2E
// (Up check to equiv and down check to equiv)
// The latter approach seems better, since less needs to be stored. The
// U2M and D2L matrices should be separated by level like M2M and L2L.
// <-- (later note) I did this.
void c2e(const Workspace& ws, BlockSparseMat& mat, const KDNode& node,
         double check_r, double equiv_r) {
    auto equiv_surf = ws.get_surf(node, equiv_r);
    auto check_surf = ws.get_surf(node, check_r);
    auto n_surf = ws.surf.size();

    auto n_rows = n_surf * ws.cfg.kernel.tensor_dim;

    std::vector<double> equiv_to_check(n_rows * n_rows);
    ws.cfg.kernel.f(
        {
            check_surf.data(), ws.surf.data(), 
            equiv_surf.data(), ws.surf.data(),
            n_surf, n_surf,
            ws.cfg.params.data()
        },
        equiv_to_check.data());

    // TODO: Currently, svd decomposition is the most time consuming part of
    // assembly. How to optimize this?
    // 1) Batch a bunch of SVDs to the gpu.
    // 2) Figure out a way to do less of them. Prune tree nodes?
    //   May get about 25-50% faster.
    // 3) A faster alternative? QR? <--- This seems like the first step.
    // auto svd = svd_decompose(equiv_to_check.data(), n_rows);
    // const double truncation_threshold = 1e-15;
    // set_threshold(svd, truncation_threshold);
    // auto pinv = svd_pseudoinverse(svd);
    auto pinv = qr_pseudoinverse(equiv_to_check.data(), n_rows);

    mat.blocks.push_back({node.idx * n_rows, node.idx * n_rows, int(n_rows),
                          int(n_rows), mat.vals.size()});
    mat.vals.insert(mat.vals.end(), pinv.begin(), pinv.end());
}

void up_collect(const Workspace& ws, const KDNode& src_n) {
    c2e(ws, ws.result.uc2e[src_n.height], src_n, ws.cfg.outer_r, ws.cfg.inner_r);
    if (src_n.is_leaf) {
        p2m(ws, src_n);
    } else {
        for (int i = 0; i < 2; i++) {
            auto child = ws.src_tree.nodes[src_n.children[i]];
            up_collect(ws, child);
            m2m(ws, src_n, child);
        }
    }
}

void down_collect(const Workspace& ws, const KDNode& obs_n) {
    c2e(ws, ws.result.dc2e[obs_n.depth], obs_n, ws.cfg.inner_r, ws.cfg.outer_r);
    if (obs_n.is_leaf) {
        l2p(ws, obs_n);
    } else {
        for (int i = 0; i < 2; i++) {
            auto child = ws.obs_tree.nodes[obs_n.children[i]];
            down_collect(ws, child);
            l2l(ws, obs_n, child);
        }
    }
}

FMMMat fmmmmmmm(const KDTree& obs_tree, const KDTree& src_tree,
                const FMMConfig& cfg) {

    auto translation_surf = surrounding_surface_sphere(cfg.order);

    FMMMat result;
    result.tensor_dim = cfg.kernel.tensor_dim;
    result.translation_surface_order = translation_surf.size();

    result.m2m.resize(src_tree.max_height + 1);
    result.l2l.resize(obs_tree.max_height + 1);
    result.uc2e.resize(src_tree.max_height + 1);
    result.dc2e.resize(obs_tree.max_height + 1);

    Workspace ws{result, obs_tree, src_tree, translation_surf, cfg};
#pragma omp parallel
#pragma omp single nowait
    {
#pragma omp task
        up_collect(ws, src_tree.root());
#pragma omp task
        down_collect(ws, obs_tree.root());
#pragma omp task
        traverse(ws, obs_tree.root(), src_tree.root());
    }

    return result;
}
}
