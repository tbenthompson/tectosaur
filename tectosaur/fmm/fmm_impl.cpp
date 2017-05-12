#include "fmm_impl.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include "lib/doctest.h"
#include "blas_wrapper.hpp"
#include "lib/test_helpers.hpp"

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

void NewMatrixFreeOp::insert(const KDNode& obs_n, const KDNode& src_n) {
    obs_n_start.push_back(obs_n.start);
    obs_n_end.push_back(obs_n.end);
    obs_n_idx.push_back(obs_n.idx);
    src_n_start.push_back(src_n.start);
    src_n_end.push_back(src_n.end);
    src_n_idx.push_back(src_n.idx);
}

void p2p(FMMMat& mat, const KDNode& obs_n, const KDNode& src_n) {
    mat.p2p_new.insert(obs_n, src_n);
    mat.p2p.blocks.push_back(MatrixFreeBlock{obs_n.idx, src_n.idx});
}

void m2p(FMMMat& mat, const KDNode& obs_n, const KDNode& src_n) {
    mat.m2p.blocks.push_back(MatrixFreeBlock{obs_n.idx, src_n.idx});
}

void l2p(FMMMat& mat, const KDNode& obs_n) {
    mat.l2p.blocks.push_back(MatrixFreeBlock{obs_n.idx, obs_n.idx});
}

void p2m(FMMMat& mat, const KDNode& src_n) {
    mat.p2m.blocks.push_back(MatrixFreeBlock{src_n.idx, src_n.idx});
}

void m2m(FMMMat& mat, const KDNode& parent_n, const KDNode& child_n) {
    mat.m2m[parent_n.height].blocks.push_back(
        MatrixFreeBlock{parent_n.idx, child_n.idx}
    );
}

void l2l(FMMMat& mat, const KDNode& parent_n, const KDNode& child_n) {
    mat.l2l[child_n.depth].blocks.push_back(
        MatrixFreeBlock{child_n.idx, parent_n.idx}
    );
}

void p2l(FMMMat& mat, const KDNode& obs_n, const KDNode& src_n) {
    mat.p2l.blocks.push_back(MatrixFreeBlock{obs_n.idx, src_n.idx});
}

void m2l(FMMMat& mat, const KDNode& obs_n, const KDNode& src_n) {
    mat.m2l.blocks.push_back(MatrixFreeBlock{obs_n.idx, src_n.idx});
}

void traverse(FMMMat& mat, const KDNode& obs_n, const KDNode& src_n) {
    auto r_src = src_n.bounds.r;
    auto r_obs = obs_n.bounds.r;
    auto sep = hypot(sub(obs_n.bounds.center, src_n.bounds.center));

    // If outer_r * r_src + inner_r * r_obs is less than the separation, then
    // the relevant check surfaces for the two interacting cells don't
    // intersect.
    // That means it should be safe to perform approximate interactions. I add
    // a small safety factor just in case!
    double safety_factor = 0.98;
    if (mat.cfg.outer_r * r_src + mat.cfg.inner_r * r_obs < safety_factor * sep) {
        // If there aren't enough src or obs to justify using the approximation,
        // then just do a p2p direct calculation between the nodes.
        bool small_src = src_n.end - src_n.start < mat.surf.size();
        bool small_obs = obs_n.end - obs_n.start < mat.surf.size();

        if (small_src && small_obs) {
            p2p(mat, obs_n, src_n);
        } else if (small_obs) {
            m2p(mat, obs_n, src_n);
        } else if (small_src) {
            p2l(mat, obs_n, src_n);
        } else {
            m2l(mat, obs_n, src_n);
        }
        return;
    }

    if (src_n.is_leaf && obs_n.is_leaf) {
        p2p(mat, obs_n, src_n);
        return;
    }

    bool split_src = ((r_obs < r_src) && !src_n.is_leaf) || obs_n.is_leaf;
    if (split_src) {
        for (int i = 0; i < 2; i++) {
            traverse(mat, obs_n, mat.src_tree.nodes[src_n.children[i]]);
        }
    } else {
        for (int i = 0; i < 2; i++) {
            traverse(mat, mat.obs_tree.nodes[obs_n.children[i]], src_n);
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
void c2e(FMMMat& mat, BlockSparseMat& sub_mat, const KDNode& node,
         double check_r, double equiv_r) {
    auto equiv_surf = mat.get_surf(node, equiv_r);
    auto check_surf = mat.get_surf(node, check_r);
    auto n_surf = mat.surf.size();

    auto n_rows = n_surf * mat.tensor_dim();

    std::vector<double> equiv_to_check(n_rows * n_rows);
    mat.cfg.kernel.f(
        {
            check_surf.data(), mat.surf.data(), 
            equiv_surf.data(), mat.surf.data(),
            n_surf, n_surf,
            mat.cfg.params.data()
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

    sub_mat.blocks.push_back({node.idx * n_rows, node.idx * n_rows, int(n_rows),
                          int(n_rows), sub_mat.vals.size()});
    sub_mat.vals.insert(sub_mat.vals.end(), pinv.begin(), pinv.end());
}

void up_collect(FMMMat& mat, const KDNode& src_n) {
    c2e(mat, mat.uc2e[src_n.height], src_n, mat.cfg.outer_r, mat.cfg.inner_r);
    if (src_n.is_leaf) {
        p2m(mat, src_n);
    } else {
        for (int i = 0; i < 2; i++) {
            auto child = mat.src_tree.nodes[src_n.children[i]];
            up_collect(mat, child);
            m2m(mat, src_n, child);
        }
    }
}

void down_collect(FMMMat& mat, const KDNode& obs_n) {
    c2e(mat, mat.dc2e[obs_n.depth], obs_n, mat.cfg.inner_r, mat.cfg.outer_r);
    if (obs_n.is_leaf) {
        l2p(mat, obs_n);
    } else {
        for (int i = 0; i < 2; i++) {
            auto child = mat.obs_tree.nodes[obs_n.children[i]];
            down_collect(mat, child);
            l2l(mat, obs_n, child);
        }
    }
}

FMMMat::FMMMat(KDTree obs_tree, KDTree src_tree, FMMConfig cfg,
        std::vector<Vec3> surf):
    obs_tree(obs_tree),
    src_tree(src_tree),
    cfg(cfg),
    surf(surf),
    translation_surface_order(surf.size())
{}

void interact_pts(const FMMConfig& cfg, double* out, double* in,
    const Vec3* obs_pts, const Vec3* obs_ns, size_t n_obs, size_t obs_pt_start,
    const Vec3* src_pts, const Vec3* src_ns, size_t n_src, size_t src_pt_start) 
{
    if (n_obs == 0 || n_src == 0) {
        return;
    }

    double* out_val_start = &out[cfg.tensor_dim() * obs_pt_start];
    double* in_val_start = &in[cfg.tensor_dim() * src_pt_start];
    cfg.kernel.f_mf(
        NBodyProblem{obs_pts, obs_ns, src_pts, src_ns, n_obs, n_src, cfg.params.data()},
        out_val_start, in_val_start
    );
}

std::vector<Vec3> FMMMat::get_surf(const KDNode& src_n, double r) {
    return inscribe_surf(src_n.bounds, r, surf);
}

void FMMMat::p2p_matvec(double* out, double* in) {
    for (auto& b: p2p.blocks) {
        auto obs_n = obs_tree.nodes[b.obs_n_idx];
        auto src_n = src_tree.nodes[b.src_n_idx];
        interact_pts(
            cfg, out, in,
            &obs_tree.pts[obs_n.start], &obs_tree.normals[obs_n.start],
            obs_n.end - obs_n.start, obs_n.start,
            &src_tree.pts[src_n.start], &src_tree.normals[src_n.start],
            src_n.end - src_n.start, src_n.start
        );
    }
}

void FMMMat::p2m_matvec(double* out, double *in) {
    for (auto& b: p2m.blocks) {
        auto src_n = src_tree.nodes[b.src_n_idx];
        auto check = get_surf(src_n, cfg.outer_r);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), src_n.idx * surf.size(),
            &src_tree.pts[src_n.start], &src_tree.normals[src_n.start],
            src_n.end - src_n.start, src_n.start
        );
    }
}

void FMMMat::m2m_matvec(double* out, double *in, int level) {
    for (auto& b: m2m[level].blocks) {
        auto parent_n = src_tree.nodes[b.obs_n_idx];
        auto child_n = src_tree.nodes[b.src_n_idx];
        auto check = get_surf(parent_n, cfg.outer_r);
        auto equiv = get_surf(child_n, cfg.inner_r);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), parent_n.idx * surf.size(),
            equiv.data(), surf.data(), 
            surf.size(), child_n.idx * surf.size()
        );
    }
}

void FMMMat::p2l_matvec(double* out, double* in) {
    for (auto& b: p2l.blocks) {
        auto obs_n = obs_tree.nodes[b.obs_n_idx];
        auto src_n = src_tree.nodes[b.src_n_idx];

        auto check = get_surf(obs_n, cfg.inner_r);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), obs_n.idx * surf.size(),
            &src_tree.pts[src_n.start], &src_tree.normals[src_n.start],
            src_n.end - src_n.start, src_n.start
        );
    }
}

void FMMMat::m2l_matvec(double* out, double* in) {
    for (auto& b: m2l.blocks) {
        auto obs_n = obs_tree.nodes[b.obs_n_idx];
        auto src_n = src_tree.nodes[b.src_n_idx];

        auto check = get_surf(obs_n, cfg.inner_r);
        auto equiv = get_surf(src_n, cfg.inner_r);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), obs_n.idx * surf.size(),
            equiv.data(), surf.data(), 
            surf.size(), src_n.idx * surf.size()
        );
    }
}

void FMMMat::l2l_matvec(double* out, double* in, int level) {
    for (auto& b: l2l[level].blocks) {
        auto child_n = obs_tree.nodes[b.obs_n_idx];
        auto parent_n = obs_tree.nodes[b.src_n_idx];

        auto check = get_surf(child_n, cfg.inner_r);
        auto equiv = get_surf(parent_n, cfg.outer_r);
        interact_pts(
            cfg, out, in,
            check.data(), surf.data(), 
            surf.size(), child_n.idx * surf.size(),
            equiv.data(), surf.data(), 
            surf.size(), parent_n.idx * surf.size()
        );
    }
}

void FMMMat::m2p_matvec(double* out, double* in) {
    for (auto& b: m2p.blocks) {
        auto obs_n = obs_tree.nodes[b.obs_n_idx];
        auto src_n = src_tree.nodes[b.src_n_idx];

        auto equiv = get_surf(src_n, cfg.inner_r);
        interact_pts(
            cfg, out, in,
            &obs_tree.pts[obs_n.start], &obs_tree.normals[obs_n.start],
            obs_n.end - obs_n.start, obs_n.start,
            equiv.data(), surf.data(),
            surf.size(), src_n.idx * surf.size()
        );
    }
}

void FMMMat::l2p_matvec(double* out, double* in) {
    for (auto& b: l2p.blocks) {
        auto obs_n = obs_tree.nodes[b.obs_n_idx];

        auto equiv = get_surf(obs_n, cfg.outer_r);
        interact_pts(
            cfg, out, in,
            &obs_tree.pts[obs_n.start], &obs_tree.normals[obs_n.start],
            obs_n.end - obs_n.start, obs_n.start,
            equiv.data(), surf.data(),
            surf.size(), obs_n.idx * surf.size()
        );
    }

}

template <typename T>
void inplace_add_vecs(std::vector<T>& a, const std::vector<T>& b) {
    for (size_t j = 0; j < a.size(); j++) {
        a[j] += b[j];
    }
}

template <typename T>
void zero_vec(std::vector<T>& v) {
    std::fill(v.begin(), v.end(), 0.0);
}

std::vector<double> FMMMat::p2p_eval(double* in) {
    auto n_outputs = obs_tree.pts.size() * tensor_dim();
    std::vector<double> out(n_outputs, 0.0);
    p2p_matvec(out.data(), in);
    return out;
}

std::vector<double> FMMMat::eval(double* in) {
    auto n_outputs = obs_tree.pts.size() * tensor_dim();
    auto n_multipoles = surf.size() * src_tree.nodes.size() * tensor_dim();
    auto n_locals = surf.size() * obs_tree.nodes.size() * tensor_dim();

    std::vector<double> out(n_outputs, 0.0);
    // p2p_matvec(out.data(), in);

    std::vector<double> m_check(n_multipoles, 0.0);
    p2m_matvec(m_check.data(), in);

    auto multipoles = uc2e[0].matvec(m_check.data(), n_multipoles);

    for (size_t i = 1; i < m2m.size(); i++) {
        zero_vec(m_check);
        m2m_matvec(m_check.data(), multipoles.data(), i);
        auto add_to_multipoles = uc2e[i].matvec(m_check.data(), n_multipoles);
        inplace_add_vecs(multipoles, add_to_multipoles);
    }

    std::vector<double> l_check(n_locals, 0.0);
    p2l_matvec(l_check.data(), in);
    m2l_matvec(l_check.data(), multipoles.data());

    std::vector<double> locals(n_locals, 0.0);
    for (size_t i = 0; i < l2l.size(); i++) {
        l2l_matvec(l_check.data(), locals.data(), i);
        auto add_to_locals = dc2e[i].matvec(l_check.data(), n_locals);
        inplace_add_vecs(locals, add_to_locals);
    }

    m2p_matvec(out.data(), multipoles.data());
    l2p_matvec(out.data(), locals.data());

    return out;
}

FMMMat fmmmmmmm(const KDTree& obs_tree, const KDTree& src_tree,
                const FMMConfig& cfg) {

    auto translation_surf = surrounding_surface_sphere(cfg.order);

    FMMMat mat(obs_tree, src_tree, cfg, translation_surf);

    mat.m2m.resize(mat.src_tree.max_height + 1);
    mat.l2l.resize(mat.obs_tree.max_height + 1);
    mat.m2m.resize(mat.src_tree.max_height + 1);
    mat.l2l.resize(mat.obs_tree.max_height + 1);
    mat.uc2e.resize(mat.src_tree.max_height + 1);
    mat.dc2e.resize(mat.obs_tree.max_height + 1);

#pragma omp parallel
#pragma omp single nowait
    {
#pragma omp task
        up_collect(mat, mat.src_tree.root());
#pragma omp task
        down_collect(mat, mat.obs_tree.root());
#pragma omp task
        traverse(mat, mat.obs_tree.root(), mat.src_tree.root());
    }

    return mat;
}

}
