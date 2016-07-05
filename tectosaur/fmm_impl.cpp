#include "fmm_impl.hpp"
#include "blas_wrapper.hpp"
#include "lib/doctest.h"
#include "test_helpers.hpp"
#include <cmath>
#include <limits>
#include <iostream>
#include <cassert>

namespace tectosaur {

std::vector<Vec3> inscribe_surf(const Sphere& b, double scaling, 
    const std::vector<Vec3>& fmm_surf) 
{
    std::vector<Vec3> out(fmm_surf.size());
    for (size_t i = 0; i < fmm_surf.size(); i++) {
        for (size_t d = 0; d < 3; d++) {
            out[i][d] = fmm_surf[i][d] * b.r * scaling + b.center[d];
        }
    }
    return out;
}

TEST_CASE("inscribe") {
    auto s = inscribe_surf({{1,1,1},2}, 0.5, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
    REQUIRE(s.size() == size_t(3));
    REQUIRE_ARRAY_EQUAL(s[0], Vec3{2,1,1}, 3);
    REQUIRE_ARRAY_EQUAL(s[1], Vec3{1,2,1}, 3);
    REQUIRE_ARRAY_EQUAL(s[2], Vec3{1,1,2}, 3);
}

struct Workspace {
    FMMMat& result;
    const KDTree& obs_tree;
    const KDTree& src_tree;
    const FMMConfig& cfg;

    Workspace(FMMMat& result, const KDTree& obs_tree,
        const KDTree& src_tree, const FMMConfig& cfg):
        result(result), obs_tree(obs_tree), src_tree(src_tree), cfg(cfg)
    {}

    std::vector<Vec3> get_surf(const KDNode& src_n, double r) const {
        return inscribe_surf(src_n.bounds, r, cfg.surf);
    }
};

double one(const Vec3& obs, const Vec3& src) { return 1.0; }

double inv_r(const Vec3& obs, const Vec3& src) {
    auto d = sub(obs, src);
    auto r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    return 1.0 / std::sqrt(r2);
}

std::vector<double> BlockSparseMat::matvec(double* vec, size_t out_size) {
    std::vector<double> out(out_size, 0.0);
    for (size_t i = 0; i < rows.size(); i++) {
        out[rows[i]] += vals[i] * vec[cols[i]]; 
    }
    return out;
}

TEST_CASE("matvec") {
    BlockSparseMat m{{0,1,1}, {1,0,1}, {2.0, 1.0, 3.0}};
    std::vector<double> in = {-1, 1};
    auto out = m.matvec(in.data(), 2);
    REQUIRE(out[0] == 2.0);
    REQUIRE(out[1] == 2.0);
}

void Kernel::direct_nbody(const Vec3* obs_pts, const Vec3* src_pts,
        size_t n_obs, size_t n_src, double* out) const
{
    for (size_t i = 0; i < n_obs; i++) {
        for (size_t j = 0; j < n_src; j++) {
            out[i * n_src + j] = f(obs_pts[i], src_pts[j]);
        }
    }
}

struct C2E {
    std::vector<Vec3> check_surf; 
    std::vector<double> op;
};

C2E check_to_equiv(const Workspace& ws, const KDNode& node,
    double equiv_r, double check_r) 
{
    // equiv surface to check surface
    auto equiv_surf = ws.get_surf(node, equiv_r);
    auto check_surf = ws.get_surf(node, check_r);
    auto n_surf = ws.cfg.surf.size();

    std::vector<double> equiv_to_check(n_surf * n_surf);
    ws.cfg.kernel.direct_nbody(
        check_surf.data(), equiv_surf.data(),
        n_surf, n_surf, equiv_to_check.data()
    );

    // invert equiv to check operator
    // In some cases, the equivalent surface to check surface operator
    // is poorly conditioned. In this case, truncate the singular values 
    // to solve a regularized least squares version of the problem.
    // TODO: There is quite a bit of numerical error incurred by storing this
    // fully inverted. Can I just store it in factored form? Increases complexity.
    auto svd = svd_decompose(equiv_to_check.data(), n_surf);
    const double truncation_threshold = 1e-15;
    set_threshold(svd, truncation_threshold);
    return {check_surf, svd_pseudoinverse(svd)};
}
        

void to_pts(const Workspace& ws, const KDNode& obs_n, BlockSparseMat& mat,
    const Vec3* src_pts, size_t n_src, size_t src_dof_start)
{
    size_t n_obs = obs_n.end - obs_n.start;
    for (size_t i = 0; i < n_obs; i++) {
        for (size_t j = 0; j < n_src; j++) {
            mat.rows.push_back(obs_n.start + i);
            mat.cols.push_back(src_dof_start + j);
        }
    }
    auto old_n_vals = mat.vals.size(); 
    mat.vals.resize(old_n_vals + n_obs * n_src);
    ws.cfg.kernel.direct_nbody(
        &ws.obs_tree.pts[obs_n.start], src_pts,
        n_obs, n_src, &mat.vals[old_n_vals]
    );
}

void p2p(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    to_pts(
        ws, obs_n, ws.result.p2p,
        &ws.src_tree.pts[src_n.start], src_n.end - src_n.start, src_n.start
    );
}

void m2p(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto equiv = ws.get_surf(src_n, ws.cfg.inner_r);
    to_pts(
        ws, obs_n, ws.result.m2p,
        equiv.data(), ws.cfg.surf.size(), src_n.idx * ws.cfg.surf.size()
    );
}

void l2p(const Workspace& ws, const KDNode& obs_n) {
    auto equiv = ws.get_surf(obs_n, ws.cfg.outer_r);
    to_pts(
        ws, obs_n, ws.result.l2p,
        equiv.data(), ws.cfg.surf.size(), obs_n.idx * ws.cfg.surf.size()
    );
}

void to_appx(const Workspace& ws, const KDNode& obs_n,
    const Vec3* src_pts, size_t n_src, size_t src_start_dof,
    BlockSparseMat& mat, double check_r, double equiv_r) 
{
    auto c2e = check_to_equiv(ws, obs_n, equiv_r, check_r);

    auto n_surf = ws.cfg.surf.size();

    // points to check surface
    std::vector<double> pts_to_check(n_src * n_surf);
    ws.cfg.kernel.direct_nbody(
         c2e.check_surf.data(), src_pts, n_surf, n_src, pts_to_check.data()
    );

    // multiply with points to check operator
    auto op = mat_mult(n_surf, n_src, false, c2e.op, false, pts_to_check);
    
    for (size_t i = 0; i < n_surf; i++) {
        for (size_t j = 0; j < n_src; j++) {
            mat.rows.push_back(obs_n.idx * n_surf + i);
            mat.cols.push_back(src_start_dof + j);
            mat.vals.push_back(op[i * n_src + j]); 
        }
    }

}

void p2m(const Workspace& ws, const KDNode& src_n) {
    to_appx(ws, src_n,
        &ws.src_tree.pts[src_n.start], src_n.end - src_n.start, src_n.start,
        ws.result.p2m, ws.cfg.outer_r, ws.cfg.inner_r
    );
}

void m2m(const Workspace& ws, const KDNode& parent_n, const KDNode& child_n) {
    auto equiv = ws.get_surf(child_n, ws.cfg.inner_r);
    to_appx(
        ws, parent_n,
        equiv.data(), ws.cfg.surf.size(), child_n.idx * ws.cfg.surf.size(),
        ws.result.m2m[parent_n.depth], ws.cfg.outer_r, ws.cfg.inner_r
    );
}

void l2l(const Workspace& ws, const KDNode& parent_n, const KDNode& child_n) {
    auto equiv = ws.get_surf(parent_n, ws.cfg.outer_r);
    to_appx(
        ws, child_n,
        equiv.data(), ws.cfg.surf.size(), parent_n.idx * ws.cfg.surf.size(),
        ws.result.l2l[child_n.depth], ws.cfg.inner_r, ws.cfg.outer_r
    );
}

void p2l(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    to_appx(ws, obs_n,
        &ws.src_tree.pts[src_n.start], src_n.end - src_n.start, src_n.start,
        ws.result.p2l, ws.cfg.inner_r, ws.cfg.outer_r
    );
}

void m2l(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto equiv = ws.get_surf(src_n, ws.cfg.inner_r);
    to_appx(
        ws, obs_n,
        equiv.data(), ws.cfg.surf.size(), src_n.idx * ws.cfg.surf.size(),
        ws.result.m2l, ws.cfg.inner_r, ws.cfg.outer_r
    );
}

void traverse(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto r_src = src_n.bounds.r;
    auto r_obs = obs_n.bounds.r;
    auto sep = hypot(sub(obs_n.bounds.center, src_n.bounds.center));

    // If outer_r * r_src + inner_r * r_obs is less than the separation, then
    // the relevant check surfaces for the two interacting cells don't intersect.
    // That means it should be safe to perform approximate interactions. I add
    // a small safety factor just in case!
    double safety_factor = 0.98;
    if (ws.cfg.outer_r * r_src + ws.cfg.inner_r * r_obs < safety_factor * sep) {
        bool small_src = src_n.end - src_n.start < ws.cfg.surf.size();
        bool small_obs = obs_n.end - obs_n.start < ws.cfg.surf.size();
        if (small_src && small_obs) {
            p2p(ws, obs_n, src_n);
        } else if (small_obs) {
            m2p(ws, obs_n, src_n);
        } else if (small_src) {
            // p2p(ws, obs_n, src_n);
            p2l(ws, obs_n, src_n);
        } else {
            // m2p(ws, obs_n, src_n);
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

void up_collect(const Workspace& ws, const KDNode& src_n) {
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

FMMMat fmmmmmmm(const KDTree& obs_tree, const KDTree& src_tree, const FMMConfig& cfg) {
    FMMMat result;
    result.m2m.resize(src_tree.max_depth + 1);
    result.l2l.resize(obs_tree.max_depth + 1);
    Workspace ws(result, obs_tree, src_tree, cfg);
    up_collect(ws, src_tree.root());
    down_collect(ws, obs_tree.root());
    traverse(ws, obs_tree.root(), src_tree.root());
    std::cout << result.m2m.size() << std::endl;
    return result;
}

}
