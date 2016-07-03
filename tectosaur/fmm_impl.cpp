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

    std::vector<std::vector<Vec3>> equiv_surfs;

    Workspace(FMMMat& result, const KDTree& obs_tree,
        const KDTree& src_tree, const FMMConfig& cfg):
        result(result), obs_tree(obs_tree), src_tree(src_tree), cfg(cfg),
        equiv_surfs(src_tree.nodes.size())
    {}

    const std::vector<Vec3>& get_equiv_surf(const KDNode& src_n) {
        if (equiv_surfs[src_n.idx].size() == 0) {
            equiv_surfs[src_n.idx] = inscribe_surf(src_n.bounds, cfg.equiv_r, cfg.surf);
        }
        return equiv_surfs[src_n.idx];
    }
};

double one(const Vec3& obs, const Vec3& src) { return 1.0; }

double inv_r(const Vec3& obs, const Vec3& src) {
    auto d = sub(obs, src);
    auto r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    return 1.0 / std::sqrt(r2);
}

std::vector<double> SparseMat::matvec(double* vec, size_t out_size) {
    std::vector<double> out(out_size, 0.0);
    for (size_t i = 0; i < rows.size(); i++) {
        out[rows[i]] += vals[i] * vec[cols[i]]; 
    }
    return out;
}

TEST_CASE("matvec") {
    SparseMat m{{0,1,1}, {1,0,1}, {2.0, 1.0, 3.0}};
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
        
void p2p(const Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    size_t n_obs = obs_n.end - obs_n.start;
    size_t n_src = src_n.end - src_n.start;
    for (size_t i = 0; i < n_obs; i++) {
        for (size_t j = 0; j < n_src; j++) {
            ws.result.p2p.rows.push_back(obs_n.start + i);
            ws.result.p2p.cols.push_back(src_n.start + j);
        }
    }
    auto old_n_vals = ws.result.p2p.vals.size(); 
    ws.result.p2p.vals.resize(old_n_vals + n_obs * n_src);
    ws.cfg.kernel.direct_nbody(
        &ws.obs_tree.pts[obs_n.start],
        &ws.src_tree.pts[src_n.start],
        n_obs, n_src, &ws.result.p2p.vals[old_n_vals]
    );
}

void m2p(Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    size_t n_src = ws.cfg.surf.size();
    size_t n_obs = obs_n.end - obs_n.start;
    for (size_t i = 0; i < n_obs; i++) {
        for (size_t j = 0; j < n_src; j++) {
            ws.result.m2p.rows.push_back(obs_n.start + i);
            ws.result.m2p.cols.push_back(src_n.idx * n_src + j);
        }
    }
    auto old_n_vals = ws.result.m2p.vals.size(); 
    ws.result.m2p.vals.resize(old_n_vals + n_obs * n_src);
    ws.cfg.kernel.direct_nbody(
        &ws.obs_tree.pts[obs_n.start],
        ws.get_equiv_surf(src_n).data(),
        n_obs, n_src, &ws.result.m2p.vals[old_n_vals]
    );
}

struct C2E {
    std::vector<Vec3> check_surf; 
    std::vector<double> op;
};

C2E check_to_equiv(Workspace& ws, const KDNode& src_n) {
    // equiv surface to check surface
    auto& equiv_surf = ws.get_equiv_surf(src_n);
    auto check_surf = inscribe_surf(src_n.bounds, ws.cfg.check_r, ws.cfg.surf);
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
    auto svd = svd_decompose(equiv_to_check.data(), n_surf);
    const double truncation_threshold = 1e-15;
    set_threshold(svd, truncation_threshold);
    return {check_surf, svd_pseudoinverse(svd)};
}

void p2m(Workspace& ws, const KDNode& src_n, C2E& c2e) {

    auto n_surf = ws.cfg.surf.size();
    auto n_pts = src_n.end - src_n.start;

    // points to check surface
    std::vector<double> pts_to_check(n_pts * n_surf);
    ws.cfg.kernel.direct_nbody(
         c2e.check_surf.data(), &ws.src_tree.pts[src_n.start],
         n_surf, n_pts, pts_to_check.data()
    );

    // multiply with points to check operator
    auto p2m_op = mat_mult(n_surf, n_pts, false, c2e.op, false, pts_to_check);
    
    for (size_t i = 0; i < n_surf; i++) {
        for (size_t j = 0; j < n_pts; j++) {
            ws.result.p2m.rows.push_back(src_n.idx * n_surf + i);
            ws.result.p2m.cols.push_back(src_n.start + j);
            ws.result.p2m.vals.push_back(p2m_op[i * n_pts + j]); 
        }
    }
}

void m2m(Workspace& ws, const KDNode& parent_n, const KDNode& child_n, C2E& c2e) {
    auto n_surf = c2e.check_surf.size();

    std::vector<double> child_to_check(n_surf * n_surf);
    ws.cfg.kernel.direct_nbody(
        c2e.check_surf.data(), ws.get_equiv_surf(child_n).data(), n_surf,
        n_surf, child_to_check.data()
    );

    auto m2m_op = mat_mult(n_surf, n_surf, false, c2e.op, false, child_to_check);

    for (size_t i = 0; i < n_surf; i++) {
        for (size_t j = 0; j < n_surf; j++) {
            ws.result.m2m.rows.push_back(parent_n.idx * n_surf + i);
            ws.result.m2m.cols.push_back(child_n.idx * n_surf + j);
            ws.result.m2m.vals.push_back(m2m_op[i * n_surf + j]);
        }
    }
}

void m2m_identity(Workspace& ws, const KDNode& src_n) {
    auto n_surf = ws.cfg.surf.size();
    for (size_t i = 0; i < n_surf; i++) {
        auto idx = src_n.idx * n_surf + i;
        ws.result.m2m.rows.push_back(idx);
        ws.result.m2m.cols.push_back(idx);
        ws.result.m2m.vals.push_back(-1.0);
    }
}

void traverse(Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto r_src = src_n.bounds.r;
    auto r_obs = obs_n.bounds.r;
    auto sep = hypot(sub(obs_n.bounds.center, src_n.bounds.center));

    if (r_src + r_obs < ws.cfg.mac * sep) {
        bool small_src = src_n.end - src_n.start < ws.cfg.surf.size();
        // bool small_obs = obs_n.end - obs_n.start < ws.cfg.surf.size();
        if (small_src) {
            p2p(ws, obs_n, src_n);
        } else {
            m2p(ws, obs_n, src_n);
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

void up_collect(Workspace& ws, const KDNode& src_n) {
    m2m_identity(ws, src_n);

    auto c2e = check_to_equiv(ws, src_n);

    if (src_n.is_leaf) {
        p2m(ws, src_n, c2e);
    } else {
        for (int i = 0; i < 2; i++) {
            auto child = ws.src_tree.nodes[src_n.children[i]];
            up_collect(ws, child);
            m2m(ws, src_n, child, c2e);
        }
    }
}

FMMMat fmmmmmmm(const KDTree& obs_tree, const KDTree& src_tree, const FMMConfig& cfg) {
    FMMMat result;
    Workspace ws(result, obs_tree, src_tree, cfg);
    traverse(ws, obs_tree.root(), src_tree.root());
    up_collect(ws, src_tree.root());
    return result;
}

}
