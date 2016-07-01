#include "fmm_impl.hpp"
#include "blas_wrapper.hpp"
#include "lib/doctest.h"
#include "test_helpers.hpp"
#include <cmath>
#include <limits>
#include <iostream>

namespace tectosaur {

std::vector<Vec3> inscribe_surf(Box b, double scaling, 
    const std::vector<Vec3>& fmm_surf) 
{
    std::vector<Vec3> out(fmm_surf.size());
    for (size_t i = 0; i < fmm_surf.size(); i++) {
        for (size_t d = 0; d < 3; d++) {
            out[i][d] = fmm_surf[i][d] * b.half_width[d] * scaling + b.center[d];
        }
    }
    return out;
}

TEST_CASE("inscribe") {
    auto s = inscribe_surf({{1,1,1},{2,2,2}}, 0.5, {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}});
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
    std::vector<bool> multipoles_needed;
    size_t next_m_idx = 0;
    const static size_t max_m_idx = std::numeric_limits<size_t>::max();

    Workspace(FMMMat& result, const KDTree& obs_tree,
        const KDTree& src_tree, const FMMConfig& cfg):
        result(result), obs_tree(obs_tree), src_tree(src_tree), cfg(cfg),
        equiv_surfs(src_tree.nodes.size()),
        multipoles_needed(src_tree.nodes.size(), false)
    {
        result.multipole_starts.resize(src_tree.nodes.size());
        for (auto& ms: result.multipole_starts) { ms = max_m_idx; }
    }

    size_t get_multipole_start(const KDNode& n) {
        auto& start = result.multipole_starts[n.idx];
        if (start == max_m_idx) {
            start = next_m_idx;
            next_m_idx += cfg.surf.size();
        }
        return start;
    }

    void need_multipoles(const KDNode& n) {
        multipoles_needed[n.idx] = true;
    }

    const std::vector<Vec3>& get_equiv_surf(const KDNode& src_n) {
        if (equiv_surfs[src_n.idx].size() == 0) {
            equiv_surfs[src_n.idx] = inscribe_surf(src_n.bounds, cfg.equiv_r, cfg.surf);
        }
        return equiv_surfs[src_n.idx];
    }
};

double hypot(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

double one(const Vec3& obs, const Vec3& src) { return 1.0; }

double inv_r(const Vec3& obs, const Vec3& src) {
    auto d = sub(obs, src);
    auto r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    return 1.0 / std::sqrt(r2);
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
            ws.result.m2p.cols.push_back(ws.get_multipole_start(src_n) + j);
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
    auto svd = svd_decompose(equiv_to_check);
    const double truncation_threshold = 1e-13;
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
            ws.result.p2m.rows.push_back(ws.get_multipole_start(src_n) + i);
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
            ws.result.m2m.rows.push_back(ws.get_multipole_start(parent_n) + i);
            ws.result.m2m.cols.push_back(ws.get_multipole_start(child_n) + j);
            ws.result.m2m.vals.push_back(m2m_op[i * n_surf + j]);
        }
    }
}

void m2m_identity(Workspace& ws, const KDNode& src_n) {
    for (size_t i = 0; i < ws.cfg.surf.size(); i++) {
        auto idx = ws.get_multipole_start(src_n) + i;
        ws.result.m2m.rows.push_back(idx);
        ws.result.m2m.cols.push_back(idx);
        ws.result.m2m.vals.push_back(-1.0);
    }
}

void traverse(Workspace& ws, const KDNode& obs_n, const KDNode& src_n) {
    auto r_src = hypot(src_n.bounds.half_width);
    auto r_obs = hypot(obs_n.bounds.half_width);
    double r_max = std::max(r_src, r_obs);
    double r_min = std::min(r_src, r_obs);
    auto sep = hypot(sub(obs_n.bounds.center, src_n.bounds.center));

    if (r_max + ws.cfg.mac * r_min <= ws.cfg.mac * sep) {
        if (src_n.end - src_n.start < ws.cfg.surf.size()) {
            p2p(ws, obs_n, src_n);
        } else {
            ws.need_multipoles(src_n);
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

void leaf_multipoles(Workspace& ws, const KDNode& src_n) {
    m2m_identity(ws, src_n);
    auto c2e = check_to_equiv(ws, src_n);
    p2m(ws, src_n, c2e);
}

bool up_collect(Workspace& ws, const KDNode& src_n) {
    if (src_n.is_leaf) {
        if (ws.multipoles_needed[src_n.idx]) {
            leaf_multipoles(ws, src_n);
            return true;
        }
        return false;
    } else {
        std::array<bool,2> child_has_ms;
        for (int i = 0; i < 2; i++) {
            child_has_ms[i] = up_collect(ws, ws.src_tree.nodes[src_n.children[i]]);
        }
        if (child_has_ms[0] || child_has_ms[1]) {
            m2m_identity(ws, src_n);
            auto c2e = check_to_equiv(ws, src_n);
            for (int i = 0; i < 2; i++) {
                auto child = ws.src_tree.nodes[src_n.children[i]];
                if (!child_has_ms[i]) {
                    leaf_multipoles(ws, child);
                }
                m2m(ws, src_n, child, c2e);
            }
            return true;
        }
        if (!ws.multipoles_needed[src_n.idx]) {
            return false;
        }
        leaf_multipoles(ws, src_n);
        return true;
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
