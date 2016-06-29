#include "fmm_impl.hpp"
#include "blas_wrapper.hpp"
#include "doctest.h"
#include "test_helpers.hpp"
#include <cmath>

#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>

namespace tectosaur {

std::vector<double> kernel(std::vector<Vec3> obs_pts, std::vector<Vec3> src_pts) {
    std::vector<double> out(obs_pts.size() * src_pts.size());
    for (size_t i = 0; i < obs_pts.size(); i++) {
        for (size_t j = 0; j < src_pts.size(); j++) {
            auto dx = obs_pts[i][0] - src_pts[j][0];
            auto dy = obs_pts[i][1] - src_pts[j][1];
            auto dz = obs_pts[i][2] - src_pts[j][2];
            out[i * src_pts.size() + j] = 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz);
        }
    }
    return out;
}

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

UpwardNode::UpwardNode(OctreeNode::Ptr& n, FMMSurf fmm_surf):
    node(n)
{
    auto check_surf = inscribe_surf(n->bounds, std::sqrt(2), *fmm_surf);

    // equiv surface to check surface
    equiv_surf = inscribe_surf(n->bounds, 0.3, *fmm_surf);
    auto equiv_to_check = kernel(equiv_surf, check_surf);

    // invert equiv to check operator
    // In some cases, the equivalent surface to check surface operator
    // is poorly conditioned. In this case, truncate the singular values 
    // to solve a regularized least squares version of the problem.
    auto svd = svd_decompose(equiv_to_check);
    const double truncation_threshold = 1e-13;
    set_threshold(svd, truncation_threshold);
    auto check_to_equiv = svd_pseudoinverse(svd);

    // make this the constructor
    if (n->is_leaf) {
        if (n->data.pts.size() > 0) {
            // points to check surface
            auto pts_to_check = kernel(n->data.pts, check_surf);

            // multiply with points to check operator
            p2m_op = mat_mult(
                n->data.pts.size(), fmm_surf->size(),
                false, pts_to_check, false, check_to_equiv
            );
        }
    } else {
        for (size_t i = 0; i < 8; i++) {
            children[i] = n->children[i].then([] (FMMSurf fmm_surf, 
                OctreeNode::Ptr& child) 
            {
                return std::make_shared<UpwardNode>(child, fmm_surf);
            }, fmm_surf);
            m2m_ops[i] = children[i].then(
                [] (std::vector<Vec3>& check_surf,
                    std::vector<double>& check_to_equiv, UpwardNode::Ptr& child) 
                {
                    // child equiv surf to parent check surface
                    auto child_to_check = kernel(child->equiv_surf, check_surf);

                    return mat_mult(
                        child->equiv_surf.size(), check_surf.size(),
                        false, child_to_check, false, check_to_equiv
                    );
                },
                std::move(check_surf),
                std::move(check_to_equiv)
            );
        }
    }
}

Upward up_up_up(Octree o, FMMSurf fmm_surf) {
    auto result = o.root
        .then([] (FMMSurf fmm_surf, OctreeNode::Ptr& n) {
            return std::make_shared<UpwardNode>(n, fmm_surf); 
        }, fmm_surf);
    return {result};
}

double hypot(const Vec3& v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

template <typename ObsF, typename SrcF>
void insert(SparseMat& result, size_t n_src, 
    std::vector<double> K, ObsF obs_idx_fnc, SrcF src_idx_fnc) 
{
    if (n_src == 0) {
        return;
    }
    for (size_t i = 0; i < K.size() / n_src; i++) {
        for (size_t j = 0; j < n_src; j++) {
            result.rows.push_back(obs_idx_fnc(i));
            result.cols.push_back(src_idx_fnc(j));
            result.vals.push_back(K[i * n_src + j]);
        }
    }
}

template <typename ObsF, typename SrcF>
void interact(SparseMat& result, const std::vector<Vec3>& obs_pts, 
    const std::vector<Vec3>& src_pts, ObsF obs_idx_fnc, SrcF src_idx_fnc)
{
    auto K = kernel(obs_pts, src_pts);
    insert(result, src_pts.size(), K, obs_idx_fnc, src_idx_fnc);
}

void m2p(SparseMat& result, UpwardNode::Ptr& src_node, OctreeNode::Ptr& obs_node) {
    interact(
        result, obs_node->data.pts, src_node->equiv_surf, 
        [&] (size_t i) { return obs_node->data.original_indices[i]; },
        [&] (size_t j) { return src_node->m_dof + j; }
    );
}

void p2p(SparseMat& result, UpwardNode::Ptr& src_node, OctreeNode::Ptr& obs_node) {
    interact(
        result, obs_node->data.pts, src_node->node->data.pts, 
        [&] (size_t i) { return obs_node->data.original_indices[i]; },
        [&] (size_t j) { return src_node->node->data.original_indices[j]; }
    );
}

void p2m(SparseMat& result, UpwardNode::Ptr& src_node) {
    insert(
        result, src_node->node->data.pts.size(), src_node->p2m_op, 
        [&] (size_t i) { return src_node->m_dof + i; },
        [&] (size_t j) { return src_node->node->data.original_indices[j]; }
    );
}

void m2m(SparseMat& result, UpwardNode::Ptr& parent_node, 
        UpwardNode::Ptr& child_node, int child_idx) 
{
    insert(
        result, parent_node->equiv_surf.size(),
        parent_node->m2m_ops[child_idx].get(),
        [&] (size_t i) { return parent_node->m_dof + i; },
        [&] (size_t j) { return child_node->m_dof + j; }
    );
}

void m2m_identity(SparseMat& result, UpwardNode::Ptr& src_node) {
    for (size_t i = 0; i < src_node->equiv_surf.size(); i++) {
        result.rows.push_back(src_node->m_dof + i);
        result.cols.push_back(src_node->m_dof + i);
        result.vals.push_back(1.0);
    }
}

void traverse(FMMMat& result, UpwardNode::Ptr& src_node, OctreeNode::Ptr& obs_node) { 
    auto r_src = hypot(src_node->node->bounds.half_width);
    auto r_obs = hypot(obs_node->bounds.half_width);
    double r_max = std::max(r_src, r_obs);
    double r_min = std::min(r_src, r_obs);
    auto sep = hypot(sub(obs_node->bounds.center, src_node->node->bounds.center));
    const double MAC = 0.3;
    if (r_max + MAC * r_min <= MAC * sep) {
        m2p(result.m2p, src_node, obs_node);
        return;
    }

    if (src_node->node->is_leaf && obs_node->is_leaf) {
        p2p(result.p2p, src_node, obs_node);
        return;
    }
    
    bool split_src = ((r_src < r_obs) && !src_node->node->is_leaf) || obs_node->is_leaf;
    if (split_src) {
        for (int i = 0; i < 8; i++) {
            traverse(result, src_node->children[i].get(), obs_node);
        }
    } else {
        for (int i = 0; i < 8; i++) {
            traverse(result, src_node, obs_node->children[i].get());
        }
    }
}

void up_collect(FMMMat& result, UpwardNode::Ptr& src_node) {
    src_node->m_dof = result.n_m_dofs;
    result.n_m_dofs += src_node->equiv_surf.size();
    m2m_identity(result.m2m, src_node);

    if (src_node->node->is_leaf) {
        p2m(result.p2m, src_node);
    } else {
        for (int i = 0; i < 8; i++) {
            auto child = src_node->children[i].get();
            up_collect(result, child);
            m2m(result.m2m, src_node, child, i);
        }
    }
}

FMMMat go_go_go(Upward src_tree, Octree obs_tree) {
    FMMMat result;
    up_collect(result, src_tree.root.get());
    traverse(result, src_tree.root.get(), obs_tree.root.get());
    return result;
}

}
