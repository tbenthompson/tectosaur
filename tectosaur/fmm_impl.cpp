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
    std::vector<UpwardNode::Ptr> ns{result.get()};
    int n_nodes = 0;
    while (ns.size() > 0) {
        auto n = ns.back();
        ns.pop_back();
        n_nodes++;
        if (!n->node->is_leaf) {
            for (int i = 0; i < 8; i++) {
                ns.push_back(n->children[i].get());
            }
        }
    }
    std::cout << "n_nodes: " << n_nodes << std::endl;
    return {result};
}

SparseMat go_go_go(Upward src_tree, Octree obs_tree) {
    return SparseMat{{},{},{}};
}

}
