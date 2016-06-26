#include "fmm_impl.hpp"
#include "taskloaf.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include "doctest.h"

#include <random>

namespace tectosaur {

inline std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")" << "\n";
    return os;
}

Box bounding_box(const std::vector<Vec3>& pts) {
    if (pts.size() == 0) {
        return {{0,0,0}, {0,0,0}};
    }
    auto min_corner = pts[0];
    auto max_corner = pts[0];
    for (size_t i = 1; i < pts.size(); i++) {
        for (size_t d = 0; d < 3; d++) {
            min_corner[d] = std::min(min_corner[d], pts[i][d]);
            max_corner[d] = std::max(max_corner[d], pts[i][d]);
        }
    }
    Vec3 center;
    Vec3 half_width;
    for (size_t d = 0; d < 3; d++) {
        center[d] = (max_corner[d] + min_corner[d]) / 2.0;
        half_width[d] = (max_corner[d] - min_corner[d]) / 2.0;
    }
    return {center, half_width};
}

TEST_CASE("bounding_box") {
    auto b = bounding_box({{0,0,0},{2,2,2}});
    for (int d = 0; d < 3; d++) {
        CHECK(b.center[d] == 1.0); CHECK(b.half_width[d] == 1.0);
    }
}

TEST_CASE("bounding box no pts") {
    auto b = bounding_box({}); (void)b;
}

int find_containing_subcell(const Box& b, const Vec3& pt) {
    int child_idx = 0;
    for (size_t d = 0; d < 3; d++) {
        if (pt[d] > b.center[d]) {
            child_idx++; 
        }
        if (d < 2) {
            child_idx = child_idx << 1;
        }
    }
    return child_idx;
}

TEST_CASE("subcell") {
    CHECK(find_containing_subcell({{0,0,0},{1,1,1}}, {-0.5,0.5,0.5}) == 3);
    CHECK(find_containing_subcell({{0,0,0},{1,1,1}}, {0.5,-0.5,0.5}) == 5);
    CHECK(find_containing_subcell({{0,0,0},{1,1,1}}, {0.5,0.5,-0.5}) == 6);
    CHECK(find_containing_subcell({{0,0,0},{1,1,1}}, {0.5,0.5,0.5}) == 7);
}

Box get_subcell(const Box& parent, int idx) {
    auto new_halfwidth = parent.half_width;
    auto new_center = parent.center;
    for (int d = 2; d >= 0; d--) {
        new_halfwidth[d] /= 2.0;
        auto which_side = idx % 2;
        idx = idx >> 1;
        new_center[d] += ((static_cast<double>(which_side) * 2) - 1) * new_halfwidth[d];
    }
    return {new_center, new_halfwidth};
}

bool in_box(const Box& b, const Vec3& pt)
{
    bool in = true;
    for (size_t d = 0; d < 3; d++) {
        auto sep = std::fabs(pt[d] - b.center[d]);
        in = in && (sep <= b.half_width[d]);
    }
    return in;
}

TEST_CASE("get subcell") {
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);
    Box parent{{0,0,0},{1,1,1}};
    for (int i = 0; i < 100; i++) {
        Vec3 pt = {dis(gen), dis(gen), dis(gen)};
        auto idx = find_containing_subcell(parent, pt);
        auto subcell = get_subcell(parent, idx);
        REQUIRE(in_box(subcell, pt));
    }
}

OctreeNode::OctreeNode(Box in_bounds, size_t max_pts_per_cell, std::vector<Vec3> in_pts) {

    bounds = in_bounds;

    if (in_pts.size() < max_pts_per_cell) {

        pts = std::move(in_pts);
        return;
    }

    std::array<std::vector<Vec3>,8> child_pts{};
    for (size_t i = 0; i < in_pts.size(); i++) {
        auto child_idx = find_containing_subcell(bounds, in_pts[i]);
        child_pts[child_idx].push_back(std::move(in_pts[i]));
    }

    
    for (int child_idx = 0; child_idx < 8; child_idx++) {
        auto child_bounds = get_subcell(bounds, child_idx);
        children[child_idx] = std::make_shared<tl::future<OctreeNode>>(
            make_node(child_bounds, max_pts_per_cell, std::move(child_pts[child_idx]))
        );
    }
}

tl::future<OctreeNode> make_node(
    Box bounds, size_t max_pts_per_cell, std::vector<Vec3> pts) 
{
    return tl::task(
        [=] (std::vector<Vec3>& pts) {
            return OctreeNode(bounds, max_pts_per_cell, std::move(pts)); 
        },
        std::move(pts)
    );  
}

Octree::Octree(size_t max_pts_per_cell, std::vector<Vec3> pts) {
    auto bounds = bounding_box(pts);
    root = make_node(bounds, max_pts_per_cell, std::move(pts));
}

} //end namespace tectosaur
