#include "fmm_impl.hpp"
#include "taskloaf.hpp"

#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>
#include "doctest.h"

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

OctreeNode::OctreeNode(size_t max_pts_per_cell, std::vector<Vec3> pts) {
    bounds = bounding_box(pts);

    if (pts.size() < max_pts_per_cell) {
        this->pts = std::move(pts);
        return;
    }

    std::array<std::vector<Vec3>,8> child_pts;
    for (size_t i = 0; i < pts.size(); i++) {
        auto child_idx = find_containing_subcell(bounds, pts[i]);
        child_pts[child_idx].push_back(pts[i]);
    }
    
    for (int child_idx = 0; child_idx < 8; child_idx++) {
        children[child_idx] = std::make_unique<tl::future<OctreeNode>>(
            make_node(max_pts_per_cell, std::move(child_pts[child_idx]))
        );
    }
}

tl::future<OctreeNode> make_node(size_t max_pts_per_cell, std::vector<Vec3> pts) {
    return tl::task(
        [=] (std::vector<Vec3>& pts) {
            return OctreeNode(max_pts_per_cell, std::move(pts)); 
        },
        std::move(pts)
    );  
}

Octree::Octree(size_t max_pts_per_cell, std::vector<Vec3> pts):
    root(make_node(max_pts_per_cell, std::move(pts)))
{}

} //end namespace tectosaur
