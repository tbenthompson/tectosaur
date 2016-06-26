#pragma once
#include <memory>
#include <array>

#include "taskloaf/future.hpp"

namespace tectosaur {

namespace tl = taskloaf;

using Vec3 = std::array<double,3>;

struct Box {
    Vec3 center;
    Vec3 half_width;
};

struct OctreeNode {
    Box bounds;
    std::vector<Vec3> pts;
    std::array<std::shared_ptr<tl::future<OctreeNode>>,8> children;

    OctreeNode() = default;
    OctreeNode(size_t max_pts_per_cell, std::vector<Vec3> pts);

    template <typename Archive>
    void serialize(Archive& ar) {}
};

tl::future<OctreeNode> make_node(size_t max_pts_per_cell, std::vector<Vec3> pts);

struct Octree {
    tl::future<OctreeNode> root;

    Octree(size_t max_pts_per_cell, std::vector<Vec3> pts);
};

} // end namespace tectosaur
