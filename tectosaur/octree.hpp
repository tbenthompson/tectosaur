#pragma once
#include <memory>
#include <array>

#include <cereal/types/memory.hpp>

#include "taskloaf/future.hpp"

namespace tectosaur {

namespace tl = taskloaf;

using Vec3 = std::array<double,3>;

struct Box {
    Vec3 center;
    Vec3 half_width;
};

struct NodeData {
    std::vector<size_t> original_indices;
    std::vector<Vec3> pts;
    std::vector<Vec3> normals;

    template <typename Archive>
    void serialize(Archive& ar) { throw std::runtime_error("undefined"); }
};

struct OctreeNode {
    using Ptr = std::shared_ptr<OctreeNode>;

    Box bounds;
    NodeData data;
    bool is_leaf = false;
    std::array<tl::future<OctreeNode::Ptr>,8> children;

    OctreeNode() = default;
    OctreeNode(size_t max_pts_per_cell, Box parent_bounds, NodeData in_data);

    template <typename Archive>
    void serialize(Archive& ar) { throw std::runtime_error("undefined"); }
};

struct Octree {
    tl::future<std::shared_ptr<OctreeNode>> root;

    Octree(size_t max_pts_per_cell, NodeData in_data);
};

int n_total_children(Octree& o);

} // end namespace tectosaur
