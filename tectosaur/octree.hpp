#pragma once
#include <array>
#include <vector>

namespace tectosaur {

using Vec3 = std::array<double,3>;

struct Box {
    Vec3 center;
    Vec3 half_width;
};

struct KDNode {
    size_t start;
    size_t end;
    Box bounds;
    bool is_leaf;
    size_t idx;
    std::array<size_t,2> children;
};

struct KDTree {
    Vec3* pts; // not owned, make sure lifetime exceeds kdtree lifetime
    Vec3* normals; // not owned, make sure lifetime exceeds kdtree lifetime
    size_t n_pts;
    std::vector<KDNode> nodes;

    const KDNode& root() const { return nodes.front(); }

    size_t add_node(
        size_t start, size_t end, int split_dim,
        size_t n_per_cell, double parent_size
    );
    KDTree(Vec3* pts, Vec3* normals, size_t n_pts, size_t n_per_cell);
};

} // end namespace tectosaur
