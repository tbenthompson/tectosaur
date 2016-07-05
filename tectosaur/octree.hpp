#pragma once
#include <array>
#include <vector>
#include <iosfwd>

namespace tectosaur {

using Vec3 = std::array<double,3>;

std::ostream& operator<<(std::ostream& os, const Vec3&);

Vec3 sub(const Vec3& a, const Vec3& b);
double hypot(const Vec3& v);

struct Sphere {
    Vec3 center;
    double r;
};

struct KDNode {
    size_t start;
    size_t end;
    Sphere bounds;
    bool is_leaf;
    int height;
    int depth;
    size_t idx;
    std::array<size_t,2> children;
};

struct KDTree {
    std::vector<Vec3> pts;
    std::vector<Vec3> normals;
    size_t n_pts;
    int max_height;
    int max_depth;
    std::vector<KDNode> nodes;

    const KDNode& root() const { return nodes.front(); }

    size_t add_node(
        size_t start, size_t end, int split_dim,
        size_t n_per_cell, double parent_size, int depth
    );
    KDTree(std::vector<Vec3> pts, std::vector<Vec3> normals,
        size_t n_pts, size_t n_per_cell);
};

} // end namespace tectosaur
