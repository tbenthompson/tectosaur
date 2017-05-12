#pragma once
#include <array>
#include <iosfwd>
#include <vector>
#include <cmath>

namespace tectosaur {

using Vec3 = std::array<double, 3>;

std::ostream& operator<<(std::ostream& os, const Vec3&);

inline double dot(const Vec3& a, const Vec3& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

inline Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

inline double hypot(const Vec3& v) {
    return std::sqrt(dot(v, v));
}

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
    std::array<size_t, 2> children;
};

struct PtNormal {
    Vec3 pt;
    Vec3 normal;
    size_t orig_idx;
};

struct KDTree {
    std::vector<Vec3> pts;
    std::vector<Vec3> normals;
    std::vector<size_t> orig_idxs;

    size_t n_pts;
    int max_height;
    std::vector<KDNode> nodes;

    const KDNode& root() const { return nodes.front(); }

    size_t add_node(size_t start, size_t end, int split_dim, size_t n_per_cell,
                    double parent_size, int depth, std::vector<PtNormal>& temp_pts);
    KDTree(Vec3* pts, Vec3* normals, size_t n_pts, size_t n_per_cell);
};

}  // end namespace tectosaur
