#include "octree.hpp"

#include <algorithm>

namespace tectosaur {

Box bounding_box(Vec3* pts, size_t n_pts) {
    if (n_pts == 0) {
        return {{0,0,0}, {0,0,0}};
    }
    auto min_corner = pts[0];
    auto max_corner = pts[0];
    for (size_t i = 1; i < n_pts; i++) {
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

Box kd_bounds(Vec3* pts, size_t n_pts, double parent_size) {
    auto box = bounding_box(pts, n_pts);

    // Limit sides to being 1 / 50 times the side length of the parent cell
    Vec3 hw;
    for (int d = 0; d < 3; d++) {
        hw[d] = std::max(parent_size / 50, box.half_width[d]);
    }
    return {box.center, hw};
}

KDTree::KDTree(Vec3* in_pts, Vec3* in_normals, size_t in_n_pts, size_t n_per_cell):
    pts(in_pts),
    normals(in_normals),
    n_pts(in_n_pts)
{
    add_node(0, in_n_pts, 0, n_per_cell, 1.0);
}

size_t KDTree::add_node(size_t start, size_t end, int split_dim,
    size_t n_per_cell, double parent_size) 
{
    auto bounds = kd_bounds(pts + start, end - start, parent_size);

    if (end - start <= n_per_cell) {
        nodes.push_back({start, end, bounds, true, nodes.size(), {0, 0}});
        return nodes.back().idx;
    } else {
        auto split = std::partition(
            pts + start, pts + end,
            [&] (const Vec3& v) { return v[split_dim] < bounds.center[split_dim]; }
        );
        auto n_idx = nodes.size();
        nodes.push_back({start, end, bounds, false, n_idx, {0, 0}});
        auto l = add_node(
            start, split - pts, (split_dim + 1) % 3,
            n_per_cell, bounds.half_width[0]
        );
        auto r = add_node(
            split - pts, end, (split_dim + 1) % 3,
            n_per_cell, bounds.half_width[0]
        );
        nodes[n_idx].children = {l, r};
        return n_idx;
    }
}

} //end namespace tectosaur
