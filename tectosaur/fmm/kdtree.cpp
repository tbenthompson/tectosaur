#include "kdtree.hpp"

#include <algorithm>
#include <iostream>

namespace tectosaur {

std::ostream& operator<<(std::ostream& os, const Vec3& v) {
    os << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    return os;
}

Sphere kd_bounds(PtNormal* pts, size_t n_pts, double parent_size) {
    Vec3 center_of_mass = {0.0, 0.0, 0.0};
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < 3; d++) {
            center_of_mass[d] += pts[i].pt[d];
        }
    }
    for (size_t d = 0; d < 3; d++) {
        center_of_mass[d] /= n_pts;
    }

    double max_r = 0.0;
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < 3; d++) {
            max_r = std::max(max_r, hypot(sub(pts[i].pt, center_of_mass)));
        }
    }

    // Limit sides to being 1 / 50 times the side length of the parent cell
    const static double side_ratio = 1.0 / 50.0;
    return {center_of_mass, std::max(parent_size * side_ratio, max_r)};
}

KDTree::KDTree(Vec3* in_pts, Vec3* in_normals,
        size_t n_pts, size_t n_per_cell):
    pts(n_pts),
    normals(n_pts),
    orig_idxs(n_pts),
    n_pts(n_pts)
{
    std::vector<PtNormal> pts_normals(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        pts_normals[i] = {in_pts[i], in_normals[i], i};
    }
    size_t n_leaves = n_pts / n_per_cell;
    // For n leaves in a binary tree, there should be 2*n total nodes. Since
    // n_leaves is just an estimate, overallocate by 50% with 3*n total nodes.
    nodes.reserve(3 * n_leaves);
    add_node(0, n_pts, 0, n_per_cell, 1.0, 0, pts_normals);
    max_height = nodes[0].height;

    for (size_t i = 0; i < n_pts; i++) {
        pts[i] = pts_normals[i].pt;
        normals[i] = pts_normals[i].normal;
        orig_idxs[i] = pts_normals[i].orig_idx;
    }
}

size_t KDTree::add_node(size_t start, size_t end, int split_dim,
    size_t n_per_cell, double parent_size, int depth, std::vector<PtNormal>& temp_pts)
{
    auto bounds = kd_bounds(temp_pts.data() + start, end - start, parent_size);

    if (end - start <= n_per_cell) {
        nodes.push_back({start, end, bounds, true, 0, depth, nodes.size(), {0, 0}});
        return nodes.back().idx;
    } else {
        auto split = std::partition(
            temp_pts.data() + start, temp_pts.data() + end,
            [&] (const PtNormal& v) {
                return v.pt[split_dim] < bounds.center[split_dim]; 
            }
        );
        auto n_idx = nodes.size();
        nodes.push_back({start, end, bounds, false, 0, depth, n_idx, {0, 0}});
        auto l = add_node(
            start, split - temp_pts.data(), (split_dim + 1) % 3,
            n_per_cell, bounds.r, depth + 1, temp_pts
        );
        auto r = add_node(
            split - temp_pts.data(), end, (split_dim + 1) % 3,
            n_per_cell, bounds.r, depth + 1, temp_pts
        );
        nodes[n_idx].children = {l, r};
        nodes[n_idx].height = std::max(nodes[l].height, nodes[r].height) + 1;
        return n_idx;
    }
}

} //end namespace tectosaur
