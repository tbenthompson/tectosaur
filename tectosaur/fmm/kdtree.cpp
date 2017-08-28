#include "kdtree.hpp"
#include <algorithm>
#include <cassert>
#include <iostream>

template <size_t dim>
KDTree<dim> build_kdtree(std::array<double,dim>* in_pts, size_t n_pts, size_t n_per_cell) {
    auto pts_idxs = combine_pts_idxs(in_pts, n_pts);
    auto bounds = root_tree_bounds(pts_idxs.data(), n_pts);

    KDTree<dim> out;
    add_node(out, 0, n_pts, 0, n_per_cell, 0, bounds, pts_idxs);

    out.max_height = out.nodes[0].height;

    out.pts.resize(n_pts);
    out.orig_idxs.resize(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        out.pts[i] = pts_idxs[i].pt;
        out.orig_idxs[i] = pts_idxs[i].orig_idx;
    }

    return out;
}

template <size_t dim>
size_t add_node(KDTree<dim>& tree, size_t start, size_t end, int split_dim, size_t n_per_cell,
        int depth, Ball<dim> bounds, std::vector<PtWithIdx<dim>>& temp_pts) 
{
    bool is_leaf = end - start <= n_per_cell; 
    auto n_idx = tree.nodes.size();
    tree.nodes.push_back({start, end, bounds, is_leaf, 0, depth, n_idx, {}});
    if (!is_leaf) {
        auto split_pt = std::partition(
            temp_pts.data() + start, temp_pts.data() + end,
            [&] (const PtWithIdx<dim>& v) {
                return v.pt[split_dim] < bounds.center[split_dim]; 
            }
        );
        auto split_idx = static_cast<size_t>(split_pt - temp_pts.data());
        std::array<size_t,3> splits = {start, split_idx, end};

        int max_child_height = 0;
        for (size_t which_half = 0; which_half < 2; which_half++) {
            auto child_start = splits[which_half];
            auto child_end = splits[which_half + 1];
            auto child_n_pts = child_end - child_start;
            auto child_bounds = child_tree_bounds(&temp_pts[child_start], child_n_pts, bounds);
            auto child_node_idx = add_node(
                tree, child_start, child_end, (split_dim + 1) % dim,
                n_per_cell, depth + 1, child_bounds, temp_pts
            );
            tree.nodes[n_idx].children[which_half] = child_node_idx;
            max_child_height = std::max(max_child_height, tree.nodes[child_node_idx].height);
        }
        tree.nodes[n_idx].height = max_child_height + 1;
    }
    return n_idx;
}

template struct KDTree<2>;
template struct KDTree<3>;
template KDTree<2> build_kdtree(std::array<double,2>* in_pts, size_t n_pts, size_t n_per_cell);
template KDTree<3> build_kdtree(std::array<double,3>* in_pts, size_t n_pts, size_t n_per_cell);
