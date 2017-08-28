#pragma once

#include <array>
#include <vector>
#include "tree_helpers.hpp"

template <size_t dim>
struct KDNode {
    size_t start;
    size_t end;
    Ball<dim> bounds;
    bool is_leaf;
    int height;
    int depth;
    size_t idx;
    std::array<size_t,2> children;
};

template <size_t dim>
std::array<int,2> kd_partition(const Ball<dim>& bounds, int split_dim,
    PtWithIdx<dim>* start, PtWithIdx<dim>* end);

template <size_t dim>
struct KDTree;

template <size_t dim>
KDTree<dim> build_kdtree(std::array<double,dim>* in_pts, size_t n_pts, size_t n_per_cell);

template <size_t _dim>
struct KDTree {
    constexpr static size_t dim = _dim;
    constexpr static size_t split = 2;
    constexpr static auto build_fnc = build_kdtree<dim>;
    using Node = KDNode<dim>;

    std::vector<std::array<double,dim>> pts;
    std::vector<size_t> orig_idxs;

    int max_height;
    std::vector<KDNode<dim>> nodes;

    const KDNode<dim>& root() const { return nodes.front(); }
};
