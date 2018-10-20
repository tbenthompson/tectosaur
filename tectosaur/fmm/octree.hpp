#pragma once

#include <array>
#include <vector>
#include "tree_helpers.hpp"

template <size_t dim>
struct OctreeNode {
    static const size_t split = 2<<(dim-1);

    size_t start;
    size_t end;
    Ball<dim> bounds;
    bool is_leaf;
    int height;
    int depth;
    size_t idx;
    std::array<size_t,split> children;
};

template <size_t dim>
std::array<int,OctreeNode<dim>::split+1> octree_partition(
        const Ball<dim>& bounds, BallWithIdx<dim>* start, BallWithIdx<dim>* end);

template <size_t dim>
Ball<dim> bounding_ball(BallWithIdx<dim>* balls, size_t n_balls);

template <size_t dim>
struct Octree;

template <size_t dim>
Octree<dim> build_octree(std::array<double,dim>* in_balls, double* in_R,
    size_t n_balls, size_t n_per_cell);

template <size_t _dim>
struct Octree {
    constexpr static size_t dim = _dim;
    constexpr static size_t split = 2 << (dim - 1);
    constexpr static auto build_fnc = build_octree<dim>;
    using Node = OctreeNode<dim>;

    std::vector<Ball<dim>> balls;
    std::vector<size_t> orig_idxs;

    int max_height;
    std::vector<OctreeNode<dim>> nodes;

    const OctreeNode<dim>& root() const { return nodes.front(); }
};
