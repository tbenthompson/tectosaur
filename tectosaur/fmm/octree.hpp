#pragma once

#include <array>
#include <vector>
#include "geometry.hpp"

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
        const Ball<dim>& bounds, PtWithIdx<dim>* start, PtWithIdx<dim>* end);

template <size_t dim>
Ball<dim> bounding_ball(PtWithIdx<dim>* pts, size_t n_pts);

template <size_t _dim>
struct Octree {
    constexpr static size_t dim = _dim;
    constexpr static size_t split = 2 << (dim - 1);
    using Node = OctreeNode<dim>;

    std::vector<std::array<double,dim>> pts;
    std::vector<size_t> orig_idxs;

    int max_height;
    std::vector<OctreeNode<dim>> nodes;

    const OctreeNode<dim>& root() const { return nodes.front(); }

    Octree(std::array<double,dim>* in_pts, size_t n_pts, size_t n_per_cell);

    size_t add_node(size_t start, size_t end, 
        size_t n_per_cell, int depth, Ball<dim> bounds,
        std::vector<PtWithIdx<dim>>& temp_pts);
};
