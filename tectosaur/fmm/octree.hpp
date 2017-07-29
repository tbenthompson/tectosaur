#pragma once

#include <array>
#include <vector>
#include <memory>
#include "geometry.hpp"

template <size_t dim>
struct OctreeNode {
    static const size_t split = 2<<(dim-1);

    size_t start;
    size_t end;
    Cube<dim> bounds;
    bool is_leaf;
    int height;
    int depth;
    size_t idx;
    std::array<size_t,split> children;
};

template <size_t dim>
struct PtNormal {
    std::array<double,dim> pt;
    std::array<double,dim> normal;
    size_t orig_idx;
};

template <size_t dim>
std::array<int,OctreeNode<dim>::split+1> octree_partition(
        const Cube<dim>& bounds, PtNormal<dim>* start, PtNormal<dim>* end);

template <size_t dim>
std::vector<PtNormal<dim>> combine_pts_normals(std::array<double,dim>* pts,
        std::array<double,dim>* normals, size_t n_pts);

template <size_t dim>
Cube<dim> bounding_box(PtNormal<dim>* pts, size_t n_pts);

template <size_t dim>
struct Octree {
    std::vector<std::array<double,dim>> pts;
    std::vector<std::array<double,dim>> normals;
    std::vector<size_t> orig_idxs;

    size_t n_pts;
    int max_height;
    std::vector<OctreeNode<dim>> nodes;

    const OctreeNode<dim>& root() const { return nodes.front(); }

    Octree(std::array<double,dim>* in_pts, std::array<double,dim>* in_normals,
            size_t n_pts, size_t n_per_cell);

    size_t add_node(size_t start, size_t end, 
        size_t n_per_cell, int depth, Cube<dim> bounds,
        std::vector<PtNormal<dim>>& temp_pts);
};
