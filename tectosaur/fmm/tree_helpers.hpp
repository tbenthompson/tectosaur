#pragma once
#include "geometry.hpp"
#include <cassert>
#include <iostream>

template <size_t dim>
struct PtWithIdx {
    std::array<double,dim> pt;
    size_t orig_idx;
};

// Note: I tried using a minimum bounding sphere algorithm. This didn't help and actually hurt a little bit(?!). Furthermore, the tree construction cost went up quite a lot. On second thought, this makes sense, because there's probably some value in having the center of a node at its center of mass for the sake of the error in the translation operator. Also, for most nodes, the center of mass approach will be quite close to the minimum bounding sphere.
template <size_t dim>
Ball<dim> tree_bounds(PtWithIdx<dim>* pts, size_t n_pts) {
    if (n_pts < 2) {
        throw std::runtime_error("Cannot form bounds for n_pts = " + std::to_string(n_pts));
    }

    std::array<double,dim> com{};

    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            com[d] += pts[i].pt[d];
        }
    }
    for (size_t d = 0; d < dim; d++) {
        com[d] /= n_pts;
    }

    double max_r2 = 0.0;
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            max_r2 = std::max(max_r2, dist2(pts[i].pt, com));
        }
    }
    return {com, std::sqrt(max_r2)};
}

template <size_t dim>
Ball<dim> root_tree_bounds(PtWithIdx<dim>* pts, size_t n_pts) {
    if (n_pts == 0) {
        return {std::array<double,dim>{}, 1.0};
    } else if (n_pts == 1) {
        return {pts[0].pt, 1.0};
    } else {
        return tree_bounds(pts, n_pts);
    }
}

template <size_t dim>
Ball<dim> child_tree_bounds(PtWithIdx<dim>* pts, size_t n_pts, const Ball<dim>& parent_bounds) {
    if (n_pts == 0) {
        return {parent_bounds.center, parent_bounds.R / 50.0};
    } else if (n_pts == 1) {
        return {pts[0].pt, parent_bounds.R / 50.0}; 
    } else {
        return tree_bounds(pts, n_pts);
    }
}

template <size_t dim>
std::vector<PtWithIdx<dim>> combine_pts_idxs(std::array<double,dim>* pts, size_t n_pts) {
    std::vector<PtWithIdx<dim>> pts_idxs(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        pts_idxs[i] = {pts[i], i};
    }
    return pts_idxs;
}
