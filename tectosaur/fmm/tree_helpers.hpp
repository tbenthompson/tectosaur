#pragma once
#include "geometry.hpp"

template <size_t dim>
struct PtWithIdx {
    std::array<double,dim> pt;
    size_t orig_idx;
};

// Note: I tried using a minimum bounding sphere algorithm. This didn't help and actually hurt a little bit(?!). Furthermore, the tree construction cost went up quite a lot. On second thought, this makes sense, because there's probably some value in having the center of a node at its center of mass for the sake of the error in the translation operator. Also, for most nodes, the center of mass approach will be quite close to the minimum bounding sphere.
template <size_t dim>
Ball<dim> tree_bounds(PtWithIdx<dim>* pts, size_t n_pts) {
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
std::vector<PtWithIdx<dim>> combine_pts_idxs(std::array<double,dim>* pts, size_t n_pts) {
    std::vector<PtWithIdx<dim>> pts_idxs(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        pts_idxs[i] = {pts[i], i};
    }
    return pts_idxs;
}
