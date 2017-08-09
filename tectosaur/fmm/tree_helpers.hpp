#pragma once
#include "geometry.hpp"

template <size_t dim>
struct PtWithIdx {
    std::array<double,dim> pt;
    size_t orig_idx;
};

// template <size_t dim>
// Ball<dim> tree_bounds(PtWithIdx<dim>* pts, size_t n_pts) {
//     std::array<double,dim> com{};
// 
//     for (size_t i = 0; i < n_pts; i++) {
//         for (size_t d = 0; d < dim; d++) {
//             com[d] += pts[i].pt[d];
//         }
//     }
//     for (size_t d = 0; d < dim; d++) {
//         com[d] /= n_pts;
//     }
// 
//     double max_r2 = 0.0;
//     for (size_t i = 0; i < n_pts; i++) {
//         for (size_t d = 0; d < dim; d++) {
//             max_r2 = std::max(max_r2, dist2(pts[i].pt, com));
//         }
//     }
// 
//     return {com, std::sqrt(max_r2)};
// }

template <size_t dim>
std::pair<size_t,double> furthest_pt(PtWithIdx<dim>* pts, size_t n_pts,
    const std::array<double,dim>& from)
{
    size_t furthest_idx = 0;
    double furthest_dist2 = dist2(from, pts[furthest_idx].pt);
    for (size_t i = 1; i < n_pts; i++) {
        double d2 = dist2(from, pts[i].pt);
        if (d2 > furthest_dist2) {
            furthest_idx = i;
            furthest_dist2 = d2;
        }
    }
    return {furthest_idx, furthest_dist2};
}

template <size_t dim>
Ball<dim> tree_bounds(PtWithIdx<dim>* pts, size_t n_pts) {
    if (n_pts == 1) {
        return {pts[0].pt, 0};
    }
    auto x = pts[0].pt;
    auto y = furthest_pt(pts, n_pts, x);
    auto z = furthest_pt(pts, n_pts, pts[y.first].pt);
    auto center = mult(add(pts[y.first].pt, pts[z.first].pt), 0.5);
    auto w = furthest_pt(pts, n_pts, center);
    auto R = std::sqrt(w.second);
    return {center, R};
}

template <size_t dim>
std::vector<PtWithIdx<dim>> combine_pts_idxs(std::array<double,dim>* pts, size_t n_pts) {
    std::vector<PtWithIdx<dim>> pts_idxs(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        pts_idxs[i] = {pts[i], i};
    }
    return pts_idxs;
}
