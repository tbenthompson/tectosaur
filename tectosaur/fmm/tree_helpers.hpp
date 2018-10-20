#pragma once
#include "geometry.hpp"
#include <cassert>
#include <iostream>

template <size_t dim>
struct BallWithIdx {
    Ball<dim> ball;
    size_t orig_idx;
};

// Note: I tried using a minimum bounding sphere algorithm. This didn't help and actually hurt a little bit(?!). Furthermore, the tree construction cost went up quite a lot. On second thought, this makes sense, because there's probably some value in having the center of a node at its center of mass for the sake of the error in the translation operator. Also, for most nodes, the center of mass approach will be quite close to the minimum bounding sphere.
template <size_t dim>
Ball<dim> tree_bounds(BallWithIdx<dim>* balls, size_t n_balls) {
    if (n_balls < 2) {
        throw std::runtime_error("Cannot form bounds for n_balls = " + std::to_string(n_balls));
    }

    std::array<double,dim> com{};

    for (size_t i = 0; i < n_balls; i++) {
        for (size_t d = 0; d < dim; d++) {
            com[d] += balls[i].ball.center[d];
        }
    }
    for (size_t d = 0; d < dim; d++) {
        com[d] /= n_balls;
    }

    double max_r = 0.0;
    for (size_t i = 0; i < n_balls; i++) {
        max_r = std::max(
            max_r,
            dist(balls[i].ball.center, com) + balls[i].ball.R
        ); 
    }
    return {com, max_r};
}

template <size_t dim>
Ball<dim> root_tree_bounds(BallWithIdx<dim>* balls, size_t n_balls) {
    if (n_balls == 0) {
        return {std::array<double,dim>{}, 1.0};
    } else if (n_balls == 1) {
        return balls[0].ball;
    } else {
        return tree_bounds(balls, n_balls);
    }
}

template <size_t dim>
Ball<dim> child_tree_bounds(BallWithIdx<dim>* balls, size_t n_balls, const Ball<dim>& parent_bounds) {
    if (n_balls == 0) {
        return {parent_bounds.center, parent_bounds.R / 50.0};
    } else if (n_balls == 1) {
        return balls[0].ball; 
    } else {
        return tree_bounds(balls, n_balls);
    }
}

template <size_t dim>
std::vector<BallWithIdx<dim>> combine_balls_idxs(
        std::array<double,dim>* balls, double* Rs, size_t n_balls) 
{
    std::vector<BallWithIdx<dim>> balls_idxs(n_balls);
    for (size_t i = 0; i < n_balls; i++) {
        balls_idxs[i] = {{balls[i], Rs[i]}, i};
    }
    return balls_idxs;
}
