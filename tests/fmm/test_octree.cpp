#include "include/doctest.h"
#include "include/test_helpers.hpp"
#include "octree.hpp"

#include <iostream>

TEST_CASE("containing subcell ball 2d") {
    Ball<2> b{{0, 0}, 1.0};
    REQUIRE(find_containing_subcell(b, {0.1, 0.1}) == 3);
    REQUIRE(find_containing_subcell(b, {0.1, -0.1}) == 2);
    REQUIRE(find_containing_subcell(b, {-0.1, -0.1}) == 0);
}

TEST_CASE("containing subcell ball 3d") {
    Ball<3> b{{0, 0, 0}, 1.0};
    REQUIRE(find_containing_subcell(b, {0.1, 0.1, 0.1}) == 7);
    REQUIRE(find_containing_subcell(b, {0.1, -0.1, -0.1}) == 4);
    REQUIRE(find_containing_subcell(b, {-0.1, -0.1, -0.1}) == 0);
}

TEST_CASE("make child idx 3d") {
    REQUIRE(make_child_idx<3>(0) == (std::array<size_t,3>{0, 0, 0}));
    REQUIRE(make_child_idx<3>(2) == (std::array<size_t,3>{0, 1, 0}));
    REQUIRE(make_child_idx<3>(7) == (std::array<size_t,3>{1, 1, 1}));
}

TEST_CASE("make child idx 2d") {
    REQUIRE(make_child_idx<2>(1) == (std::array<size_t,2>{0, 1}));
    REQUIRE(make_child_idx<2>(3) == (std::array<size_t,2>{1, 1}));
}

TEST_CASE("get subcell") 
{
    Ball<3> b{{0, 1, 0}, 2 * std::sqrt(3.0)};
    auto child = get_subbox(b, {1,0,1});
    REQUIRE_ARRAY_CLOSE(child.center, (std::array<double,3>{1,0,1}), 3, 1e-14);
    REQUIRE_CLOSE(child.R, std::sqrt(3.0), 1e-14);
}

TEST_CASE("bounding ball contains its balls")
{
    for (size_t i = 0; i < 100; i++) {
        auto centers = random_pts<2>(10, -1, 1); 
        auto Rs = random_pts<1>(10, 0.0, 0.1);
        auto balls_idxs = combine_balls_idxs(
            centers.data(),
            reinterpret_cast<double*>(Rs.data()),
            centers.size()
        );
        auto b = tree_bounds(balls_idxs.data(), centers.size());
        auto b_shrunk = b;
        b_shrunk.R /= 1.01;
        bool all_balls_in_shrunk = true;
        for (size_t j = 0; j < centers.size(); j++) {
            REQUIRE(ball_in_ball(b, balls_idxs[j].ball)); // Check that the bounding ball is sufficient.
            all_balls_in_shrunk = all_balls_in_shrunk && ball_in_ball(b_shrunk, balls_idxs[j].ball);
        }
        REQUIRE(!all_balls_in_shrunk); // Check that the bounding ball is minimal.
    }
}

TEST_CASE("octree partition") {
    size_t n_balls = 100;
    auto balls = random_pts<3>(n_balls, -1, 1);    
    auto Rs = random_pts<1>(n_balls, 0.0, 0.1);
    auto balls_idxs = combine_balls_idxs(
        balls.data(),
        reinterpret_cast<double*>(Rs.data()),
        balls.size()
    );
    auto bounds = tree_bounds(balls_idxs.data(), balls.size());
    auto splits = octree_partition(bounds, balls_idxs.data(), balls_idxs.data() + n_balls);
    for (int i = 0; i < 8; i++) {
        for (int j = splits[i]; j < splits[i + 1]; j++) {
            REQUIRE(find_containing_subcell(bounds, balls_idxs[j].ball.center) == i);
        }
    }
}

TEST_CASE("one level octree") 
{
    auto es = random_pts<3>(3);
    auto r = random_pts<1>(3);
    auto oct = Octree<3>::build_fnc(
        es.data(),
        reinterpret_cast<double*>(r.data()), 
        es.size(), 4
    );
    REQUIRE(oct.max_height == 0);
    REQUIRE(oct.nodes.size() == 1);
    REQUIRE(oct.root().is_leaf);
    REQUIRE(oct.root().end - oct.root().start);
    REQUIRE(oct.root().depth == 0);
    REQUIRE(oct.root().height == 0);
}

TEST_CASE("many level octree") 
{
    auto centers = random_pts<3>(1000);
    auto Rs = random_pts<1>(1000, 0.0, 0.1);
    auto oct = Octree<3>::build_fnc(
        centers.data(),
        reinterpret_cast<double*>(Rs.data()),
        centers.size(), 999
    ); 
    REQUIRE(oct.orig_idxs.size() == 1000);
    REQUIRE(oct.nodes[oct.root().children[0]].depth == 1);
}
