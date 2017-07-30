#include "doctest.h"
#include "test_helpers.hpp"
#include "octree.hpp"

#include <iostream>

TEST_CASE("containing subcell box 2d") {
    Cube<2> b{{0, 0}, 1.0};
    REQUIRE(find_containing_subcell(b, {0.1, 0.1}) == 3);
    REQUIRE(find_containing_subcell(b, {0.1, -0.1}) == 2);
    REQUIRE(find_containing_subcell(b, {-0.1, -0.1}) == 0);
}

TEST_CASE("containing subcell box 3d") {
    Cube<3> b{{0, 0, 0}, 1.0};
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
    Cube<3> b{{0, 1, 0}, 2};
    auto child = get_subcell(b, {1,0,1});
    REQUIRE_ARRAY_CLOSE(child.center, (std::array<double,3>{1,0,1}), 3, 1e-14);
    REQUIRE_CLOSE(child.width, 1.0, 1e-14);
}

TEST_CASE("box R") {
    Cube<2> b2{{0,0}, 1.0};
    REQUIRE(b2.R() == std::sqrt(2.0));
    Cube<3> b3{{0,0,0}, 1.0};
    REQUIRE(b3.R() == std::sqrt(3.0));
}

TEST_CASE("bounding box contains its pts")
{
    for (size_t i = 0; i < 10; i++) {
        auto pts = random_pts<2>(10, -1, 1); 
        auto pts_idxs = combine_pts_idxs(pts.data(), pts.size());
        auto b = bounding_box(pts_idxs.data(), pts.size());
        auto b_shrunk = b;
        b_shrunk.width /= 1 + 1e-10;
        bool all_pts_in_shrunk = true;
        for (auto p: pts) {
            REQUIRE(in_box(b, p)); // Check that the bounding box is sufficient.
            all_pts_in_shrunk = all_pts_in_shrunk && in_box(b_shrunk, p);
        }
        REQUIRE(!all_pts_in_shrunk); // Check that the bounding box is minimal.
    }
}

TEST_CASE("octree partition") {
    size_t n_pts = 100;
    auto pts = random_pts<3>(n_pts, -1, 1);    
    auto pts_idxs = combine_pts_idxs(pts.data(), n_pts);
    auto bounds = bounding_box(pts_idxs.data(), pts.size());
    auto splits = octree_partition(bounds, pts_idxs.data(), pts_idxs.data() + n_pts);
    for (int i = 0; i < 8; i++) {
        for (int j = splits[i]; j < splits[i + 1]; j++) {
            REQUIRE(find_containing_subcell(bounds, pts_idxs[j].pt) == i);
        }
    }
}

TEST_CASE("one level octree") 
{
    auto es = random_pts<3>(3);
    Octree<3> oct(es.data(), es.size(), 4);
    REQUIRE(oct.max_height == 0);
    REQUIRE(oct.nodes.size() == 1);
    REQUIRE(oct.root().is_leaf);
    REQUIRE(oct.root().end - oct.root().start);
    REQUIRE(oct.root().depth == 0);
    REQUIRE(oct.root().height == 0);
}

TEST_CASE("many level octree") 
{
    auto pts = random_pts<3>(1000);
    Octree<3> oct(pts.data(), pts.size(), 999); 
    REQUIRE(oct.orig_idxs.size() == 1000);
    REQUIRE(oct.nodes[oct.root().children[0]].depth == 1);
}
