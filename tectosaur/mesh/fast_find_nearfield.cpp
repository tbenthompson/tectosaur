<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['sources'].extend(['../fmm/octree.cpp'])
cfg['dependencies'].extend([
    '../fmm/octree.hpp',
    '../include/pybind11_nparray.hpp',
    '../fmm/tree_helpers.hpp'
])
%>

#include <atomic>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/pybind11_nparray.hpp"
#include "../include/timing.hpp"
#include "../fmm/octree.hpp"

namespace py = pybind11;

template <size_t dim>
void query_helper(std::vector<long>& out,
    const OctreeNode<dim>& obs_node, const Octree<dim>& obs_tree,
    const std::vector<double>& obs_expanded_r, double* obs_radius_ptr,
    const OctreeNode<dim>& src_node, const Octree<dim>& src_tree,
    const std::vector<double>& src_expanded_r, double* src_radius_ptr,
    double threshold) 
{
    double r1 = obs_expanded_r[obs_node.idx];
    double r2 = src_expanded_r[src_node.idx];
    double limit = std::pow((r1 + r2) * threshold, 2);
    if (dist2(obs_node.bounds.center, src_node.bounds.center) > limit) {
        return;
    }
    if (obs_node.is_leaf && src_node.is_leaf) {
        for (size_t i = obs_node.start; i < obs_node.end; i++) {
            auto orig_idx_i = obs_tree.orig_idxs[i];
            for (size_t j = src_node.start; j < src_node.end; j++) {
                auto orig_idx_j = src_tree.orig_idxs[j];
                auto pt_touching_sep = obs_radius_ptr[i] + src_radius_ptr[j];
                auto pt_limit = std::pow(pt_touching_sep * threshold, 2);
                if (dist2(obs_tree.balls[i].center, src_tree.balls[j].center) > pt_limit) {
                    continue;
                }
                out.push_back(orig_idx_i);
                out.push_back(orig_idx_j);
            }
        }
        return;
    }
    bool split2 = ((r1 < r2) && !src_node.is_leaf) || obs_node.is_leaf;
    if (split2) {
        for (size_t i = 0; i < OctreeNode<dim>::split; i++) {
            query_helper(
                out, 
                obs_node, obs_tree, obs_expanded_r, obs_radius_ptr,
                src_tree.nodes[src_node.children[i]], src_tree,
                src_expanded_r, src_radius_ptr,
                threshold
            ); 
        }
    } else {
        for (size_t i = 0; i < OctreeNode<dim>::split; i++) {
            query_helper(
                out,
                obs_tree.nodes[obs_node.children[i]], obs_tree,
                obs_expanded_r, obs_radius_ptr,
                src_node, src_tree, src_expanded_r, src_radius_ptr,
                threshold
            ); 
        }
    }
}

//TODO: This is redundant now that octree/kdtree support ball radius
template <size_t dim>
std::vector<double> get_expanded_node_r(const Octree<dim>& tree, double* radius_ptr) {
    std::vector<double> expanded_node_r(tree.nodes.size());
#pragma omp parallel for
    for (size_t i = 0; i < tree.nodes.size(); i++) {
        auto& n = tree.nodes[i];
        double max_radius = n.bounds.R;
        for (size_t j = n.start; j < n.end; j++) {
            auto orig_idx = tree.orig_idxs[j];
            auto modified_dist = dist(tree.balls[j].center, n.bounds.center) + radius_ptr[orig_idx];
            if (modified_dist > max_radius) {
                max_radius = modified_dist;
            }
        }
        expanded_node_r[i] = max_radius;
    }
    return expanded_node_r;
}

template <size_t dim>
std::vector<long> query_ball_points(
    const Octree<dim>& obs_tree, const std::vector<double>& obs_expanded_r,
    std::array<double,dim>* obs_pt_ptr, double* obs_radius_ptr, size_t n_obs,
    const Octree<dim>& src_tree, const std::vector<double>& src_expanded_r,
    std::array<double,dim>* src_pt_ptr, double* src_radius_ptr, size_t n_src,
    double threshold) 
{

    std::vector<long> out;
    constexpr int parallelize_depth = 2;
    std::atomic<int> n_pairs{0};
#pragma omp parallel
    {

        std::vector<long> out_private;
#pragma omp for
        for (size_t i = 0; i < obs_tree.nodes.size(); i++) {
            auto& obs_n = obs_tree.nodes[i];
            if (obs_n.depth > parallelize_depth) {
                continue;
            }
            if (obs_n.depth != parallelize_depth && !obs_n.is_leaf) {
                continue;
            }
            query_helper(
                out_private, 
                obs_n, obs_tree, obs_expanded_r, obs_radius_ptr,
                src_tree.root(), src_tree, src_expanded_r, src_radius_ptr,
                threshold
            );
        }
        size_t insertion_start_idx = n_pairs.fetch_add(out_private.size());
#pragma omp barrier
#pragma omp single
        {
            out.resize(n_pairs);
        }

        for (size_t i = 0; i < out_private.size(); i++) {
            out[insertion_start_idx + i] = out_private[i];
        }
    }

    return out;
}

std::array<std::vector<long>,3> split_adjacent_close(long* close_pairs,
    size_t n_pairs, long* tris_A, long* tris_B)
{
    std::array<std::vector<long>,3> out;
    for (size_t i = 0; i < n_pairs; i++) {
        auto idx1 = close_pairs[i * 2];
        auto idx2 = close_pairs[i * 2 + 1];
        std::pair<long,long> pair1 = {-1,-1};
        std::pair<long,long> pair2 = {-1,-1};
        bool coincident = false;
        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                if (tris_A[idx1 * 3 + d1] != tris_B[idx2 * 3 + d2]) {
                    continue;
                }
                if (pair1.first == -1) {
                    pair1 = {d1, d2};
                } else if (pair2.first == -1) {
                    pair2 = {d1, d2};
                } else {
                    coincident = true;
                }
            }
        }
        if (coincident) {
            continue;
        }
        if (pair1.first == -1) {
            out[0].insert(out[0].end(), {idx1, idx2});
        } else if (pair2.first == -1) {
            out[1].insert(out[1].end(), {idx1, idx2, pair1.first, pair1.second});
        } else {
            out[2].insert(
                out[2].end(), 
                {idx1, idx2, pair1.first, pair1.second, pair2.first, pair2.second}
            );
        }
    }
    return out;
}

PYBIND11_MODULE(fast_find_nearfield,m) {
    constexpr static int dim = 3;

    m.def("get_nearfield",
        [] (NPArrayD obs_pts, NPArrayD obs_radius,
            NPArrayD src_pts, NPArrayD src_radius,
            double threshold, int leaf_size) 
        {
            Timer t{true};
            auto obs_pts_ptr = as_ptr<std::array<double,dim>>(obs_pts);
            auto obs_radius_ptr = as_ptr<double>(obs_radius);
            auto n_obs = obs_pts.request().shape[0];
            auto obs_tree = Octree<dim>::build_fnc(obs_pts_ptr, obs_radius_ptr, n_obs, leaf_size);
            auto obs_expanded_r = get_expanded_node_r(obs_tree, obs_radius_ptr);

            auto src_pts_ptr = as_ptr<std::array<double,dim>>(src_pts);
            auto src_radius_ptr = as_ptr<double>(src_radius);
            auto n_src = src_pts.request().shape[0];
            auto src_tree = Octree<dim>::build_fnc(src_pts_ptr, src_radius_ptr, n_src, leaf_size);
            auto src_expanded_r = get_expanded_node_r(src_tree, src_radius_ptr);
            t.report("setup");

            auto out_vec = query_ball_points(
                obs_tree, obs_expanded_r, obs_pts_ptr, obs_radius_ptr, n_obs,
                src_tree, src_expanded_r, src_pts_ptr, src_radius_ptr, n_src,
                threshold
            );
            t.report("query");

            auto out_arr = array_from_vector(out_vec, {out_vec.size() / 2, 2});
            t.report("make out");

            return out_arr;
        });

    m.def("self_get_nearfield",
        [] (NPArrayD pts, NPArrayD radius, double threshold, int leaf_size) {
            Timer t{true};
            auto pts_ptr = as_ptr<std::array<double,dim>>(pts);
            auto radius_ptr = as_ptr<double>(radius);
            auto n_obs = pts.request().shape[0];
            auto tree = Octree<dim>::build_fnc(pts_ptr, radius_ptr, n_obs, leaf_size);
            auto expanded_r = get_expanded_node_r(tree, radius_ptr);
            t.report("setup");

            auto out_vec = query_ball_points(
                tree, expanded_r, pts_ptr, radius_ptr, n_obs,
                tree, expanded_r, pts_ptr, radius_ptr, n_obs,
                threshold
            );
            t.report("query");

            auto out_arr = array_from_vector(out_vec, {out_vec.size() / 2, 2});
            t.report("make out");

            return out_arr;
        });
    
    m.def("split_adjacent_close",
        [] (NPArray<long> close_pairs, NPArray<long> trisA, NPArray<long> trisB) {
            auto close_pairs_ptr = as_ptr<long>(close_pairs);
            auto trisA_ptr = as_ptr<long>(trisA);
            auto trisB_ptr = as_ptr<long>(trisB);
            auto n_pairs = close_pairs.request().shape[0];

            auto out = split_adjacent_close(close_pairs_ptr, n_pairs, trisA_ptr, trisB_ptr);
            return py::make_tuple(
                array_from_vector(out[0], {out[0].size() / 2, 2}),
                array_from_vector(out[1], {out[1].size() / 4, 4}),
                array_from_vector(out[2], {out[2].size() / 6, 6})
            );
        });
}
