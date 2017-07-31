/*
<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['sources'].extend(['../fmm/octree.cpp'])
cfg['dependencies'].extend(['../fmm/octree.hpp', '../include/pybind11_nparray.hpp'])
%>
*/

#include <atomic>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/pybind11_nparray.hpp"
#include "../include/timing.hpp"
#include "../fmm/octree.hpp"

namespace py = pybind11;

template <size_t dim>
void query_helper(std::vector<long>& out,
    const Octree<dim>& tree, const std::vector<double>& expanded_node_r,
    const OctreeNode<dim>& node1, const OctreeNode<dim>& node2,
    double* radius_ptr, double threshold) 
{
    double r1 = expanded_node_r[node1.idx];
    double r2 = expanded_node_r[node2.idx];
    double limit = std::pow((r1 + r2) * threshold, 2);
    if (dist2(node1.bounds.center, node2.bounds.center) > limit) {
        return;
    }
    if (node1.is_leaf && node2.is_leaf) {
        for (size_t i = node1.start; i < node1.end; i++) {
            auto orig_idx_i = tree.orig_idxs[i];
            for (size_t j = node2.start; j < node2.end; j++) {
                if (i == j) {
                    continue;
                }
                auto orig_idx_j = tree.orig_idxs[j];
                auto pt_touching_sep = radius_ptr[i] + radius_ptr[j];
                auto pt_limit = std::pow(pt_touching_sep * threshold, 2);
                if (dist2(tree.pts[i], tree.pts[j]) > pt_limit) {
                    continue;
                }
                out.push_back(orig_idx_i);
                out.push_back(orig_idx_j);
            }
        }
        return;
    }
    bool split2 = ((r1 < r2) && !node2.is_leaf) || node1.is_leaf;
    if (split2) {
        for (size_t i = 0; i < OctreeNode<dim>::split; i++) {
            query_helper(
                out, tree, expanded_node_r, 
                node1, tree.nodes[node2.children[i]],
                radius_ptr, threshold
            ); 
        }
    } else {
        for (size_t i = 0; i < OctreeNode<dim>::split; i++) {
            query_helper(
                out, tree, expanded_node_r, 
                tree.nodes[node1.children[i]], node2,
                radius_ptr, threshold
            ); 
        }
    }
}

template <size_t dim>
std::vector<double> get_expanded_node_r(const Octree<dim>& tree, double* radius_ptr) {
    std::vector<double> expanded_node_r(tree.nodes.size());
#pragma omp parallel for
    for (size_t i = 0; i < tree.nodes.size(); i++) {
        auto& n = tree.nodes[i];
        double max_radius = n.bounds.R();
        for (size_t j = n.start; j < n.end; j++) {
            auto orig_idx = tree.orig_idxs[j];
            auto modified_dist = dist(tree.pts[j], n.bounds.center) + radius_ptr[orig_idx];
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
    const Octree<dim>& tree, const std::vector<double>& expanded_node_r,
    std::array<double,dim>* pt_ptr, double* radius_ptr,
    size_t n_entities, double threshold) 
{

    std::vector<long> out;
    constexpr int parallelize_depth = 2;
    std::atomic<int> n_pairs{0};
#pragma omp parallel
    {

        std::vector<long> out_private;
#pragma omp for
        for (size_t i = 0; i < tree.nodes.size(); i++) {
            auto& n = tree.nodes[i];
            if (n.depth > parallelize_depth) {
                continue;
            }
            if (n.depth != parallelize_depth && !n.is_leaf) {
                continue;
            }
            query_helper(
                out_private, tree, expanded_node_r,
                n, tree.root(), radius_ptr, threshold
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
    size_t n_pairs, long* tris)
{
    std::array<std::vector<long>,3> out;
    for (size_t i = 0; i < n_pairs; i++) {
        auto idx1 = close_pairs[i * 2];
        auto idx2 = close_pairs[i * 2 + 1];
        std::pair<long,long> pair1 = {-1,-1};
        std::pair<long,long> pair2 = {-1,-1};
        for (int d1 = 0; d1 < 3; d1++) {
            for (int d2 = 0; d2 < 3; d2++) {
                if (tris[idx1 * 3 + d1] != tris[idx2 * 3 + d2]) {
                    continue;
                }
                if (pair1.first == -1) {
                    pair1 = {d1, d2};
                } else {
                    pair2 = {d1, d2};
                }
            }
        }
        if (pair1.first == -1) {
            out[0].insert(out[0].end(), {idx1, idx2});
        } else if (pair2.first == -1) {
            out[1].insert(out[1].end(), {idx1, idx2, pair1.first, pair1.second});
        } else {
            out[2].insert(out[2].end(), {
                idx1, idx2, pair1.first, pair1.second, pair2.first, pair2.second
            });
        }
    }
    return out;
}

PYBIND11_PLUGIN(fast_find_nearfield) {
    py::module m("fast_find_nearfield", "");
    constexpr static int dim = 3;

    m.def("get_nearfield",
        [] (NPArrayD pt, NPArrayD radius, double threshold, int leaf_size) {
            auto pt_ptr = as_ptr<std::array<double,dim>>(pt);
            auto radius_ptr = as_ptr<double>(radius);
            auto n_entities = pt.request().shape[0];
            Timer t(true);
            Octree<dim> tree(pt_ptr, n_entities, leaf_size);
            t.report("build tree");
            auto expanded_r = get_expanded_node_r(tree, radius_ptr);
            t.report("expanded R");
            auto out_vec = query_ball_points(
                tree, expanded_r, pt_ptr, radius_ptr, n_entities, threshold
            );
            t.report("query");
            auto out_arr = array_from_vector(out_vec, {out_vec.size() / 2, 2});
            t.report("to array");
            return out_arr;
        });
    
    m.def("split_adjacent_close",
        [] (NPArray<long> close_pairs, NPArray<long> tris) {
            auto close_pairs_ptr = as_ptr<long>(close_pairs);
            auto tris_ptr = as_ptr<long>(tris);
            auto n_pairs = close_pairs.request().shape[0];

            auto out = split_adjacent_close(close_pairs_ptr, n_pairs, tris_ptr);
            return py::make_tuple(
                array_from_vector(out[0], {out[0].size() / 2, 2}),
                array_from_vector(out[1], {out[1].size() / 4, 4}),
                array_from_vector(out[2], {out[2].size() / 6, 6})
            );
        });

    return m.ptr();
}
