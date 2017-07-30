#include "octree.hpp"

template <size_t dim>
std::array<int,OctreeNode<dim>::split+1> octree_partition(
        const Cube<dim>& bounds, PtWithIdx<dim>* start, PtWithIdx<dim>* end) 
{
    std::array<std::vector<PtWithIdx<dim>>,OctreeNode<dim>::split> chunks{};
    for (auto* entry = start; entry < end; entry++) {
        chunks[find_containing_subcell(bounds, entry->pt)].push_back(*entry);
    }

    auto* next = start;
    std::array<int,OctreeNode<dim>::split+1> splits{};
    for (size_t subcell_idx = 0; subcell_idx < OctreeNode<dim>::split; subcell_idx++) {
        size_t subcell_n_pts = chunks[subcell_idx].size();
        for (size_t i = 0; i < subcell_n_pts; i++) {
            *next = chunks[subcell_idx][i];
            next++;
        }
        splits[subcell_idx + 1] = splits[subcell_idx] + subcell_n_pts;
    }

    return splits;
}

template <size_t dim>
Cube<dim> bounding_box(PtWithIdx<dim>* pts, size_t n_pts) {
    std::array<double,dim> center_of_mass{};
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            center_of_mass[d] += pts[i].pt[d];
        }
    }
    for (size_t d = 0; d < dim; d++) {
        center_of_mass[d] /= n_pts;
    }

    double max_width = 0.0;
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            max_width = std::max(max_width, fabs(pts[i].pt[d] - center_of_mass[d]));
        }
    }

    return {center_of_mass, max_width};
}


template <size_t dim>
Octree<dim>::Octree(std::array<double,dim>* in_pts, size_t n_pts, size_t n_per_cell):
    pts(n_pts),
    orig_idxs(n_pts),
    n_pts(n_pts)
{
    auto pts_idxs = combine_pts_idxs(in_pts, n_pts);

    auto bounds = bounding_box(pts_idxs.data(), n_pts);
    add_node(0, n_pts, n_per_cell, 0, bounds, pts_idxs);

    max_height = nodes[0].height;

    for (size_t i = 0; i < n_pts; i++) {
        pts[i] = pts_idxs[i].pt;
        orig_idxs[i] = pts_idxs[i].orig_idx;
    }
}

template <size_t dim>
size_t Octree<dim>::add_node(size_t start, size_t end, 
    size_t n_per_cell, int depth, Cube<dim> bounds,
    std::vector<PtWithIdx<dim>>& temp_pts)
{
    bool is_leaf = end - start <= n_per_cell; 
    auto n_idx = nodes.size();
    nodes.push_back({start, end, bounds, is_leaf, 0, depth, n_idx, {}});
    if (!is_leaf) {
        auto splits = octree_partition(bounds, temp_pts.data() + start, temp_pts.data() + end);
        int max_child_height = 0;
        for (size_t octant = 0; octant < OctreeNode<dim>::split; octant++) {
            auto child_bounds = get_subcell(bounds, make_child_idx<dim>(octant));
            auto child_start = start + splits[octant];
            auto child_end = start + splits[octant + 1];
            // TODO: Should we use adaptive bounds?
            // auto child_n_pts = child_end - child_start;
            // auto child_bounds = bounding_box(&temp_pts[child_start], child_n_pts);
            auto child_node_idx = add_node(
                child_start, child_end,
                n_per_cell, depth + 1, child_bounds, temp_pts
            );
            nodes[n_idx].children[octant] = child_node_idx;
            max_child_height = std::max(max_child_height, nodes[child_node_idx].height);
        }
        nodes[n_idx].height = max_child_height + 1;
    }
    return n_idx;
}

template Cube<2> bounding_box(PtWithIdx<2>* pts, size_t n_pts);
template Cube<3> bounding_box(PtWithIdx<3>* pts, size_t n_pts);

template struct Octree<2>;
template struct Octree<3>;
