#include "octree.hpp"

template <size_t dim>
std::array<int,OctreeNode<dim>::split+1> octree_partition(
        const Ball<dim>& bounds, BallWithIdx<dim>* start, BallWithIdx<dim>* end) 
{
    std::array<std::vector<BallWithIdx<dim>>,OctreeNode<dim>::split> chunks{};
    for (auto* entry = start; entry < end; entry++) {
        chunks[find_containing_subcell(bounds, entry->ball.center)].push_back(*entry);
    }

    auto* next = start;
    std::array<int,OctreeNode<dim>::split+1> splits{};
    for (size_t subcell_idx = 0; subcell_idx < OctreeNode<dim>::split; subcell_idx++) {
        size_t subcell_n_balls = chunks[subcell_idx].size();
        for (size_t i = 0; i < subcell_n_balls; i++) {
            *next = chunks[subcell_idx][i];
            next++;
        }
        splits[subcell_idx + 1] = splits[subcell_idx] + subcell_n_balls;
    }

    return splits;
}


template <size_t dim>
Octree<dim> build_octree(std::array<double,dim>* in_balls, double* in_R,
    size_t n_balls, size_t n_per_cell) 
{
    auto balls_idxs = combine_balls_idxs(in_balls, in_R, n_balls);

    auto bounds = root_tree_bounds(balls_idxs.data(), n_balls);
    bounds.R *= std::sqrt(dim);

    Octree<dim> out;
    add_node(out, 0, n_balls, n_per_cell, 0, bounds, balls_idxs);

    out.max_height = out.nodes[0].height;

    out.balls.resize(n_balls);
    out.orig_idxs.resize(n_balls);
    for (size_t i = 0; i < n_balls; i++) {
        out.balls[i] = balls_idxs[i].ball;
        out.orig_idxs[i] = balls_idxs[i].orig_idx;
    }

    return out;
}

template <size_t dim>
size_t add_node(Octree<dim>& tree, size_t start, size_t end, 
    size_t n_per_cell, int depth, Ball<dim> bounds,
    std::vector<BallWithIdx<dim>>& temp_balls)
{
    bool is_leaf = end - start <= n_per_cell; 
    auto n_idx = tree.nodes.size();
    tree.nodes.push_back({start, end, bounds, is_leaf, 0, depth, n_idx, {}});
    if (!is_leaf) {
        auto splits = octree_partition(bounds, temp_balls.data() + start, temp_balls.data() + end);
        int max_child_height = 0;
        for (size_t octant = 0; octant < OctreeNode<dim>::split; octant++) {
            auto child_bounds = get_subbox(bounds, make_child_idx<dim>(octant));
            auto child_start = start + splits[octant];
            auto child_end = start + splits[octant + 1];
            auto child_node_idx = add_node(
                tree, child_start, child_end,
                n_per_cell, depth + 1, child_bounds, temp_balls
            );
            tree.nodes[n_idx].children[octant] = child_node_idx;
            max_child_height = std::max(max_child_height, tree.nodes[child_node_idx].height);
        }
        tree.nodes[n_idx].height = max_child_height + 1;
    }
    return n_idx;
}

template struct Octree<2>;
template struct Octree<3>;

template Octree<2> build_octree(std::array<double,2>* in_balls, double* in_R,
        size_t n_balls, size_t n_per_cell);
template Octree<3> build_octree(std::array<double,3>* in_balls, double* in_R,
        size_t n_balls, size_t n_per_cell);
