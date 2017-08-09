#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

#include "include/timing.hpp"
#include "fmm_impl.hpp"

struct InteractionLists {
    std::vector<std::vector<int>> p2m;
    std::vector<std::vector<std::vector<int>>> m2m;
    std::vector<std::vector<std::vector<int>>> u2e;

    std::vector<std::vector<int>> l2p;
    std::vector<std::vector<std::vector<int>>> l2l;
    std::vector<std::vector<std::vector<int>>> d2e;

    std::vector<std::vector<int>> p2p; 
    std::vector<std::vector<int>> m2p; 
    std::vector<std::vector<int>> p2l; 
    std::vector<std::vector<int>> m2l; 
};

CompressedInteractionList compress(const std::vector<std::vector<int>>& list) {
    CompressedInteractionList out;

    int nonempty = 0;
    for (size_t i = 0; i < list.size(); i++) {
        if (list[i].size() > 0) {
            nonempty++;
        }
    }

    out.obs_n_idxs.resize(nonempty);
    out.obs_src_starts.resize(nonempty + 1);
    out.obs_src_starts[0] = 0;
    size_t next = 0;
    for (size_t i = 0; i < list.size(); i++) {
        if (list[i].size() == 0) {
            continue;
        }
        out.obs_n_idxs[next] = i;
        out.obs_src_starts[next + 1] = out.obs_src_starts[next] + list[i].size();
        next++;
    }

    out.src_n_idxs.resize(out.obs_src_starts.back());
    next = 0;
    for (size_t i = 0; i < list.size(); i++) {
        for (size_t j = 0; j < list[i].size(); j++) {
            out.src_n_idxs[next] = list[i][j];
            next++;
        }
    }

    return out;
}

template <typename TreeT>
InteractionLists init_interaction_lists(const FMMMat<TreeT>& mat) {
    InteractionLists lists;

    lists.m2m.resize(mat.src_tree.max_height + 1);
    lists.u2e.resize(mat.src_tree.max_height + 1);
    for (int i = 0; i < mat.src_tree.max_height + 1; i++) {
        lists.m2m[i].resize(mat.src_tree.nodes.size());
        lists.u2e[i].resize(mat.src_tree.nodes.size());
    }
    lists.p2m.resize(mat.src_tree.nodes.size());

    lists.l2l.resize(mat.obs_tree.max_height + 1);
    lists.d2e.resize(mat.obs_tree.max_height + 1);
    for (int i = 0; i < mat.obs_tree.max_height + 1; i++) {
        lists.l2l[i].resize(mat.obs_tree.nodes.size());
        lists.d2e[i].resize(mat.obs_tree.nodes.size());
    }
    lists.l2p.resize(mat.obs_tree.nodes.size());

    lists.p2p.resize(mat.obs_tree.nodes.size());
    lists.m2p.resize(mat.obs_tree.nodes.size());
    lists.p2l.resize(mat.obs_tree.nodes.size());
    lists.m2l.resize(mat.obs_tree.nodes.size());

    return lists;
}

template <typename TreeT>
void compress_interaction_lists(FMMMat<TreeT>& mat, const InteractionLists& lists) {
    mat.m2m.resize(lists.m2m.size());
    mat.u2e.resize(lists.m2m.size());
    for (size_t i = 0; i < lists.m2m.size(); i++) {
        mat.m2m[i] = compress(lists.m2m[i]);
        mat.u2e[i] = compress(lists.u2e[i]);
    }
    mat.p2m = compress(lists.p2m);

    mat.l2l.resize(lists.l2l.size());
    mat.d2e.resize(lists.l2l.size());
    for (size_t i = 0; i < lists.l2l.size(); i++) {
        mat.l2l[i] = compress(lists.l2l[i]);
        mat.d2e[i] = compress(lists.d2e[i]);
    }
    mat.l2p = compress(lists.l2p);

    mat.p2p = compress(lists.p2p);
    mat.m2p = compress(lists.m2p);
    mat.p2l = compress(lists.p2l);
    mat.m2l = compress(lists.m2l);
}

template <typename TreeT>
void traverse(FMMMat<TreeT>& mat,
        InteractionLists& interaction_lists,
        const typename TreeT::Node& obs_n,
        const typename TreeT::Node& src_n) 
{
    auto r_src = src_n.bounds.R;
    auto r_obs = obs_n.bounds.R;
    auto sep = hypot(sub(obs_n.bounds.center, src_n.bounds.center));

    // If outer_r * r_src + inner_r * r_obs is less than the separation, then
    // the relevant check surfaces for the two interacting cells don't
    // intersect.
    // That means it should be safe to perform approximate interactions. I add
    // a small safety factor just in case!
    double safety_factor = 0.98;
    if (mat.cfg.outer_r * r_src + mat.cfg.inner_r * r_obs < safety_factor * sep) {
        // If there aren't enough src or obs to justify using the approximation,
        // then just do a p2p direct calculation between the nodes.
        // TODO: There could be a minor gain in doing better distance checks when small_src or small_obs are true. The check surfaces are no longer relevant, just the bounds of the tree node.
        bool small_src = src_n.end - src_n.start < mat.cfg.order;
        bool small_obs = obs_n.end - obs_n.start < mat.cfg.order;

        if (small_src && small_obs) {
            mat.obs_tree.for_all_leaves_of(obs_n,
                [&] (const typename TreeT::Node& leaf_obs_n) {
                    interaction_lists.p2p[leaf_obs_n.idx].push_back(src_n.idx);
                }
            );
        } else if (small_obs) {
            mat.obs_tree.for_all_leaves_of(obs_n,
                [&] (const typename TreeT::Node& leaf_obs_n) {
                    interaction_lists.m2p[leaf_obs_n.idx].push_back(src_n.idx);
                }
            );
        } else if (small_src) {
            interaction_lists.p2l[obs_n.idx].push_back(src_n.idx);
        } else {
            interaction_lists.m2l[obs_n.idx].push_back(src_n.idx);
        }

        return;
    }

    if (src_n.is_leaf && obs_n.is_leaf) {
        interaction_lists.p2p[obs_n.idx].push_back(src_n.idx);
        return;
    }

    bool split_src = ((r_obs < r_src) && !src_n.is_leaf) || obs_n.is_leaf;
    if (split_src) {
        for (size_t i = 0; i < TreeT::split; i++) {
            traverse(mat, interaction_lists, obs_n, mat.src_tree.nodes[src_n.children[i]]);
        }
    } else {
        for (size_t i = 0; i < TreeT::split; i++) {
            traverse(mat, interaction_lists, mat.obs_tree.nodes[obs_n.children[i]], src_n);
        }
    }
}

template <typename TreeT>
void up_collect(FMMMat<TreeT>& mat, InteractionLists& interaction_lists, 
        const typename TreeT::Node& src_n) 
{
    auto level = mat.src_tree.max_height - src_n.depth;
    interaction_lists.u2e[level][src_n.idx].push_back(src_n.idx);
    if (src_n.is_leaf) {
        interaction_lists.p2m[src_n.idx].push_back(src_n.idx);
    } else {
        for (size_t i = 0; i < TreeT::split; i++) {
            auto child_n = mat.src_tree.nodes[src_n.children[i]];
            up_collect(mat, interaction_lists, child_n);
            interaction_lists.m2m[level][src_n.idx].push_back(child_n.idx);
        }
    }
}

template <typename TreeT>
void down_collect(FMMMat<TreeT>& mat, InteractionLists& interaction_lists, 
        const typename TreeT::Node& obs_n) 
{
    interaction_lists.d2e[obs_n.depth][obs_n.idx].push_back(obs_n.idx);
    if (obs_n.is_leaf) {
        interaction_lists.l2p[obs_n.idx].push_back(obs_n.idx);
    } else {
        for (size_t i = 0; i < TreeT::split; i++) {
            auto child_n = mat.obs_tree.nodes[obs_n.children[i]];
            down_collect(mat, interaction_lists, child_n);
            interaction_lists.l2l[child_n.depth][child_n.idx].push_back(obs_n.idx);
        }
    }
}

template <typename TreeT>
FMMMat<TreeT>::FMMMat(TreeT obs_tree, TreeT src_tree, FMMConfig<TreeT::dim> cfg):
    obs_tree(obs_tree),
    src_tree(src_tree),
    cfg(cfg)
{}

template <typename TreeT>
FMMMat<TreeT> fmmmmmmm(const TreeT& obs_tree, const TreeT& src_tree,
    const FMMConfig<TreeT::dim>& cfg)
{

    //TODO: Creating the translation surface takes a trivial amount of time
    //and can be moved to python.
    FMMMat<TreeT> mat(obs_tree, src_tree, cfg);

    mat.u2e.resize(mat.src_tree.max_height + 1);
    mat.d2e.resize(mat.obs_tree.max_height + 1);

    auto interaction_lists = init_interaction_lists(mat);

    up_collect(mat, interaction_lists, mat.src_tree.root());
    down_collect(mat, interaction_lists, mat.obs_tree.root());
    traverse(mat, interaction_lists, mat.obs_tree.root(), mat.src_tree.root());
    compress_interaction_lists(mat, interaction_lists);


    return mat;
}

template 
FMMMat<Octree<2>> fmmmmmmm(const Octree<2>& obs_tree,
    const Octree<2>& src_tree,
    const FMMConfig<2>& cfg);
template 
FMMMat<Octree<3>> fmmmmmmm(const Octree<3>& obs_tree,
    const Octree<3>& src_tree,
    const FMMConfig<3>& cfg);
template struct FMMMat<Octree<2>>;
template struct FMMMat<Octree<3>>;
