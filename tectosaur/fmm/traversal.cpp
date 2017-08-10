#include "traversal.hpp"
#include "octree.hpp"
#include "kdtree.hpp"

struct InteractionLists {
    std::vector<std::vector<size_t>> p2m;
    std::vector<std::vector<std::vector<size_t>>> m2m;
    std::vector<std::vector<std::vector<size_t>>> u2e;

    std::vector<std::vector<size_t>> l2p;
    std::vector<std::vector<std::vector<size_t>>> l2l;
    std::vector<std::vector<std::vector<size_t>>> d2e;

    std::vector<std::vector<size_t>> p2p; 
    std::vector<std::vector<size_t>> m2p; 
    std::vector<std::vector<size_t>> p2l; 
    std::vector<std::vector<size_t>> m2l; 
};

CompressedInteractionList compress(const std::vector<std::vector<size_t>>& list) {
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
InteractionLists init_interaction_lists(const TreeT& obs_tree, const TreeT& src_tree) {
    InteractionLists lists;

    lists.m2m.resize(src_tree.max_height + 1);
    lists.u2e.resize(src_tree.max_height + 1);
    for (int i = 0; i < src_tree.max_height + 1; i++) {
        lists.m2m[i].resize(src_tree.nodes.size());
        lists.u2e[i].resize(src_tree.nodes.size());
    }
    lists.l2l.resize(obs_tree.max_height + 1);
    lists.d2e.resize(obs_tree.max_height + 1);
    for (int i = 0; i < obs_tree.max_height + 1; i++) {
        lists.l2l[i].resize(obs_tree.nodes.size());
        lists.d2e[i].resize(obs_tree.nodes.size());
    }

    lists.p2m.resize(src_tree.nodes.size());
    lists.l2p.resize(obs_tree.nodes.size());
    lists.p2p.resize(obs_tree.nodes.size());
    lists.m2p.resize(obs_tree.nodes.size());
    lists.p2l.resize(obs_tree.nodes.size());
    lists.m2l.resize(obs_tree.nodes.size());

    return lists;
}

Interactions compress_interaction_lists(const InteractionLists& lists) {
    Interactions out;
    out.m2m.resize(lists.m2m.size());
    out.u2e.resize(lists.m2m.size());
    for (size_t i = 0; i < lists.m2m.size(); i++) {
        out.m2m[i] = compress(lists.m2m[i]);
        out.u2e[i] = compress(lists.u2e[i]);
    }

    out.l2l.resize(lists.l2l.size());
    out.d2e.resize(lists.l2l.size());
    for (size_t i = 0; i < lists.l2l.size(); i++) {
        out.l2l[i] = compress(lists.l2l[i]);
        out.d2e[i] = compress(lists.d2e[i]);
    }

    out.p2m = compress(lists.p2m);
    out.l2p = compress(lists.l2p);
    out.p2p = compress(lists.p2p);
    out.m2p = compress(lists.m2p);
    out.p2l = compress(lists.p2l);
    out.m2l = compress(lists.m2l);
    return out;
}


template <typename TreeT, typename F>
void for_all_leaves_of(const TreeT& t, const typename TreeT::Node& n, const F& f) {
    if (n.is_leaf) {
        f(n);
        return;
    }
    for (size_t i = 0; i < n.children.size(); i++) {
        for_all_leaves_of(t, t.nodes[n.children[i]], f);
    }
}

template <typename TreeT>
void traverse(const TreeT& obs_tree, const TreeT& src_tree,
        InteractionLists& interaction_lists,
        const typename TreeT::Node& obs_n, const typename TreeT::Node& src_n,
        double inner_r, double outer_r, size_t order) 
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

    if (outer_r * r_src + inner_r * r_obs < safety_factor * sep) {
        // If there aren't enough src or obs to justify using the approximation,
        // then just do a p2p direct calculation between the nodes.
        bool small_src = src_n.end - src_n.start < order;
        bool small_obs = obs_n.end - obs_n.start < order;

        if (small_src && small_obs) {
            for_all_leaves_of(obs_tree, obs_n,
                [&] (const typename TreeT::Node& leaf_obs_n) {
                    interaction_lists.p2p[leaf_obs_n.idx].push_back(src_n.idx);
                }
            );
        } else if (small_obs) {
            for_all_leaves_of(obs_tree, obs_n,
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
            traverse(
                obs_tree, src_tree, interaction_lists,
                obs_n, src_tree.nodes[src_n.children[i]],
                inner_r, outer_r, order
            );
        }
    } else {
        for (size_t i = 0; i < TreeT::split; i++) {
            traverse(
                obs_tree, src_tree, interaction_lists,
                obs_tree.nodes[obs_n.children[i]], src_n,
                inner_r, outer_r, order
            );
        }
    }
}

template <typename TreeT>
void up_collect(const TreeT& src_tree, InteractionLists& interaction_lists, 
        const typename TreeT::Node& src_n) 
{
    auto level = src_tree.max_height - src_n.depth;
    interaction_lists.u2e[level][src_n.idx].push_back(src_n.idx);
    if (src_n.is_leaf) {
        interaction_lists.p2m[src_n.idx].push_back(src_n.idx);
    } else {
        for (size_t i = 0; i < TreeT::split; i++) {
            auto child_n = src_tree.nodes[src_n.children[i]];
            up_collect(src_tree, interaction_lists, child_n);
            interaction_lists.m2m[level][src_n.idx].push_back(child_n.idx);
        }
    }
}

template <typename TreeT>
void down_collect(const TreeT& obs_tree, InteractionLists& interaction_lists, 
        const typename TreeT::Node& obs_n) 
{
    interaction_lists.d2e[obs_n.depth][obs_n.idx].push_back(obs_n.idx);
    if (obs_n.is_leaf) {
        interaction_lists.l2p[obs_n.idx].push_back(obs_n.idx);
    } else {
        for (size_t i = 0; i < TreeT::split; i++) {
            auto child_n = obs_tree.nodes[obs_n.children[i]];
            down_collect(obs_tree, interaction_lists, child_n);
            interaction_lists.l2l[child_n.depth][child_n.idx].push_back(obs_n.idx);
        }
    }
}

template <typename TreeT>
Interactions fmmmm_interactions(const TreeT& obs_tree, const TreeT& src_tree,
    double inner_r, double outer_r, size_t order)
{
    auto interaction_lists = init_interaction_lists(obs_tree, src_tree);

    up_collect(src_tree, interaction_lists, src_tree.root());
    down_collect(obs_tree, interaction_lists, obs_tree.root());
    traverse(
        obs_tree, src_tree, interaction_lists,
        obs_tree.root(), src_tree.root(),
        inner_r, outer_r, order
    );

    return compress_interaction_lists(interaction_lists);
}

template 
Interactions fmmmm_interactions(const Octree<2>& obs_tree, const Octree<2>& src_tree,
    double inner_r, double outer_r, size_t order);
template 
Interactions fmmmm_interactions(const Octree<3>& obs_tree, const Octree<3>& src_tree,
    double inner_r, double outer_r, size_t order);

template 
Interactions fmmmm_interactions(const KDTree<2>& obs_tree, const KDTree<2>& src_tree,
    double inner_r, double outer_r, size_t order);
template 
Interactions fmmmm_interactions(const KDTree<3>& obs_tree, const KDTree<3>& src_tree,
    double inner_r, double outer_r, size_t order);
