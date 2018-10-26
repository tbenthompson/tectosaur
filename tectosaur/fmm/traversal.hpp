#pragma once

#include <vector>
#include <cstddef>

struct CompressedInteractionList {
    std::vector<size_t> obs_n_idxs;//TODO: Make these 64-bit.
    std::vector<size_t> obs_src_starts;
    std::vector<size_t> src_n_idxs;
};

template <typename TreeT>
size_t count_interactions(const CompressedInteractionList& list,
    const TreeT& obs_tree, const TreeT& src_tree,
    bool obs_surf, bool src_surf, int n_surf) 
{
    size_t n = 0;
    for (size_t i = 0; i < list.obs_n_idxs.size(); i++) {
        int obs_n_idx = list.obs_n_idxs[i];
        int n_obs = n_surf;
        if (!obs_surf) {
            auto& n = obs_tree.nodes[obs_n_idx];
            n_obs = n.end - n.start;
        }
        for (size_t j = list.obs_src_starts[i]; j < list.obs_src_starts[i + 1]; j++) {
            size_t src_n_idx = list.src_n_idxs[j];
            int n_src = n_surf;
            if (!src_surf) {
                auto& n = src_tree.nodes[src_n_idx];
                n_src = n.end - n.start;
            }
            n += n_obs * n_src;
        }
    }
    return n;
}

struct Interactions {
    CompressedInteractionList p2m;
    std::vector<CompressedInteractionList> m2m;
    std::vector<CompressedInteractionList> u2e;

    CompressedInteractionList l2p;
    std::vector<CompressedInteractionList> l2l;
    std::vector<CompressedInteractionList> d2e;

    CompressedInteractionList p2p;
    CompressedInteractionList p2l;
    CompressedInteractionList m2p;
    CompressedInteractionList m2l;
};

template <typename TreeT>
Interactions fmmmm_interactions(const TreeT& obs_tree, const TreeT& src_tree,
    double inner_r, double outer_r, size_t order, bool treecode);
