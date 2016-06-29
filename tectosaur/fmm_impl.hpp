#pragma once

#include "octree.hpp"

namespace tectosaur {

using FMMSurf = std::shared_ptr<std::vector<Vec3>>;

struct UpwardNode {
    using Ptr = std::shared_ptr<UpwardNode>;

    OctreeNode::Ptr node;
    size_t m_dof;
    std::vector<Vec3> equiv_surf; // can be calced on the fly
    std::vector<double> p2m_op;
    std::array<tl::future<std::vector<double>>,8> m2m_ops;
    std::array<tl::future<UpwardNode::Ptr>,8> children; 

    UpwardNode() = default;
    UpwardNode(OctreeNode::Ptr& n, FMMSurf fmm_surf);

    template <typename Archive>
    void serialize(Archive& ar) {}
};

struct Upward {
    tl::future<UpwardNode::Ptr> root;
};

struct FMMConfig {
    double MAC;
    FMMSurf surf;
};

Upward up_up_up(Octree o, FMMSurf fmm_surf);

struct SparseMat {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
};

struct FMMMat {
    SparseMat p2p;
    SparseMat p2m;
    SparseMat m2m;
    SparseMat m2p;
    size_t n_m_dofs = 0;
};

FMMMat go_go_go(Upward src_tree, Octree obs_tree);

} //end namespace tectosaur
