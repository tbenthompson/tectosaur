#pragma once

#include "octree.hpp"

namespace tectosaur {

using FMMSurf = std::shared_ptr<std::vector<Vec3>>;

struct UpwardNode {
    using Ptr = std::shared_ptr<UpwardNode>;

    OctreeNode::Ptr node;
    std::vector<Vec3> equiv_surf;
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

Upward up_up_up(Octree o, FMMSurf fmm_surf);

struct SparseMat {
    std::vector<int> rows;
    std::vector<int> cols;
    std::vector<double> vals;
};

SparseMat go_go_go(Upward src_tree, Octree obs_tree);

} //end namespace tectosaur
