#pragma once

#include "octree.hpp"

namespace tectosaur {

struct upward_traversal_node {
    using ptr = std::shared_ptr<upward_traversal_node>;

    OctreeNode::Ptr node;
    std::vector<double> p2m_op;
    std::array<tl::future<std::vector<double>>,8> m2m_ops;
    std::array<tl::future<upward_traversal_node::ptr>,8> children; 

    upward_traversal_node() = default;
    upward_traversal_node(OctreeNode::Ptr& n);

    template <typename Archive>
    void serialize(Archive& ar) {}
};

struct upward_traversal {
    tl::future<upward_traversal_node> root;
};

upward_traversal up_up_up(Octree o);

} //end namespace tectosaur
