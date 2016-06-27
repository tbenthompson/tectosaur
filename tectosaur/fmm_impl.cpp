#include "fmm_impl.hpp"

#include <cereal/types/vector.hpp>

namespace tectosaur {

upward_traversal_node::upward_traversal_node(OctreeNode::Ptr& n) {
    // make this the constructor
    if (n->is_leaf) {
        // compute p2m op.
        p2m_op = 
    } else {
        for (size_t i = 0; i < 8; i++) {
            children[i] = n->children[i].then([] (OctreeNode::Ptr& n) {
                return std::make_shared<upward_traversal_node>(n);
            });
            m2m_ops[i] = children[i].then(
                [] (OctreeNode::Ptr& child_n, upward_traversal_node::ptr& up_n) {
                    // compute m2m op!
                    return std::vector<double>{};
                },
                n
            );
            // then collect the necessary child info for M2M.
            // perform M2M kernel.
        }
        // return {n, children};
        // tasks.m2ms.push_back({cell});
    }
}

upward_traversal up_up_up(Octree o) {
    auto result = o.root
        .then([] (OctreeNode::Ptr& n) {
            return upward_traversal_node(n); 
        });
    return {result};
}

}
