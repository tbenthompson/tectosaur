import numpy as np

from tectosaur.util.quadrature import gauss2d_tri
from tectosaur.kernels import kernels
import tectosaur.util.gpu as gpu

from tectosaur.util.cpp import imp
traversal_ext = imp("tectosaur.fmm.traversal_wrapper")

def make_config(**kwargs):
    if 'treecode' not in kwargs:
        kwargs['treecode'] = True
    kwargs['quad'] = gauss2d_tri(kwargs['quad_order'])
    kwargs['gpu_module'] = gpu.load_gpu(
        'fmm/ts_kernels.cl',
        tmpl_args = dict(
            order = kwargs['order'],
            gpu_float_type = gpu.np_to_c_type(kwargs['float_type']),
            quad_pts = kwargs['quad'][0],
            quad_wts = kwargs['quad'][1]
        )
    )
    kwargs['traversal_module'] = traversal_ext.three.octree
    return kwargs

def make_tree(m, cfg, max_pts_per_cell):
    tri_pts = m[0][m[1]]
    centers = np.mean(tri_pts, axis = 1)
    pt_dist = tri_pts - centers[:,np.newaxis,:]
    Rs = np.max(np.linalg.norm(pt_dist, axis = 2), axis = 1)
    tree = cfg['traversal_module'].Tree.build(centers, Rs, max_pts_per_cell)
    return tree

class TSFMM:
    def __init__(self, obs_tree, obs_m, src_tree, src_m, cfg):
        self.cfg = cfg
        self.obs_tree = obs_tree
        self.obs_m = obs_m
        self.src_tree = src_tree
        self.src_m = src_m
        self.gpu_data = dict()

        self.setup_interactions()
        # self.collect_gpu_ops()
        self.setup_output_sizes()
        self.params_to_gpu()
        self.tree_to_gpu(obs_m, src_m)
        self.interactions_to_gpu()

    def setup_interactions(self):
        self.interactions = self.cfg['traversal_module'].fmmmm_interactions(
            self.obs_tree, self.src_tree, 1.0, self.cfg['mac'],
            self.cfg['order'], self.cfg['treecode']
        )

    def setup_output_sizes(self):
        order = self.cfg['order']
        # n dim = [0, order],
        # m dim = [0, n],
        # 4 moments,
        # 2 = real and imaginary parts
        self.n_multipoles = (order + 1) * (order + 1) * 4 * 2 * self.src_tree.n_nodes
        # self.n_locals = self.n_surf_dofs * self.obs_tree.n_nodes
        self.n_input = self.src_m[1].shape[0] * 9
        self.n_output = self.obs_m[1].shape[0] * 9

    def float_gpu(self, arr):
        return gpu.to_gpu(arr, self.cfg['float_type'])

    def int_gpu(self, arr):
        return gpu.to_gpu(arr, np.int32)

    def params_to_gpu(self):
        self.gpu_data['params'] = self.float_gpu(self.cfg['params'])

    def tree_to_gpu(self, obs_m, src_m):
        gd = self.gpu_data

        gd['obs_pts'] = self.float_gpu(obs_m[0])
        gd['obs_tris'] = self.int_gpu(obs_m[1][self.obs_tree.orig_idxs])
        gd['src_pts'] = self.float_gpu(src_m[0])
        gd['src_tris'] = self.int_gpu(src_m[1][self.src_tree.orig_idxs])

        obs_tree_nodes = self.obs_tree.nodes
        src_tree_nodes = self.src_tree.nodes

        for name, tree in [('src', self.src_tree), ('obs', self.obs_tree)]:
            gd[name + '_n_C'] = self.float_gpu(tree.node_centers)
            # gd[name + '_n_R'] = self.float_gpu(tree.node_Rs)

        for name, tree in [('src', src_tree_nodes), ('obs', obs_tree_nodes)]:
            gd[name + '_n_start'] = self.int_gpu(np.array([n.start for n in tree]))
            gd[name + '_n_end'] = self.int_gpu(np.array([n.end for n in tree]))

    def interactions_to_gpu(self):
        op_names = ['p2p', 'p2m', 'p2l', 'm2p', 'm2m', 'm2l', 'l2p', 'l2l']
        for name in op_names:
            op = getattr(self.interactions, name)
            if type(op) is list:
                for i, op_level in enumerate(op):
                    self.op_to_gpu(name + str(i), op_level)
            else:
                self.op_to_gpu(name, op)

    def op_to_gpu(self, name, op):
        for data_name in ['obs_n_idxs', 'obs_src_starts', 'src_n_idxs']:
            self.gpu_data[name + '_' + data_name] = self.int_gpu(
                np.array(getattr(op, data_name), copy = False)
            )

    def to_tree(self, input_orig):
        orig_idxs = np.array(self.src_tree.orig_idxs)
        input_orig = input_orig.reshape((-1,9))
        return input_orig[orig_idxs,:].flatten()

    def to_orig(self, output_tree):
        orig_idxs = np.array(self.obs_tree.orig_idxs)
        output_tree = output_tree.reshape((-1, 9))
        output_orig = np.empty_like(output_tree)
        output_orig[orig_idxs,:] = output_tree
        return output_orig.flatten()
