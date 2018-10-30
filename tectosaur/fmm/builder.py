import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.fmm.c2e import build_c2e

import logging
logger = logging.getLogger(__name__)

def make_tree(m, cfg, max_pts_per_cell):
    tri_pts = m[0][m[1]]
    centers = np.mean(tri_pts, axis = 1)
    pt_dist = tri_pts - centers[:,np.newaxis,:]
    Rs = np.max(np.linalg.norm(pt_dist, axis = 2), axis = 1)
    tree = cfg.traversal_module.Tree.build(centers, Rs, max_pts_per_cell)
    return tree

class FMM:
    def __init__(self, obs_tree, obs_m, src_tree, src_m, cfg):
        self.cfg = cfg
        self.obs_tree = obs_tree
        self.obs_m = obs_m
        self.src_tree = src_tree
        self.src_m = src_m
        self.gpu_data = dict()

        self.setup_interactions()
        self.collect_gpu_ops()
        self.setup_output_sizes()
        self.params_to_gpu()
        self.tree_to_gpu(obs_m, src_m)
        self.interactions_to_gpu()
        self.d2e_u2e_ops_to_gpu()

    def setup_interactions(self):
        self.interactions = self.cfg.traversal_module.fmmmm_interactions(
            self.obs_tree, self.src_tree, self.cfg.inner_r, self.cfg.outer_r,
            self.cfg.order, self.cfg.treecode
        )

    def collect_gpu_ops(self):
        self.gpu_ops = dict()
        for a in ['s', 'p']:
            for b in ['s', 'p']:
                name = a + '2' + b
                self.gpu_ops[name] = getattr(self.cfg.gpu_module, name + '_' + self.cfg.K.name)

    def setup_output_sizes(self):
        self.n_surf_tris = self.cfg.surf[1].shape[0]
        self.n_surf_dofs = self.n_surf_tris * 9
        self.n_multipoles = self.n_surf_dofs * self.src_tree.n_nodes
        self.n_locals = self.n_surf_dofs * self.obs_tree.n_nodes
        self.n_input = self.src_m[1].shape[0] * 9
        self.n_output = self.obs_m[1].shape[0] * 9

    def float_gpu(self, arr):
        return gpu.to_gpu(arr, self.cfg.float_type)

    def int_gpu(self, arr):
        return gpu.to_gpu(arr, np.int32)

    def params_to_gpu(self):
        self.gpu_data['params'] = self.float_gpu(self.cfg.params)

    def tree_to_gpu(self, obs_m, src_m):
        gd = self.gpu_data

        gd['obs_pts'] = self.float_gpu(obs_m[0])
        gd['obs_tris'] = self.int_gpu(obs_m[1][self.src_tree.orig_idxs])
        gd['src_pts'] = self.float_gpu(src_m[0])
        gd['src_tris'] = self.int_gpu(src_m[1][self.src_tree.orig_idxs])

        obs_tree_nodes = self.obs_tree.nodes
        src_tree_nodes = self.src_tree.nodes

        for name, tree in [('src', self.src_tree), ('obs', self.obs_tree)]:
            gd[name + '_n_C'] = self.float_gpu(tree.node_centers)
            gd[name + '_n_R'] = self.float_gpu(tree.node_Rs)

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

    def d2e_u2e_ops_to_gpu(self):
        gd = self.gpu_data

        gd['u2e_obs_n_idxs'] = [
            self.int_gpu(np.array(self.interactions.u2e[level].obs_n_idxs, copy = False))
            for level in range(len(self.interactions.m2m))
        ]

        gd['d2e_obs_n_idxs'] = [
            self.int_gpu(np.array(self.interactions.d2e[level].obs_n_idxs, copy = False))
            for level in range(len(self.interactions.l2l))
        ]

        self.u2e_ops = build_c2e(
            gd['params'], gd['src_n_C'], gd['src_n_R'],
            self.cfg.outer_r, self.cfg.inner_r, self.cfg
        )
        self.d2e_ops = build_c2e(
            gd['params'], gd['obs_n_C'], gd['obs_n_R'],
            self.cfg.inner_r, self.cfg.outer_r, self.cfg
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

def report_interactions(fmm_obj):
    dim = fmm_obj.obs_m[1].shape[1]
    order = fmm_obj.cfg.surf[1].shape[0]
    def count_interactions(op_name, op):
        obs_surf = False if op_name[2] == 'p' else True
        src_surf = False if op_name[0] == 'p' else True
        return fmm_obj.cfg.traversal_module.count_interactions(
            op, fmm_obj.obs_tree, fmm_obj.src_tree,
            obs_surf, src_surf, order
        )

    n_obs_tris = fmm_obj.obs_m[1].shape[0]
    n_src_tris = fmm_obj.src_m[1].shape[0]
    level_ops = ['m2m', 'l2l']
    ops = ['p2m', 'p2l', 'm2l', 'p2p', 'm2p', 'l2p']

    interactions = dict()
    for op_name in ops:
        op = getattr(fmm_obj.interactions, op_name)
        interactions[op_name] = count_interactions(op_name, op)

    for op_name in level_ops:
        ops = getattr(fmm_obj.interactions, op_name)
        for op in ops:
            if op_name not in interactions:
                interactions[op_name] = 0
            interactions[op_name] += count_interactions(op_name, op)

    direct_i = n_obs_tris * n_src_tris
    fmm_i = sum([v for k,v in interactions.items()])

    logger.debug('compression factor: ' + str(fmm_i / direct_i))
    logger.debug('# obs tris: ' + str(n_obs_tris))
    logger.debug('# src tris: ' + str(n_src_tris))
    logger.debug('total tree interactions: %e' % fmm_i)
    for k, v in interactions.items():
        logger.debug('total %s interactions: %e' % (k, v))
