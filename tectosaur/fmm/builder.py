import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.fmm.c2e import build_c2e

from tectosaur import setup_logger
logger = setup_logger(__name__)

def make_tree(pts, cfg, max_pts_per_cell):
    return cfg.traversal_module.Tree(pts, max_pts_per_cell)

class FMM:
    def __init__(self, obs_tree, obs_normals, src_tree, src_normals, cfg):
        self.cfg = cfg
        self.obs_tree = obs_tree
        self.src_tree = src_tree
        self.gpu_data = dict()

        self.setup_interactions()
        self.collect_gpu_ops()
        self.setup_output_sizes()
        self.params_to_gpu()
        self.tree_to_gpu(obs_normals, src_normals)
        self.interactions_to_gpu()
        self.d2e_u2e_ops_to_gpu()

    def setup_interactions(self):
        self.interactions = self.cfg.traversal_module.fmmmm_interactions(
            self.obs_tree, self.src_tree, self.cfg.inner_r, self.cfg.outer_r, self.cfg.order
        )

    def collect_gpu_ops(self):
        self.gpu_ops = dict()
        for a in ['s', 'p']:
            for b in ['s', 'p']:
                name = a + '2' + b
                self.gpu_ops[name] = getattr(self.cfg.gpu_module, name + '_' + self.cfg.K.name)
        self.gpu_ops['c2e'] = self.cfg.gpu_module.c2e_kernel

    def setup_output_sizes(self):
        self.n_surf_pts = self.cfg.surf.shape[0]
        self.n_surf_dofs = self.n_surf_pts * self.cfg.K.tensor_dim
        self.n_multipoles = self.n_surf_dofs * self.src_tree.n_nodes
        self.n_locals = self.n_surf_dofs * self.obs_tree.n_nodes
        self.n_input = self.cfg.K.tensor_dim * self.src_tree.pts.shape[0]
        self.n_output = self.cfg.K.tensor_dim * self.obs_tree.pts.shape[0]

    def float_gpu(self, arr):
        return gpu.to_gpu(arr, self.cfg.float_type)

    def int_gpu(self, arr):
        return gpu.to_gpu(arr, np.int32)

    def params_to_gpu(self):
        self.gpu_data['params'] = self.float_gpu(self.cfg.params)

    def tree_to_gpu(self, obs_normals, src_normals):
        gd = self.gpu_data

        gd['obs_pts'] = self.float_gpu(self.obs_tree.pts)
        gd['obs_normals'] = self.float_gpu(obs_normals)
        gd['src_pts'] = self.float_gpu(self.src_tree.pts)
        gd['src_normals'] = self.float_gpu(src_normals)

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
            self.gpu_data[name + '_' + data_name] = self.int_gpu(getattr(op, data_name))

    def d2e_u2e_ops_to_gpu(self):
        gd = self.gpu_data

        gd['u2e_obs_n_idxs'] = [
            self.int_gpu(self.interactions.u2e[level].obs_n_idxs)
            for level in range(len(self.interactions.m2m))
        ]

        gd['d2e_obs_n_idxs'] = [
            self.int_gpu(self.interactions.d2e[level].obs_n_idxs)
            for level in range(len(self.interactions.l2l))
        ]

        gd['u2e_ops'] = self.float_gpu(
            build_c2e(self.src_tree, self.cfg.outer_r, self.cfg.inner_r, self.cfg),
        )
        gd['d2e_ops'] = self.float_gpu(
            build_c2e(self.obs_tree, self.cfg.inner_r, self.cfg.outer_r, self.cfg),
        )

    def to_orig(self, output_tree):
        orig_idxs = np.array(self.obs_tree.orig_idxs)
        output_tree = output_tree.reshape((-1, self.cfg.K.tensor_dim))
        output_orig = np.empty_like(output_tree)
        output_orig[orig_idxs,:] = output_tree
        return output_orig

def report_interactions(fmm_obj):
    dim = fmm_obj.obs_tree.pts.shape[1]
    def count_interactions(op_name, op):
        obs_surf = False if op_name[2] == 'p' else True
        src_surf = False if op_name[0] == 'p' else True
        return fmm_obj.cfg.traversal_module.count_interactions(
            op, fmm_obj.obs_tree, fmm_obj.src_tree,
            obs_surf, src_surf, fmm_obj.cfg.order
        )

    n_obs_pts = fmm_obj.obs_tree.pts.shape[0]
    n_src_pts = fmm_obj.src_tree.pts.shape[0]
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

    direct_i = n_obs_pts * n_src_pts
    fmm_i = sum([v for k,v in interactions.items()])

    logger.debug('compression factor: ' + str(fmm_i / direct_i))
    logger.debug('# obs pts: ' + str(n_obs_pts))
    logger.debug('# src pts: ' + str(n_src_pts))
    logger.debug('total tree interactions: %e' % fmm_i)
    for k, v in interactions.items():
        logger.debug('total %s interactions: %e' % (k, v))
