import numpy as np

from tectosaur.util.quadrature import gauss2d_tri, gauss4d_tri
from tectosaur.kernels import kernels
import tectosaur.util.gpu as gpu

from tectosaur.util.cpp import imp
traversal_ext = imp("tectosaur.fmm.traversal_wrapper")

traversal_module = traversal_ext.three.octree

import logging
logger = logging.getLogger(__name__)

def make_tree(m, max_pts_per_cell):
    tri_pts = m[0][m[1]]
    centers = np.mean(tri_pts, axis = 1)
    pt_dist = tri_pts - centers[:,np.newaxis,:]
    Rs = np.max(np.linalg.norm(pt_dist, axis = 2), axis = 1)
    tree = traversal_module.Tree.build(centers, Rs, max_pts_per_cell)
    return tree

class TSFMM:
    def __init__(self, obs_m, src_m, **kwargs):
        self.cfg = kwargs
        self.obs_m = obs_m
        self.src_m = src_m
        self.obs_tree = make_tree(self.obs_m, self.cfg['max_pts_per_cell'])
        self.src_tree = make_tree(self.src_m, self.cfg['max_pts_per_cell'])
        self.gpu_data = dict()

        self.setup_interactions()
        self.setup_output_sizes()
        self.params_to_gpu()
        self.tree_to_gpu()
        self.interactions_to_gpu()
        self.load_gpu_module()
        self.setup_arrays()

    def load_gpu_module(self):
        quad = gauss2d_tri(self.cfg['quad_order'])
        self.gpu_module = gpu.load_gpu(
            'fmm/ts_kernels.cl',
            tmpl_args = dict(
                order = self.cfg['order'],
                gpu_float_type = gpu.np_to_c_type(self.cfg['float_type']),
                quad_pts = quad[0],
                quad_wts = quad[1]
            )
        )
        quad4d = gauss4d_tri(self.cfg['quad_order'], self.cfg['quad_order'])
        args = dict(
            n_workers_per_block = self.cfg['n_workers_per_block'],
            gpu_float_type = gpu.np_to_c_type(self.cfg['float_type']),
            surf_pts = np.array([[0,0,0]]),
            surf_tris = np.array([[0,0,0]]),
            quad_pts = quad4d[0],
            quad_wts = quad4d[1],
            K = kernels['elasticU3'] #TODO #TODO #TODO #TODO #TODO #TODO
        )
        self.old_gpu_module = gpu.load_gpu(
            'fmm/tri_gpu_kernels.cl',
            tmpl_args = args
        )

    def setup_interactions(self):
        self.interactions = traversal_module.fmmmm_interactions(
            self.obs_tree, self.src_tree, 1.0, self.cfg['mac'],
            0, True
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

    def tree_to_gpu(self):
        gd = self.gpu_data

        gd['obs_pts'] = self.float_gpu(self.obs_m[0])
        gd['obs_tris'] = self.int_gpu(self.obs_m[1][self.obs_tree.orig_idxs])
        gd['src_pts'] = self.float_gpu(self.src_m[0])
        gd['src_tris'] = self.int_gpu(self.src_m[1][self.src_tree.orig_idxs])

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

    def setup_arrays(self):
        self.gpu_multipoles = gpu.empty_gpu(self.n_multipoles, self.cfg['float_type'])
        self.gpu_out = gpu.empty_gpu(self.n_output, self.cfg['float_type'])
        self.gpu_in = gpu.empty_gpu(self.n_input, self.cfg['float_type'])

    def p2m(self):
        self.gpu_module.p2m(
            self.gpu_multipoles,
            self.gpu_in,
            self.gpu_data['p2m_obs_n_idxs'],
            self.gpu_data['src_n_C'],
            self.gpu_data['src_n_start'],
            self.gpu_data['src_n_end'],
            self.gpu_data['src_pts'],
            self.gpu_data['src_tris'],
            grid = (self.gpu_data['p2m_obs_n_idxs'].shape[0],1,1),
            block = (1,1,1)
        )

    def m2m(self, level):
        self.gpu_module.m2m(
            self.gpu_multipoles,
            self.gpu_data['m2m' + str(level) + '_obs_n_idxs'],
            self.gpu_data['m2m' + str(level) + '_obs_src_starts'],
            self.gpu_data['m2m' + str(level) + '_src_n_idxs'],
            self.gpu_data['src_n_C'],
            grid = (self.gpu_data['m2m' + str(level) + '_obs_n_idxs'].shape[0],1,1),
            block = (1,1,1)
        )

    def m2p(self):
        n_obs_n = self.gpu_data['m2p_obs_n_idxs'].shape[0]
        if n_obs_n == 0:
            return
        self.gpu_module.m2p_U(
            self.gpu_out,
            self.gpu_multipoles,
            self.gpu_data['params'],
            self.gpu_data['m2p_obs_n_idxs'],
            self.gpu_data['m2p_obs_src_starts'],
            self.gpu_data['m2p_src_n_idxs'],
            self.gpu_data['obs_n_start'],
            self.gpu_data['obs_n_end'],
            self.gpu_data['obs_pts'],
            self.gpu_data['obs_tris'],
            self.gpu_data['src_n_C'],
            grid = (n_obs_n,1,1),
            block = (1,1,1)
        )

    def p2p(self):
        n_obs_n = self.gpu_data['p2p_obs_n_idxs'].shape[0]
        if n_obs_n == 0:
            return
        self.old_gpu_module.p2p_elasticU3(
            self.gpu_out,
            self.gpu_in,
            np.int32(n_obs_n),
            self.gpu_data['params'],
            self.gpu_data['p2p_obs_n_idxs'],
            self.gpu_data['p2p_obs_src_starts'],
            self.gpu_data['p2p_src_n_idxs'],
            self.gpu_data['obs_n_start'],
            self.gpu_data['obs_n_end'],
            self.gpu_data['obs_pts'],
            self.gpu_data['obs_tris'],
            self.gpu_data['src_n_start'],
            self.gpu_data['src_n_end'],
            self.gpu_data['src_pts'],
            self.gpu_data['src_tris'],
            grid = (n_obs_n, 1, 1),
            block = (self.cfg['n_workers_per_block'], 1, 1)
        )


    def dot(self, v):
        self.gpu_in[:] = self.to_tree(v)
        self.gpu_out.fill(0)

        self.p2p()
        self.p2m()
        for i in range(1, len(self.interactions.m2m)):
            self.m2m(i)
        self.m2p()

        return self.to_orig(self.gpu_out.get())

def report_interactions(fmm_obj):
    def count_interactions(op_name, op):
        obs_surf = False if op_name[2] == 'p' else True
        src_surf = False if op_name[0] == 'p' else True
        return traversal_module.count_interactions(
            op, fmm_obj.obs_tree, fmm_obj.src_tree,
            obs_surf, src_surf, 1
        )

    n_obs_tris = fmm_obj.obs_m[1].shape[0]
    n_src_tris = fmm_obj.src_m[1].shape[0]

    p2p = count_interactions('p2p', fmm_obj.interactions.p2p)
    total = n_obs_tris * n_src_tris
    not_p2p = total - p2p

    logger.info('# obs tris: ' + str(n_obs_tris))
    logger.info('# src tris: ' + str(n_src_tris))
    logger.info('total: ' + str(total))
    logger.info('p2p percent: ' + str(p2p / total))
    logger.info('m2p percent: ' + str(not_p2p / total))
