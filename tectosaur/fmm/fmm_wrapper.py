import os
import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer
from tectosaur import setup_logger

import tectosaur.fmm
from tectosaur.kernels import kernels

from cppimport import cppimport
fmm = cppimport("tectosaur.fmm.fmm")

logger = setup_logger(__name__)

#TODO: This file should be split up!
# I think the gpu_data thing should become a FMM class.

two = fmm.two
three = fmm.three
module = dict()
module[2] = two
module[3] = three

n_workers_per_block = 128
n_c2e_block_rows = 16

def get_gpu_module(surf, K, float_type):
    args = dict(
        n_workers_per_block = n_workers_per_block,
        n_c2e_block_rows = n_c2e_block_rows,
        gpu_float_type = gpu.np_to_c_type(float_type),
        surf = surf,
        K = K
    )
    gpu_module = gpu.load_gpu(
        'gpu_kernels.cl',
        tmpl_dir = tectosaur.fmm.source_dir,
        tmpl_args = args
    )
    return gpu_module

def report_interactions(fmm_mat):
    n_obs_pts = fmm_mat.obs_tree.pts.shape[0]
    n_src_pts = fmm_mat.src_tree.pts.shape[0]
    dim = fmm_mat.obs_tree.pts.shape[1]

    level_ops = ['m2m', 'l2l']
    ops = [
        ('p2m', True, False),
        ('p2l', True, False),
        ('m2l', True, True),
        ('p2p', False, False),
        ('m2p', False, True),
        ('l2p', False, True)
    ]
    interactions = dict()
    for op_name, obs_surf, src_surf in ops:
        interactions[op_name] = module[dim].count_interactions(
            getattr(fmm_mat, op_name + '_new'),
            fmm_mat.obs_tree, fmm_mat.src_tree,
            obs_surf, src_surf, len(fmm_mat.surf)
        )

    direct_i = n_obs_pts * n_src_pts
    fmm_i = sum([v for k,v in interactions.items()])

    logger.debug('compression factor: ' + str(fmm_i / direct_i))
    logger.debug('# obs pts: ' + str(n_obs_pts))
    logger.debug('# src pts: ' + str(n_src_pts))
    logger.debug('total tree interactions: %e' % fmm_i)
    for k, v in interactions.items():
        logger.debug('total %s interactions: %e' % (k, v))

def data_to_gpu(fmm_mat, float_type):
    src_tree_nodes = fmm_mat.src_tree.nodes
    obs_tree_nodes = fmm_mat.obs_tree.nodes

    gd = dict()

    surf = np.array(fmm_mat.surf)
    K_name = fmm_mat.cfg.kernel_name
    K = kernels[K_name]

    gd['float_type'] = float_type
    gd['fmm_mat'] = fmm_mat
    gd['module'] = get_gpu_module(surf, K, float_type)
    for a in ['s', 'p']:
        for b in ['s', 'p']:
            name = a + '2' + b
            gd[name] = getattr(gd['module'], name + '_' + K_name)
    gd['c2e'] = gd['module'].c2e_kernel

    gd['obs_pts'] = gpu.to_gpu(fmm_mat.obs_tree.pts, float_type)
    gd['obs_normals'] = gpu.to_gpu(fmm_mat.obs_normals, float_type)
    gd['src_pts'] = gpu.to_gpu(fmm_mat.src_tree.pts, float_type)
    gd['src_normals'] = gpu.to_gpu(fmm_mat.src_normals, float_type)

    gd['tensor_dim'] = fmm_mat.cfg.tensor_dim
    gd['n_surf_pts'] = np.int32(surf.shape[0])
    gd['n_surf_dofs'] = gd['n_surf_pts'] * gd['tensor_dim']
    gd['params'] = gpu.to_gpu(np.array(fmm_mat.cfg.params), float_type)

    gd['n_multipoles'] = gd['n_surf_dofs'] * len(src_tree_nodes)
    gd['n_locals'] = gd['n_surf_dofs'] * len(obs_tree_nodes)

    gd['in'] = gpu.empty_gpu(gd['tensor_dim'] * gd['src_pts'].shape[0], float_type)
    gd['out'] = gpu.empty_gpu(gd['tensor_dim'] * gd['obs_pts'].shape[0], float_type)
    gd['m_check'] = gpu.empty_gpu(gd['n_multipoles'], float_type)
    gd['multipoles'] = gpu.empty_gpu(gd['n_multipoles'], float_type)
    gd['l_check'] = gpu.empty_gpu(gd['n_locals'], float_type)
    gd['locals'] = gpu.empty_gpu(gd['n_locals'], float_type)

    for name, tree in [('src', src_tree_nodes), ('obs', obs_tree_nodes)]:
        gd[name + '_n_center'] = gpu.to_gpu(
            np.array([n.bounds.center for n in tree]).flatten(), float_type
        )
        gd[name + '_n_R'] = gpu.to_gpu(
            np.array([n.bounds.R for n in tree]), float_type
        )
        gd[name + '_n_start'] = gpu.to_gpu(
            np.array([n.start for n in tree]), np.int32
        )
        gd[name + '_n_end'] = gpu.to_gpu(
            np.array([n.end for n in tree]), np.int32
        )

    n_src_levels = len(fmm_mat.m2m_new)
    gd['u2e_node_n_idx'] = [
        gpu.to_gpu(fmm_mat.u2e[level].src_n_idx, np.int32) for level in range(n_src_levels)
    ]
    gd['u2e_ops'] = gpu.to_gpu(fmm_mat.u2e_ops, float_type)

    n_obs_levels = len(fmm_mat.l2l_new)
    gd['d2e_node_n_idx'] = [
        gpu.to_gpu(fmm_mat.d2e[level].src_n_idx, np.int32) for level in range(n_obs_levels)
    ]
    gd['d2e_ops'] = gpu.to_gpu(fmm_mat.d2e_ops, float_type)

    return gd

def get_op(op_name, gd):
    return gd[op_name[:3].replace('m', 's').replace('l', 's')]

def get_block_data(op_name, data_name, gd, suffix = ''):
    saved_name = op_name + '_' + data_name
    if saved_name not in gd:
        if len(op_name) > 3:
            level = int(op_name[3:])
            gd[saved_name] = gpu.to_gpu(
                getattr(getattr(gd['fmm_mat'], op_name[:3] + suffix)[level], data_name),
                np.int32
            )
        else:
            gd[saved_name] = gpu.to_gpu(
                getattr(getattr(gd['fmm_mat'], op_name + suffix), data_name),
                np.int32
            )
    return gd[saved_name]

def get_n_blocks(op_name, obs_type, gd):
    return gd[op_name + '_obs_n_idx'].shape[0]

def to_ev_list(maybe_evs):
    return [ev for ev in maybe_evs if ev is not None]

def gpu_fmm_op(op_name, out_name, in_name, obs_type, src_type, gd, wait_for):
    op = get_op(op_name, gd)

    call_data = [
        get_block_data(op_name, 'obs_n_idxs', gd, suffix = '_new'),
        get_block_data(op_name, 'obs_src_starts', gd, suffix = '_new'),
        get_block_data(op_name, 'src_n_idxs', gd, suffix = '_new'),
    ]
    for name, type in [('obs', obs_type), ('src', src_type)]:
        if type[0] == 'pts':
            call_data.extend([
                gd[type[1] + '_n_start'], gd[type[1] + '_n_end'],
                gd[type[1] + '_pts'], gd[type[1] + '_normals']
            ])
        else:
            call_data.extend([
                gd[type[1] + '_n_center'], gd[type[1] + '_n_R'],
                gd['float_type'](type[2]),
            ])

    n_obs_n = call_data[0].shape[0]
    if n_obs_n > 0:
        return op(
            gpu.gpu_queue,
            (n_obs_n * n_workers_per_block,), (n_workers_per_block,),
            gd[out_name].data, gd[in_name].data,
            np.int32(n_obs_n), gd['params'].data,
            *[d.data for d in call_data],
            wait_for = to_ev_list(wait_for)
        )
    else:
        return None

def gpu_p2p(fmm_mat, gd):
    return gpu_fmm_op(
        'p2p', 'out', 'in', ('pts', 'obs'), ('pts', 'src'), gd, []
    )

def gpu_m2p(fmm_mat, gd, u2e_ev):
    return gpu_fmm_op(
        'm2p', 'out', 'multipoles',
        ('pts', 'obs'), ('surf', 'src', fmm_mat.cfg.inner_r), gd, u2e_ev
    )

def gpu_p2m(fmm_mat, gd):
    return gpu_fmm_op(
        'p2m', 'm_check', 'in',
        ('surf', 'src', fmm_mat.cfg.outer_r), ('pts', 'src'), gd, []
    )

def gpu_m2m(fmm_mat, gd, level, u2e_ev):
    return gpu_fmm_op(
        'm2m' + str(level), 'm_check', 'multipoles',
        ('surf', 'src', fmm_mat.cfg.outer_r), ('surf', 'src', fmm_mat.cfg.inner_r), gd, []
    )

def gpu_p2l(fmm_mat, gd):
    return gpu_fmm_op(
        'p2l', 'l_check', 'in',
        ('surf', 'obs', fmm_mat.cfg.inner_r), ('pts', 'src'), gd, []
    )

def gpu_m2l(fmm_mat, gd, u2e_ev):
    return gpu_fmm_op(
        'm2l', 'l_check', 'multipoles',
        ('surf', 'obs', fmm_mat.cfg.inner_r), ('surf', 'src', fmm_mat.cfg.inner_r), gd, []
    )

def gpu_l2l(fmm_mat, gd, level, d2e_ev):
    return gpu_fmm_op(
        'l2l' + str(level), 'l_check', 'locals',
        ('surf', 'obs', fmm_mat.cfg.inner_r), ('surf', 'obs', fmm_mat.cfg.outer_r), gd, []
    )

def gpu_l2p(fmm_mat, gd, d2e_ev):
    return gpu_fmm_op(
        'l2p', 'out', 'locals',
        ('pts', 'obs'), ('surf', 'obs', fmm_mat.cfg.outer_r), gd, []
    )

def gpu_c2e(fmm_mat, gd, level, depth, evs, d_or_u, out_arr, in_arr):
    c2e = gd['c2e']
    n_c2e = gd[d_or_u + '2e_node_n_idx'][level].shape[0]
    n_c2e_rows = gd['n_surf_dofs']
    if n_c2e > 0:
        n_rows = int(np.ceil(n_c2e / n_c2e_block_rows) * n_c2e_block_rows)
        n_cols = int(np.ceil(n_c2e_rows / n_c2e_block_rows) * n_c2e_block_rows)
        return c2e(
            gpu.gpu_queue,
            (n_rows, n_cols),
            (n_c2e_block_rows, n_c2e_block_rows),
            out_arr.data, in_arr.data,
            np.int32(n_c2e), np.int32(n_c2e_rows),
            gd[d_or_u + '2e_node_n_idx'][level].data,
            np.int32(depth),
            gd[d_or_u + '2e_ops'].data,
            wait_for = to_ev_list(evs)
        )
    else:
        return None

def gpu_d2e(fmm_mat, gd, level, evs):
    return gpu_c2e(fmm_mat, gd, level, level, evs, 'd', gd['locals'], gd['l_check'])

def gpu_u2e(fmm_mat, gd, level, evs):
    n_depth = fmm_mat.src_tree.max_height - level;
    return gpu_c2e(fmm_mat, gd, level, n_depth, evs, 'u', gd['multipoles'], gd['m_check'])

def print_timing(p2m_ev, m2m_evs, u2e_evs,
        p2l_ev, m2l_ev, l2l_evs, d2e_evs,
        p2p_ev, m2p_ev, l2p_ev):

    def get_time(ev):
        if ev is not None:
            return (ev.profile.end - ev.profile.start) * 1e-9
        return 0

    logger.debug('p2m took ' + str(get_time(p2m_ev)))
    logger.debug('m2m took ' + str(sum([get_time(level) for level in m2m_evs])))
    logger.debug('u2e took ' + str(sum([get_time(level) for level in u2e_evs])))
    logger.debug('p2l took ' + str(get_time(p2l_ev)))
    logger.debug('m2l took ' + str(get_time(m2l_ev)))
    logger.debug('l2l took ' + str(sum([get_time(level) for level in l2l_evs])))
    logger.debug('d2e took ' + str(sum([get_time(level) for level in d2e_evs])))
    logger.debug('p2p took ' + str(get_time(p2p_ev)))
    logger.debug('m2p took ' + str(get_time(m2p_ev)))
    logger.debug('l2p took ' + str(get_time(l2p_ev)))

def prep_data_for_eval(gd, input_vals):
    gd['in'][:] = input_vals.astype(gd['float_type']).flatten()
    gd['out'][:] = 0
    gd['m_check'][:] = 0
    gd['multipoles'][:] = 0
    gd['l_check'][:] = 0
    gd['locals'][:] = 0

def eval_ocl(fmm_mat, input_vals, gpu_data, should_print_timing = True):
    t = Timer(silent = True)

    prep_data_for_eval(gpu_data, input_vals)
    t.report('prep for eval')

    p2m_ev = gpu_p2m(fmm_mat, gpu_data)
    t.report('p2m')
    p2p_ev = gpu_p2p(fmm_mat, gpu_data)
    t.report('p2p')
    p2l_ev = gpu_p2l(fmm_mat, gpu_data)
    t.report('p2l')

    m2m_evs = []
    u2e_evs = []
    u2e_evs.append(gpu_u2e(fmm_mat, gpu_data, 0, [p2m_ev]))

    for i in range(1, len(fmm_mat.m2m_new)):
        m2m_evs.append(gpu_m2m(fmm_mat, gpu_data, i, [u2e_evs[-1]]))
        u2e_evs.append(gpu_u2e(fmm_mat, gpu_data, i, [m2m_evs[-1]]))
    t.report('m2m')

    m2l_ev = gpu_m2l(fmm_mat, gpu_data, [u2e_evs[-1]])
    t.report('m2l')
    m2p_ev = gpu_m2p(fmm_mat, gpu_data, [u2e_evs[-1]])
    t.report('m2p')

    l2l_evs = []
    d2e_evs = []
    d2e_wait_for = [] if m2l_ev is None else [m2l_ev]
    if p2l_ev is not None:
        d2e_wait_for.append(p2l_ev)
    d2e_evs.append(gpu_d2e(fmm_mat, gpu_data, 0, d2e_wait_for))
    t.report('p2l')

    for i in range(1, len(fmm_mat.l2l_new)):
        l2l_evs.append(gpu_l2l(fmm_mat, gpu_data, i, d2e_evs[-1]))
        d2e_evs.append(gpu_d2e(fmm_mat, gpu_data, i, [l2l_evs[-1]]))
    t.report('l2l')

    l2p_ev = gpu_l2p(fmm_mat, gpu_data, d2e_evs[-1])

    t.report('l2p')
    if l2p_ev is not None:
        l2p_ev.wait()
    if m2p_ev is not None:
        m2p_ev.wait()
    if p2p_ev is not None:
        p2p_ev.wait()
    t.report('fmm done running')

    retval = gpu_data['out'].get()
    t.report('fmm data returned')

    if should_print_timing:
        print_timing(
            p2m_ev, m2m_evs, u2e_evs,
            p2l_ev, m2l_ev, l2l_evs, d2e_evs,
            p2p_ev, m2p_ev, l2p_ev
        )

    return retval
