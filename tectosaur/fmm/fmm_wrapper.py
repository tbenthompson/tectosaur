import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer
from tectosaur import setup_logger

import tectosaur.fmm
from tectosaur.fmm.cfg import float_type
from tectosaur.kernels import kernels

from cppimport import cppimport
fmm = cppimport("tectosaur.fmm.fmm")

logger = setup_logger(__name__)

for k in dir(fmm):
    locals()[k] = getattr(fmm, k)

n_workers_per_block = 128

def get_gpu_module(surf, K):
    args = dict(
        n_workers_per_block = n_workers_per_block,
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
    surf_n = len(fmm_mat.surf)

    starts = fmm_mat.p2m.src_n_start
    ends = fmm_mat.p2m.src_n_end
    p2m_i = np.sum((ends - starts) * surf_n)

    m2m_i = (
        sum([len(fmm_mat.m2m[level].obs_n_idx) for level in range(len(fmm_mat.m2m))])
        * surf_n * surf_n
    )

    starts = fmm_mat.p2l.src_n_start
    ends = fmm_mat.p2l.src_n_end
    p2l_i = np.sum((ends - starts) * surf_n)

    m2l_i = len(fmm_mat.m2l.obs_n_idx) * surf_n * surf_n

    l2l_i = (
        sum([len(fmm_mat.l2l[level].obs_n_idx) for level in range(len(fmm_mat.l2l))])
        * surf_n * surf_n
    )

    starts = np.array(fmm_mat.m2p.obs_n_start)
    ends = np.array(fmm_mat.m2p.obs_n_end)
    m2p_i = np.sum((ends - starts) * surf_n)

    starts = fmm_mat.l2p.obs_n_start
    ends = fmm_mat.l2p.obs_n_end
    l2p_i = np.sum((ends - starts) * surf_n)

    obs_ends = np.array(fmm_mat.p2p.obs_n_end)
    obs_starts = np.array(fmm_mat.p2p.obs_n_start)
    src_ends = np.array(fmm_mat.p2p.src_n_end)
    src_starts = np.array(fmm_mat.p2p.src_n_start)
    p2p_i = np.sum((obs_ends - obs_starts) * (src_ends - src_starts))

    tree_i = p2p_i + m2p_i + p2m_i + m2m_i + m2l_i + l2l_i + l2p_i + p2l_i
    direct_i = len(fmm_mat.obs_tree.pts) * len(fmm_mat.src_tree.pts)

    logger.debug('compression factor: ' + str(tree_i / direct_i))
    logger.debug('# obs pts: ' + str(fmm_mat.obs_tree.pts.shape[0]))
    logger.debug('# src pts: ' + str(fmm_mat.src_tree.pts.shape[0]))
    logger.debug('total tree interactions: %e' % tree_i)
    logger.debug('total p2m interactions: %e' % p2m_i)
    logger.debug('total m2m interactions: %e' % m2m_i)
    logger.debug('total p2l interactions: %e' % p2l_i)
    logger.debug('total m2l interactions: %e' % m2l_i)
    logger.debug('total l2l interactions: %e' % l2l_i)
    logger.debug('total p2p interactions: %e' % p2p_i)
    logger.debug('total m2p interactions: %e' % m2p_i)
    logger.debug('total l2p interactions: %e' % l2p_i)

def data_to_gpu(fmm_mat):
    src_tree_nodes = fmm_mat.src_tree.nodes
    obs_tree_nodes = fmm_mat.obs_tree.nodes

    gd = dict()

    surf = np.array(fmm_mat.surf)
    K_name = fmm_mat.cfg.kernel_name
    K = kernels[K_name]

    gd['fmm_mat'] = fmm_mat
    gd['module'] = get_gpu_module(surf, K)
    for a in ['s', 'p']:
        for b in ['s', 'p']:
            name = a + '2' + b
            gd[name] = getattr(gd['module'], name + '_' + K_name)

    gd['obs_pts'] = gpu.to_gpu(fmm_mat.obs_tree.pts, float_type)
    gd['obs_normals'] = gpu.to_gpu(fmm_mat.obs_tree.normals, float_type)
    gd['src_pts'] = gpu.to_gpu(fmm_mat.src_tree.pts, float_type)
    gd['src_normals'] = gpu.to_gpu(fmm_mat.src_tree.normals, float_type)

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
        gd[name + '_n_width'] = gpu.to_gpu(
            np.array([n.bounds.width for n in tree]), float_type
        )

    n_src_levels = len(fmm_mat.m2m)
    gd['u2e_node_n_idx'] = [
        gpu.to_gpu(fmm_mat.u2e[level].src_n_idx, np.int32) for level in range(n_src_levels)
    ]
    gd['u2e_node_depth'] = gpu.to_gpu(np.array([n.depth for n in src_tree_nodes]), np.int32)
    gd['u2e_ops'] = gpu.to_gpu(fmm_mat.u2e_ops, float_type)

    n_obs_levels = len(fmm_mat.l2l)
    gd['d2e_node_n_idx'] = [
        gpu.to_gpu(fmm_mat.d2e[level].src_n_idx, np.int32) for level in range(n_obs_levels)
    ]
    gd['d2e_node_depth'] = gpu.to_gpu(np.array([n.depth for n in obs_tree_nodes]), np.int32)
    gd['d2e_ops'] = gpu.to_gpu(fmm_mat.d2e_ops, float_type)

    return gd

def get_op(op_name, gd):
    return gd[op_name[:3].replace('m', 's').replace('l', 's')]

def get_block_data(op_name, data_name, gd):
    saved_name = op_name + '_' + data_name
    if saved_name not in gd:
        if len(op_name) > 3:
            level = int(op_name[3:])
            gd[saved_name] = gpu.to_gpu(
                getattr(getattr(gd['fmm_mat'], op_name[:3])[level], data_name),
                np.int32
            )
        else:
            gd[saved_name] = gpu.to_gpu(
                getattr(getattr(gd['fmm_mat'], op_name), data_name),
                np.int32
            )
    return gd[saved_name]

def get_data(op_name, name, type, gd):
    if type[0] == 'pts':
        return [
            get_block_data(op_name, name + '_n_start', gd),
            get_block_data(op_name, name + '_n_end', gd),
            gd[type[1] + '_pts'], gd[type[1] + '_normals']
        ]
    else:
        return [
            get_block_data(op_name, name + '_n_idx', gd),
            gd[type[1] + '_n_center'], gd[type[1] + '_n_width'],
            float_type(type[2]),
        ]

def get_n_blocks(op_name, obs_type, gd):
    if obs_type[0] == 'pts':
        return gd[op_name + '_obs_n_start'].shape[0]
    else:
        return gd[op_name + '_obs_n_idx'].shape[0]

def gpu_fmm_op(op_name, out_name, in_name, obs_type, src_type, gd, wait_for):
    op = get_op(op_name, gd)

    obs_data = get_data(op_name, 'obs', obs_type, gd)
    src_data = get_data(op_name, 'src', src_type, gd)

    n_blocks = get_n_blocks(op_name, obs_type, gd)
    if n_blocks > 0:
        return op(
            gpu.gpu_queue,
            (n_blocks * n_workers_per_block,), (n_workers_per_block,),
            gd[out_name].data, gd[in_name].data,
            np.int32(n_blocks), gd['params'].data,
            *[d.data for d in obs_data], *[d.data for d in src_data],
            wait_for = wait_for
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

def gpu_d2e(fmm_mat, gd, level, evs):
    c2e = gd['module'].c2e_kernel
    n_d2e = gd['d2e_node_n_idx'][level].shape[0]
    n_d2e_rows = gd['n_surf_dofs']
    if n_d2e > 0:
        return c2e(
            gpu.gpu_queue,
            (n_d2e * n_workers_per_block,),
            (n_workers_per_block,),
            gd['locals'].data, gd['l_check'].data,
            np.int32(n_d2e), np.int32(n_d2e_rows),
            gd['d2e_node_n_idx'][level].data,
            gd['d2e_node_depth'].data,
            gd['d2e_ops'].data,
            wait_for = evs
        )
    else:
        return None

def gpu_u2e(fmm_mat, gd, level, m2m_ev):
    c2e = gd['module'].c2e_kernel
    n_u2e = gd['u2e_node_n_idx'][level].shape[0]
    n_u2e_rows = gd['n_surf_dofs']
    if n_u2e > 0:
        return c2e(
            gpu.gpu_queue,
            (n_u2e * n_workers_per_block,),
            (n_workers_per_block,),
            gd['multipoles'].data, gd['m_check'].data,
            np.int32(n_u2e), np.int32(n_u2e_rows),
            gd['u2e_node_n_idx'][level].data,
            gd['u2e_node_depth'].data,
            gd['u2e_ops'].data,
            wait_for = [m2m_ev]
        )
    else:
        return None

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
    gd['in'][:] = input_vals.astype(float_type).flatten()
    gd['out'][:] = 0
    gd['m_check'][:] = 0
    gd['multipoles'][:] = 0
    gd['l_check'][:] = 0
    gd['locals'][:] = 0

def eval_ocl(fmm_mat, input_vals, gpu_data = None, should_print_timing = True):
    if gpu_data is None:
        gpu_data = data_to_gpu(fmm_mat)

    prep_data_for_eval(gpu_data, input_vals)

    p2p_ev = gpu_p2p(fmm_mat, gpu_data)
    p2l_ev = gpu_p2l(fmm_mat, gpu_data)
    p2m_ev = gpu_p2m(fmm_mat, gpu_data)

    m2m_evs = []
    u2e_evs = []
    u2e_evs.append(gpu_u2e(fmm_mat, gpu_data, 0, p2m_ev))

    for i in range(1, len(fmm_mat.m2m)):
        m2m_evs.append(gpu_m2m(fmm_mat, gpu_data, i, [u2e_evs[-1]]))
        u2e_evs.append(gpu_u2e(fmm_mat, gpu_data, i, m2m_evs[-1]))

    m2l_ev = gpu_m2l(fmm_mat, gpu_data, [u2e_evs[-1]])
    m2p_ev = gpu_m2p(fmm_mat, gpu_data, [u2e_evs[-1]])

    l2l_evs = []
    d2e_evs = []
    d2e_wait_for = [] if m2l_ev is None else [m2l_ev]
    if p2l_ev is not None:
        d2e_wait_for.append(p2l_ev)
    d2e_evs.append(gpu_d2e(fmm_mat, gpu_data, 0, d2e_wait_for))

    for i in range(1, len(fmm_mat.l2l)):
        l2l_evs.append(gpu_l2l(fmm_mat, gpu_data, i, d2e_evs[-1]))
        d2e_evs.append(gpu_d2e(fmm_mat, gpu_data, i, [l2l_evs[-1]]))

    l2p_ev = gpu_l2p(fmm_mat, gpu_data, d2e_evs[-1])

    if l2p_ev is not None:
        l2p_ev.wait()
    if m2p_ev is not None:
        m2p_ev.wait()
    if p2p_ev is not None:
        p2p_ev.wait()

    retval = gpu_data['out'].get()

    if should_print_timing:
        print_timing(
            p2m_ev, m2m_evs, u2e_evs,
            p2l_ev, m2l_ev, l2l_evs, d2e_evs,
            p2p_ev, m2p_ev, l2p_ev
        )

    return retval

def eval_cpu(fmm_mat, input_vals):
    tensor_dim = fmm_mat.cfg.tensor_dim
    n_out = fmm_mat.obs_tree.pts.shape[0] * tensor_dim
    n_multipoles = fmm_mat.src_tree.n_nodes * len(fmm_mat.surf) * tensor_dim
    n_locals = fmm_mat.obs_tree.n_nodes * len(fmm_mat.surf) * tensor_dim

    out = np.zeros(n_out)
    m_check = np.zeros(n_multipoles)
    multipoles = np.zeros(n_multipoles)
    l_check = np.zeros(n_locals)
    locals = np.zeros(n_locals)

    fmm_mat.p2m_eval(m_check, input_vals)
    fmm_mat.u2e_eval(multipoles, m_check, 0)

    for i in range(1, len(fmm_mat.m2m)):
        fmm_mat.m2m_eval(m_check, multipoles, i)
        fmm_mat.u2e_eval(multipoles, m_check, i)

    fmm_mat.p2l_eval(l_check, input_vals)
    fmm_mat.m2l_eval(l_check, multipoles)
    fmm_mat.d2e_eval(locals, l_check, 0)

    for i in range(1, len(fmm_mat.l2l)):
        fmm_mat.l2l_eval(l_check, locals, i)
        fmm_mat.d2e_eval(locals, l_check, i)

    fmm_mat.l2p_eval(out, locals)

    fmm_mat.p2p_eval(out, input_vals)
    fmm_mat.m2p_eval(out, multipoles)

    return out
