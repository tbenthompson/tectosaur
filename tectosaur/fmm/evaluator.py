import numpy as np

import tectosaur.util.gpu as gpu
from tectosaur.util.timer import Timer

import logging
logger = logging.getLogger(__name__)

def convert_op_name_to_kernel_name(op_name):
    return op_name[:3].replace('m', 's').replace('l', 's')

class FMMEvaluator:
    def __init__(self, fmm):
        self.fmm = fmm

        gd = fmm.gpu_data

        self.inp = gpu.empty_gpu(fmm.n_input, fmm.cfg.float_type)
        self.out = gpu.empty_gpu(fmm.n_output, fmm.cfg.float_type)
        self.m_check = gpu.empty_gpu(fmm.n_multipoles, fmm.cfg.float_type)
        self.multipoles = gpu.empty_gpu(fmm.n_multipoles, fmm.cfg.float_type)
        self.c2e_scratch = gpu.empty_gpu(fmm.n_multipoles, fmm.cfg.float_type)
        self.l_check = gpu.empty_gpu(fmm.n_locals, fmm.cfg.float_type)
        self.locals = gpu.empty_gpu(fmm.n_locals, fmm.cfg.float_type)

    def call_kernel(self, name, kernel, *args, **kwargs):
        ev = kernel(*args, **kwargs)
        self.kernel_evs[name] = self.kernel_evs.get(name, []) + [ev]
        return ev

    def to_ev_list(self, maybe_evs):
        return [ev for ev in maybe_evs if ev is not None]

    def gpu_fmm_op(self, op_name, out_arr, in_arr, obs_type, src_type, wait_for):
        gd = self.fmm.gpu_data
        op = self.fmm.gpu_ops[convert_op_name_to_kernel_name(op_name)]

        call_data = [
            gd[op_name + name]
            for name in ['_obs_n_idxs', '_obs_src_starts', '_src_n_idxs']
        ]

        for name, type in [('obs', obs_type), ('src', src_type)]:
            if type[0] == 'pts':
                call_data.extend([
                    gd[type[1] + '_n_start'], gd[type[1] + '_n_end'],
                    gd[type[1] + '_pts'], gd[type[1] + '_tris']
                ])
            else:
                call_data.extend([
                    gd[type[1] + '_n_C'], gd[type[1] + '_n_R'],
                    self.fmm.cfg.float_type(type[2]),
                ])

        n_obs_n = call_data[0].shape[0]
        if n_obs_n > 0:
            out = self.call_kernel(
                op_name, op, out_arr, in_arr,
                np.int32(n_obs_n), gd['params'], *call_data,
                grid = (n_obs_n, 1, 1),
                block = (self.fmm.cfg.n_workers_per_block, 1, 1)
            )
            return out
        else:
            return None

    def gpu_p2p(self):
        return self.gpu_fmm_op(
            'p2p', self.out, self.inp, ('pts', 'obs'), ('pts', 'src'), []
        )

    def gpu_m2p(self, u2e_ev):
        return self.gpu_fmm_op(
            'm2p', self.out, self.multipoles,
            ('pts', 'obs'), ('surf', 'src', self.fmm.cfg.inner_r), u2e_ev
        )

    def gpu_p2m(self):
        return self.gpu_fmm_op(
            'p2m', self.m_check, self.inp,
            ('surf', 'src', self.fmm.cfg.outer_r), ('pts', 'src'), []
        )

    def gpu_m2m(self, level, u2e_ev):
        return self.gpu_fmm_op(
            'm2m' + str(level), self.m_check, self.multipoles,
            ('surf', 'src', self.fmm.cfg.outer_r), ('surf', 'src', self.fmm.cfg.inner_r), []
        )

    def gpu_p2l(self):
        return self.gpu_fmm_op(
            'p2l', self.l_check, self.inp,
            ('surf', 'obs', self.fmm.cfg.inner_r), ('pts', 'src'), []
        )

    def gpu_m2l(self, u2e_ev):
        return self.gpu_fmm_op(
            'm2l', self.l_check, self.multipoles,
            ('surf', 'obs', self.fmm.cfg.inner_r), ('surf', 'src', self.fmm.cfg.inner_r), []
        )

    def gpu_l2l(self, level, d2e_ev):
        return self.gpu_fmm_op(
            'l2l' + str(level), self.l_check, self.locals,
            ('surf', 'obs', self.fmm.cfg.inner_r), ('surf', 'obs', self.fmm.cfg.outer_r), []
        )

    def gpu_l2p(self, d2e_ev):
        return self.gpu_fmm_op(
            'l2p', self.out, self.locals,
            ('pts', 'obs'), ('surf', 'obs', self.fmm.cfg.outer_r), []
        )

    def gpu_c2e(self, level, depth, evs, d_or_u, out_arr, in_arr):
        gd = self.fmm.gpu_data
        op1 = self.fmm.gpu_ops['c2e1']
        op2 = self.fmm.gpu_ops['c2e2']

        name = d_or_u + '2e'
        src_obs = 'obs' if d_or_u == 'd' else 'src'
        n_nodes = gd[name + '_obs_n_idxs'][level].shape[0]
        n_c2e_rows = gd[name + '_UT'].shape[0]

        block_size = 16
        n_node_blocks = int(np.ceil(n_nodes / block_size))
        n_c2e_row_blocks = int(np.ceil(n_c2e_rows / block_size))

        self.call_kernel(
            name, op1,
            self.c2e_scratch, in_arr,
            np.int32(n_nodes), np.int32(n_c2e_rows),
            gd[name + '_obs_n_idxs'][level], gd[name + '_UT'],
            grid = (n_node_blocks, n_c2e_row_blocks, 1),
            block = (block_size, block_size, 1)
        )

        self.call_kernel(
            name, op2,
            out_arr, self.c2e_scratch,
            np.int32(n_nodes), np.int32(n_c2e_rows),
            gd[name + '_obs_n_idxs'][level], gd[src_obs + '_n_R'],
            np.float32(self.fmm.cfg.alpha),
            gd[name + '_V'], gd[name + '_E'],
            grid = (n_node_blocks, n_c2e_row_blocks, 1),
            block = (block_size, block_size, 1)
        )

    def gpu_d2e(self, level, evs):
        return self.gpu_c2e(level, level, evs, 'd', self.locals, self.l_check)

    def gpu_u2e(self, level, evs):
        n_depth = self.fmm.src_tree.max_height - level
        return self.gpu_c2e(level, n_depth, evs, 'u', self.multipoles, self.m_check)

    def prep_data_for_eval(self, input_vals):
        self.inp[:] = input_vals.astype(self.fmm.cfg.float_type).flatten()
        for arr in ['out', 'm_check', 'multipoles', 'l_check', 'locals', 'c2e_scratch']:
            getattr(self, arr).fill(0)

    async def eval(self, tsk_w, input_vals, should_log_timing = False, return_all_intermediates = False):
        self.kernel_evs = dict()

        self.prep_data_for_eval(input_vals)

        p2m_ev = self.gpu_p2m()
        p2p_ev = self.gpu_p2p()
        p2l_ev = self.gpu_p2l()

        m2m_evs = []
        u2e_evs = []
        u2e_evs.append(self.gpu_u2e(0, [p2m_ev]))

        for i in range(1, len(self.fmm.interactions.m2m)):
            m2m_evs.append(self.gpu_m2m(i, [u2e_evs[-1]]))
            u2e_evs.append(self.gpu_u2e(i, [m2m_evs[-1]]))

        m2l_ev = self.gpu_m2l([u2e_evs[-1]])
        m2p_ev = self.gpu_m2p([u2e_evs[-1]])

        l2l_evs = []
        d2e_evs = []
        d2e_wait_for = [] if m2l_ev is None else [m2l_ev]
        if p2l_ev is not None:
            d2e_wait_for.append(p2l_ev)
        d2e_evs.append(self.gpu_d2e(0, d2e_wait_for))

        for i in range(1, len(self.fmm.interactions.l2l)):
            l2l_evs.append(self.gpu_l2l(i, d2e_evs[-1]))
            d2e_evs.append(self.gpu_d2e(i, [l2l_evs[-1]]))

        l2p_ev = self.gpu_l2p(d2e_evs[-1])

        # result = await gpu.get(tsk_w, self.out)
        result = self.out.get()

        if should_log_timing:
            self.log_timing()

        if return_all_intermediates:
            return result, self.m_check.get(), self.multipoles.get(), self.l_check.get(), self.locals.get()
        else:
            return result

    def log_timing(self):
        def get_time(ev):
            if ev is not None:
                return (ev.profile.end - ev.profile.start) * 1e-9
            return 0

        times = dict()
        for k, evs in self.kernel_evs.items():
            name = k[:3]
            times[name] = times.get(name, 0) +sum([get_time(e) for e in evs])

        for k, t in times.items():
            logger.info(k + ' took ' + str(t))
