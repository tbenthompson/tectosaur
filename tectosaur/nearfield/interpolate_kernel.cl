<%
import numpy as np
%>
${cluda_preamble}

#define Real ${float_type}

KERNEL
void interpolate(GLOBAL_MEM Real* result,
    int n_interp_pts, int n_eval_pts, int n_output_dims,
    GLOBAL_MEM Real* pts, GLOBAL_MEM Real* wts, 
    GLOBAL_MEM Real* vals, GLOBAL_MEM Real* xhat)
{
    const int i = get_global_id(0);
    const int worker_idx = get_local_id(0);

    LOCAL_MEM Real sh_pts[${block_size}][${n_input_dims}];
    LOCAL_MEM Real sh_wts[${block_size}];
    LOCAL_MEM Real sh_vals[${block_size}];

    Real this_xhat[${n_input_dims}];
    if (i < n_eval_pts) {
        for (int d = 0; d < ${n_input_dims}; d++) {
            this_xhat[d] = xhat[i * ${n_input_dims} + d];
        }
    }

    // TODO: Should the order of the loops be interchanged? I think that would
    // reduce duplicated work. (faster!)
    for (int out_d = 0; out_d < n_output_dims; out_d++) {
        Real denom = 0;
        Real numer = 0;
        for (int chunk_start = 0; chunk_start < n_interp_pts; chunk_start += ${block_size}) {

            int load_idx = chunk_start + worker_idx;
            if (load_idx < n_interp_pts) {
                for (int d = 0; d < ${n_input_dims}; d++) {
                    sh_pts[worker_idx][d] = pts[load_idx * ${n_input_dims} + d];
                }
                sh_wts[worker_idx] = wts[load_idx];
                sh_vals[worker_idx] = vals[load_idx * n_output_dims + out_d];
            }

            LOCAL_BARRIER;

            if (i < n_eval_pts) {
                int chunk_j_max = min(${block_size}, n_interp_pts - chunk_start);
                for (int chunk_j = 0; chunk_j < chunk_j_max; chunk_j++) {
                    Real kern = 1.0;
                    for (int d = 0; d < ${n_input_dims}; d++) {
                        Real dist = this_xhat[d] - sh_pts[chunk_j][d];
                        if (dist == 0) {
                            dist = ${5 * np.finfo(np_float_type).eps};
                        }
                        kern *= dist;
                    }
                    kern = sh_wts[chunk_j] / kern; 
                    denom += kern;
                    numer += kern * sh_vals[chunk_j];
                }
            }

            LOCAL_BARRIER;
        }

        if (i < n_eval_pts) {
            result[i * n_output_dims + out_d] = numer / denom;
        }
    }
}
