<%
import numpy as np
%>
${cluda_preamble}

#define Real ${float_type}

//TODO: Why isn't this a more general interpolation algorithm?

<%def name="lookup_interpolation(dims)">
KERNEL
void lookup_interpolation${dims}(GLOBAL_MEM Real* result,
    int n_eval_pts, int n_interp_pts,
    GLOBAL_MEM Real* table_limits, GLOBAL_MEM Real* table_log_coeffs,
    GLOBAL_MEM Real* interp_pts, GLOBAL_MEM Real* interp_wts, 
    GLOBAL_MEM Real* pts)
{
    const int i = get_global_id(0);
    const int worker_idx = get_local_id(0);

    LOCAL_MEM Real sh_interp_pts[${block_size}][${dims}];
    LOCAL_MEM Real sh_interp_wts[${block_size}];
    LOCAL_MEM Real sh_table_limits[${block_size}];
    LOCAL_MEM Real sh_table_log_coeffs[${block_size}];

    Real this_pt[${dims}];
    if (i < n_eval_pts) {
        for (int d = 0; d < ${dims}; d++) {
            this_pt[d] = pts[i * ${dims} + d];
        }
    }

    for (int out_d = 0; out_d < 81; out_d++) {
        Real denom = 0;
        Real numer_lim = 0;
        Real numer_log_coeffs = 0;
        for (int chunk_start = 0; chunk_start < n_interp_pts; chunk_start += ${block_size}) {

            int load_idx = chunk_start + worker_idx;
            if (load_idx < n_interp_pts) {
                % for d in range(dims):
                sh_interp_pts[worker_idx][${d}] = interp_pts[load_idx * ${dims} + ${d}];
                % endfor
                sh_interp_wts[worker_idx] = interp_wts[load_idx];
                sh_table_limits[worker_idx] = table_limits[load_idx * 81 + out_d];
                sh_table_log_coeffs[worker_idx] = table_log_coeffs[load_idx * 81 + out_d];
            }

            LOCAL_BARRIER;

            if (i < n_eval_pts) {
                int chunk_j_max = min(${block_size}, n_interp_pts - chunk_start);
                for (int chunk_j = 0; chunk_j < chunk_j_max; chunk_j++) {
                    Real kern = 1.0;
                    for (int d = 0; d < ${dims}; d++) {
                        Real dist = this_pt[d] - sh_interp_pts[chunk_j][d];
                        if (dist == 0) {
                            dist = ${5 * np.finfo(np_float_type).eps};
                        }
                        kern *= dist;
                    }
                    kern = sh_interp_wts[chunk_j] / kern; 
                    denom += kern;
                    numer_lim += kern * sh_table_limits[chunk_j];
                    numer_log_coeffs += kern * sh_table_log_coeffs[chunk_j];
                }
            }

            LOCAL_BARRIER;
        }

        if (i < n_eval_pts) {
            result[i * 81 * 2 + out_d * 2] = numer_lim / denom;
            result[i * 81 * 2 + out_d * 2 + 1] = numer_log_coeffs / denom;
        }
    }
}
</%def>

${lookup_interpolation(2)}
${lookup_interpolation(3)}
