${cluda_preamble}

#define Real ${float_type}

<%def name="lookup_interpolation(dims)">
KERNEL
void lookup_interpolation${dims}(GLOBAL_MEM Real* result,
    int n_interp_pts, GLOBAL_MEM Real* table_limits, GLOBAL_MEM Real* table_log_coeffs,
    GLOBAL_MEM Real* interp_pts, GLOBAL_MEM Real* interp_wts, 
    GLOBAL_MEM Real* pts)
{
    const int i = get_global_id(0);
    Real this_pt[${dims}];
    for (int d = 0; d < ${dims}; d++) {
        this_pt[d] = pts[i * ${dims} + d];
    }

    for (int out_d = 0; out_d < 81; out_d++) {
        Real denom = 0;
        Real numer_lim = 0;
        Real numer_log_coeffs = 0;
        for (int j = 0; j < n_interp_pts; j++) {
            Real kern = 1.0;
            for (int d = 0; d < ${dims}; d++) {
                Real dist = this_pt[d] - interp_pts[j * ${dims} + d];
                if (dist == 0) {
                    dist = 1e-16;
                }
                kern *= dist;
            }
            kern = interp_wts[j] / kern; 
            denom += kern;
            numer_lim += kern * table_limits[j * 81 + out_d];
            numer_log_coeffs += kern * table_log_coeffs[j * 81 + out_d];
        }
        result[i * 81 * 2 + out_d * 2] = numer_lim / denom;
        result[i * 81 * 2 + out_d * 2 + 1] = numer_log_coeffs / denom;
    }
}
</%def>

${lookup_interpolation(2)}
${lookup_interpolation(3)}
