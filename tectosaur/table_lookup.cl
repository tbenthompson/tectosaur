#pragma OPENCL EXTENSION cl_khr_fp64: enable

#define Real ${float_type}

__kernel
void coincident_lookup_interpolation(__global Real* result,
    int n_interp_pts, __global Real* table_limits, __global Real* table_log_coeffs,
    __global Real* interp_pts, __global Real* interp_wts, 
    __global Real* pts)
{
    const int i = get_global_id(0);

    for (int out_d = 0; out_d < 81; out_d++) {
        Real denom = 0;
        Real numer_lim = 0;
        Real numer_log_coeffs = 0;
        for (int j = 0; j < n_interp_pts; j++) {
            Real kern = 1.0;
            for (int d = 0; d < 3; d++) {
                Real dist = pts[i * 3 + d] - interp_pts[j * 3 + d];
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
        result[i * 81 + out_d * 2] = numer_lim / denom;
        result[i * 81 + out_d * 2 + 1] = numer_log_coeffs / denom;
    }
}
