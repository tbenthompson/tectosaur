__kernel void compute_integrals(__global float* result, int n_quad_pts, __global float* quad_pts, 
        __global float* quad_wts, __global float* mins, __global float* maxs, float eps, float A)
{
    const int i = get_global_id(0) * get_global_size(1) + get_global_id(1);
    float minx = mins[i * 3 + 0];
    float miny = mins[i * 3 + 1];
    float minz = mins[i * 3 + 2];
    float deltax = (maxs[i * 3 + 0] - minx) * 0.5;
    float deltay = (maxs[i * 3 + 1] - miny) * 0.5;
    float deltaz = (maxs[i * 3 + 2] - minz) * 0.5;

    float sum = 0.0;
    for (int iq = 0; iq < n_quad_pts; iq++) {
        float xx = minx + deltax * (quad_pts[3 * iq] + 1);
        float xy = miny + deltay * (quad_pts[3 * iq + 1] + 1);
        float yx = minz + deltaz * (quad_pts[3 * iq + 2] + 1);
        float w = quad_wts[iq] * deltax * deltay * deltaz;
        float xd = xx - yx;
        float yd = xy - 0.3;
        float val = eps * xd * xd * pow(xd * xd + yd * yd + eps * eps, -A - 1.0f);
        sum += val * w;
    }
    result[i] = sum;
}
