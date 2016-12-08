#include <stdio.h>
__global__
void compute_integrals(double* result, int n_quad_pts,
    double* quad_pts, double* quad_wts, double* mins, double* maxs,
    double eps, double A)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double minxx = mins[i * 4 + 0];
    double minxy = mins[i * 4 + 1];
    double minyx = mins[i * 4 + 2];
    double minyy = mins[i * 4 + 3];
    double deltaxx = (maxs[i * 4 + 0] - minxx) * 0.5;
    double deltaxy = (maxs[i * 4 + 1] - minxy) * 0.5;
    double deltayx = (maxs[i * 4 + 2] - minyx) * 0.5;
    double deltayy = (maxs[i * 4 + 3] - minyy) * 0.5;

    double sum = 0.0;
    for (int iq = 0; iq < n_quad_pts; iq++) {
        double xx = minxx + deltaxx * (quad_pts[4 * iq] + 1);
        double xy = minxy + deltaxy * (quad_pts[4 * iq + 1] + 1);
        double yx = minyx + deltayx * (quad_pts[4 * iq + 2] + 1);
        double yy = minyy + deltayy * (quad_pts[4 * iq + 3] + 1);
        double w = quad_wts[iq] * deltaxx * deltaxy * deltayx * deltayy;
        double xd = xx - yx;
        double yd = xy - yy;
        double val = eps * xd * xd * powf(xd * xd + yd * yd + eps * eps, -A - 1.0);
        sum += val * w;
    }
    result[i] = sum;
}
