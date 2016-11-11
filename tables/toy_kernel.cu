#include <stdio.h>
__global__
void compute_integrals(double* result, int n_quad_pts,
    double* quad_pts, double* quad_wts, double* mins, double* maxs)
{
    const double eps = 0.001;
    const double A = 1.5;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double minx = mins[i * 3 + 0];
    double miny = mins[i * 3 + 1];
    double minz = mins[i * 3 + 2];
    double deltax = (maxs[i * 3 + 0] - minx) * 0.5;
    double deltay = (maxs[i * 3 + 1] - miny) * 0.5;
    double deltaz = (maxs[i * 3 + 2] - minz) * 0.5;

    double sum = 0.0;
    for (int iq = 0; iq < n_quad_pts; iq++) {
        double x = minx + deltax * (quad_pts[iq] + 1);
        double y = miny + deltay * (quad_pts[n_quad_pts * 1 + iq] + 1);
        double z = minz + deltaz * (quad_pts[n_quad_pts * 2 + iq] + 1);
        double w = quad_wts[iq] * deltax * deltay * deltaz;
        double xd = x - z;
        double yd = y - 0.3;
        double val = powf(xd * xd + yd * yd + eps * eps, -A);
        sum += val * w;
    }
    result[i] = sum;
}
