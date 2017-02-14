#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define Real float
#define RealSizedInt unsigned int

// Atomic floating point addition for opencl
// from: https://streamcomputing.eu/blog/2016-02-09/atomic-operations-for-floats-in-opencl-improved/
// I was worried this would cause a significant decrease in performance, but
// it doesn't seem to cause any problems
void atomic_fadd(volatile __global Real *addr, Real val) {
    union {
        RealSizedInt u;
        Real f;
    } next, expected, current;
    current.f = *addr;
    do {
        expected.f = current.f;
        next.f = expected.f + val;
        current.u = atomic_cmpxchg(
            (volatile __global RealSizedInt *)addr, expected.u, next.u
        );
    } while(current.u != expected.u);
}

__kernel
void p2p_kernel(__global Real* out, __global Real* in,
        __global int* obs_n_start, __global int* obs_n_end,
        __global int* src_n_start, __global int* src_n_end,
        __global Real* obs_pts, __global Real* obs_normals,
        __global Real* src_pts, __global Real* src_normals)
{
    const int block_idx = get_global_id(0);
    int obs_start = obs_n_start[block_idx];
    int obs_end = obs_n_end[block_idx];
    int src_start = src_n_start[block_idx];
    int src_end = src_n_end[block_idx];

    for (int i = obs_start; i < obs_end; i++) {
        Real xx = obs_pts[i * 3 + 0];
        Real xy = obs_pts[i * 3 + 1];
        Real xz = obs_pts[i * 3 + 2];
        float sum = 0.0;
        for (int j = src_start; j < src_end; j++) {
            Real yx = src_pts[j * 3 + 0];
            Real yy = src_pts[j * 3 + 1];
            Real yz = src_pts[j * 3 + 2];

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            Real kernel_val = 1.0 / sqrt(r2);
            sum += kernel_val * in[j];
        }
        atomic_fadd(&out[i], sum);
    }
}
