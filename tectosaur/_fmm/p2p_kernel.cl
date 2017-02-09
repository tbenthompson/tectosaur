#define Real float

void atomicAdd_g_f(volatile __global float *addr, float val)
{
   union{
       unsigned int u32;
       float        f32;
   } next, expected, current;
current.f32    = *addr;
   do{
   expected.f32 = current.f32;
       next.f32     = expected.f32 + val;
    current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                           expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
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
        for (int j = src_start; j < src_end; j++) {
            Real yx = src_pts[j * 3 + 0];
            Real yy = src_pts[j * 3 + 1];
            Real yz = src_pts[j * 3 + 2];

            Real Dx = yx - xx;
            Real Dy = yy - xy; 
            Real Dz = yz - xz;
            Real r2 = Dx * Dx + Dy * Dy + Dz * Dz;

            Real kernel_val = 1.0 / sqrt(r2);

            //TODO: This design results in simultaneous writing of out from multiple threads. Avoid doing this in the future!
            volatile __global Real* loc = &out[i];
            atomicAdd_g_f(loc, kernel_val);
        }
    }
}
