__kernel
void add(__global float* result, __global float* in) {
    int i = get_global_id(0);
    result[i] = in[i] + ${arg};
}
