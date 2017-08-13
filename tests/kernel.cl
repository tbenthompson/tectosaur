${cluda_preamble}

KERNEL
void add(GLOBAL_MEM float* result, GLOBAL_MEM float* in) {
    int i = get_global_id(0);
    result[i] = in[i] + ${arg};
}
