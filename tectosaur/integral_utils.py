import tectosaur.util.kernel_exprs

def pairs_func_name(singular, k_name):
    singular_label = 'N'
    if singular:
        singular_label = 'S'
    return 'single_pairs' + singular_label + k_name

kernel_names = ['U', 'T', 'A', 'H']
kernels = tectosaur.util.kernel_exprs.get_kernels()

def dn(dim):
    return ['x', 'y', 'z'][dim]

kronecker = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
