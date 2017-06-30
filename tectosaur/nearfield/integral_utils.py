import tectosaur.kernels.kernel_exprs

def pairs_func_name(singular, k_name, check0):
    singular_label = 'N'
    if singular:
        singular_label = 'S'
    check0_label = 'N'
    if check0:
        check0_label = 'Z'
    return 'single_pairs' + singular_label + check0_label + k_name

kernel_names = ['U', 'T', 'A', 'H']
# kernels = tectosaur.util.kernel_exprs.get_kernels()

def dn(dim):
    return ['x', 'y', 'z'][dim]

kronecker = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
