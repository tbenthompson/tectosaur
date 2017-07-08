import os
import pickle

import tectosaur
from tectosaur.kernels.sympy_to_cpp import to_cpp

def get_kernel(kernel_builder):
    out = dict()
    out['expr'] = []
    for i in range(3):
        out['expr'].append([])
        for j in range(3):
            result = kernel_builder(i, j)
            result['expr'] = to_cpp(result['expr'])
            out['symmetric'] = result['symmetric']
            out['expr'][i].append(result['expr'])
    return out

def get_kernels():
    import tectosaur.kernels.elastic as elastic
    kernels = dict()
    for k_name in ['U','T','A','H']:
        kernels[k_name] = get_kernel(getattr(elastic, k_name))
    return kernels
