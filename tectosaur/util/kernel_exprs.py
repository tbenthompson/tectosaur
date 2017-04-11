import os
import pickle

from tectosaur.util.sympy_to_cpp import to_cpp

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
    filename = 'kernels.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            kernels = pickle.load(f)
    else:
        import tectosaur.elastic as elastic
        kernels = dict()
        for k_name in ['U','T','A','H']:
            kernels[k_name] = get_kernel(getattr(elastic, k_name))
        with open(filename, 'wb') as f:
            pickle.dump(kernels, f)
    return kernels
