import os
import pickle

def get_kernels():
    filename = 'kernels.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            kernels = pickle.load(f)
    else:
        import tectosaur.elastic as elastic
        kernels = dict()
        for k_name in kernel_names:
            ks[k[0]] = elastic.get_kernel(getattr(elastic, k_name))
        with open(filename, 'wb') as f:
            pickle.dump(ks, f)
    return kernels
