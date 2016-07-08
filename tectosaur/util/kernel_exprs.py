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
        for k_name in ['U','T','A','H']:
            print(k_name)
            kernels[k_name] = elastic.get_kernel(getattr(elastic, k_name))
        with open(filename, 'wb') as f:
            pickle.dump(kernels, f)
    return kernels
