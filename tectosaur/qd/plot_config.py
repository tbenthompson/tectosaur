import os
from IPython import get_ipython
import matplotlib.pyplot as plt

def configure(gpu_idx = 0, fast_plot = True, n_omp_threads = None):
    set_gpu(gpu_idx)
    if fast_plot:
        configure_mpl_fast()
    else:
        configure_mpl_pretty()
    configure_omp(n_omp_threads)

def configure_omp(n_omp_threads):
    var = 'OMP_NUM_THREADS'
    if n_omp_threads is None:
        if var in os.environ:
            del os.environ[var]
    else:
        os.environ[var] = str(n_omp_threads)


def set_gpu(idx):
    os.environ['CUDA_DEVICE'] = str(idx)

def configure_mpl_fast():
    #TODO: try pdf or svg?
    get_ipython().magic('config InlineBackend.figure_format = \'png\'')
    configure_mpl()

def configure_mpl_pretty():
    get_ipython().magic('config InlineBackend.figure_format = \'retina\'')
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = '\\usepackage{amsmath}'
    configure_mpl()

def configure_mpl():
    plt.style.use('dark_background')
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['xtick.labelsize'] = 16
    plt.rcParams['ytick.labelsize'] = 16
    plt.rcParams['legend.fontsize'] = 20
    plt.rcParams['figure.titlesize'] = 22
    plt.rcParams['savefig.transparent'] = False
