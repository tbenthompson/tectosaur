import os
from IPython import get_ipython
import matplotlib.pyplot as plt

meade03_socket_idx = 0

def configure_meade03(idx):
    global meade03_socket_idx
    meade03_socket_idx = idx
    configure(gpu_idx = idx, fast_plot = True, socket = idx)

def configure(gpu_idx = 0, fast_plot = True, socket = None):
    set_gpu(gpu_idx)
    if fast_plot:
        configure_mpl_fast()
    else:
        configure_mpl_pretty()
    configure_omp(socket)

def configure_omp(socket):
    if socket is None:
        if 'OMP_NUM_THREADS' in os.environ:
            del os.environ['OMP_NUM_THREADS']
    else:
        first_core = socket * 20
        last_core = (socket + 1) * 20
        OMP_PLACES='{' + str(first_core) + ':' + str(last_core) + ':1}'
        os.environ['OMP_NUM_THREADS'] = str(20)
        os.environ['OMP_PLACES']=OMP_PLACES

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
