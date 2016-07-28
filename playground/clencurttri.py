from scipy.fftpack import ifft
import numpy as np
from tectosaur.quadrature import *
import sympy

def clencurt(n1):
    """ Computes the Clenshaw Curtis nodes and weights """
    if n1 == 1:
        x = 0
        w = 2
    else:
        n = n1 - 1
        C = np.zeros((n1,2))
        k = 2*(1+np.arange(np.floor(n/2)))
        C[::2,0] = 2/np.hstack((1, 1-k*k))
        C[1,1] = -n
        V = np.vstack((C,np.flipud(C[1:n,:])))
        F = np.real(ifft(V, n=None, axis=0))
        x = F[0:n1,1]
        w = np.hstack((F[0,0],2*F[1:n,0],F[n,0]))
    return x,w

eps = 0.01
correct = 2.0/(eps**2 * np.sqrt(1.0+eps**2.0))
def f(x):
    return 1.0 / ((x ** 2 + y ** 2 + eps ** 2) ** 1.5)


