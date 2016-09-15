"""
Multidimensional adaptive quadrature
http://ab-initio.mit.edu/wiki/index.php/Cubature <-- using this
http://mint.sbg.ac.at/HIntLib/
http://www.feynarts.de/cuba/

is hcubature or pcubature better? answer: doesn't seem to matter much but probably p

problem! running out of memory for higher accuracy integrals.. how to fix? can i split up
the domain of integration? how do i reduce the accuracy requirements after splitting a domain?
i suspect the answer is buried in the cubature code...
another idea for reducing memory requirements by a factor of 81 is to split up the integrals so that each dimension and basis func is computed separately. or just split by basis func since different basis function pairs should need very different sets of points -- some will be zero in different places than others

should i used vector integrands or is it better to split up the integral into each
individual component so that they don't all need to take as long as the most expensive one
(DONE)test out the richardson process

TODO:
start with 0,0-1,0-0,1 and try all the rotation and scaling stuff.

can i explicitly model and remove the divergence? projects/archive/analytical_bem_integrals/experiments/calc_quad.py has the start of some code to do this based on a modified interpolation basis for the extrapolation process.
make sure i can get 6 digits of accuracy reasonably easily for all kernels

write the table lookup procedure

test the lookup by selecting random legal triangles and comparing the interpolation
"""

import time
from math import cos, pi
import numpy as np

import cppimport
adaptive_integrator = cppimport.imp('adaptive_integrator')

import standardize

"""Chebyshev nodes in the [a,b] interval. Good for interpolation. Avoids Runge phenomenon."""
def chebab(a, b, n):
    out = []
    for i in range(n):
        out.append(0.5 * (a + b) + 0.5 * (b - a) * cos(((2 * i + 1) * pi) / (2 * n)))
    return out

def richardson_limit(step_ratio, values):
    n_steps = len(values)
    last_level = values
    this_level = None

    for m in range(1, n_steps):
        this_level = []
        for i in range(n_steps - m):
            mult = step_ratio ** m
            factor = 1.0 / (mult - 1.0)
            low = last_level[i]
            high = last_level[i + 1]
            moreacc = factor * (mult * high - low)
            this_level.append(moreacc)
        last_level = this_level
    return this_level[0]

# These A, B limits come from ablims.py
minlegalA = 0.0939060848748 - 0.01
minlegalB = 0.233648154379 - 0.01
maxlegalB = 0.865565992417 + 0.01

# for i in range(2, 15):
#     print(richardson_limit(2.0, ((1.0 / 2.0) ** np.arange(0, i)) ** i))
# import sys;sys.exit()

tol = 0.01
eps = 0.01
pr = 0.25
sm = 1.0
K = "U"

tri = [[0.0,0.0,0.0], [1.1,0.0,0.0], [0.0,1.0,0.0]]
standard_tri, labels, R, factor = standardize.standardize(np.array(tri))

start = time.time()
correct = np.array(adaptive_integrator.integrate(
    K, tri, tol, eps, sm, pr
)).reshape((3,3,3,3))[0,:,0,:]
print(time.time() - start)

standard_untransformed = np.array(adaptive_integrator.integrate(
    K, standard_tri.tolist(), tol, eps, sm, pr
)).reshape((3,3,3,3))[0,:,0,:]
standardized = R.T.dot(standard_untransformed.dot(R)) * factor
# standardized = np.zeros((3,3))
# for d1 in range(3):
#     for d2 in range(3):
#         standardized[labels[d1], labels[d2]] = rot_scaled[d1, d2]
print(time.time() - start)
print(standardized)
print(correct)
print(np.trace(standardized))
print(np.trace(correct))
import ipdb; ipdb.set_trace()
