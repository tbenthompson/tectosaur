import numpy as np
from cppimport import cppimport
import matplotlib.pyplot as plt

n = 100
xs = np.linspace(0, 1, n)
X, Y = np.meshgrid(xs, xs)
X = X * 4 - 1.5
Y = Y * 4 - 2

out = np.empty((n,n))
for i in range(n):
    for j in range(n):
        adaptive_integrate = cppimport('adaptive_integrate')
        res = np.array(adaptive_integrate.integrate_interior(
            "T", [X[i,j],Y[i,j],0.2],[0,1,0],
            [[0,0,0],[1,0,0],[0,1,0]],
            1e-2, 1.0, 0.25
        )).reshape((3,3))
        out[i,j] = res.dot((1,0,0))[0]
# plt.imshow(np.log10(np.abs(out)), interpolation = 'none')
plt.imshow(out, interpolation = 'none')
plt.colorbar()
plt.show()
