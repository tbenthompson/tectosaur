from ndadapt import *
import numpy as np

def make_integrator(d, p, f):
    ps = [p] * d
    q_unmapped = tensor_gauss(ps)

    def integrator(mins, maxs):
        print("calc " + str(mins.shape[0]))
        out = []
        for i in range(mins.shape[0]):
            q = map_to(q_unmapped, mins[i,:], maxs[i,:])
            fvs = f(q[0])
            if len(fvs.shape) == 1:
                fvs = fvs[:,np.newaxis]
            assert(fvs.shape[0] == q[1].shape[0])
            out.append(np.sum(fvs * q[1][:,np.newaxis], axis = 0))
        return np.array(out)
    return integrator

eps = 0.01
def f(pts):
    dx = pts[:, 0] - pts[:, 2]
    dy = pts[:, 1] - pts[:, 3]
    denom = (dx ** 2 + dy ** 2 + eps ** 2) ** 2.5
    out = eps * dx ** 2 / denom
    return out

p = 7
res, count = hadapt_nd_iterative(make_integrator(4, p, f), (0,0,0,0), (1,1,1,1), 1e-3)
print(res, count * p ** 2)
