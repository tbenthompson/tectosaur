import numpy as np
import tectosaur as tct
import matplotlib.pyplot as plt
import matplotlib.tri
import okada_wrapper

which = 'H'
TCTN = 50
SEP = 0.0
FAR = 4
NEAR = 8
VERTEX = 10
OKADAN = 20
CURVE = 0.0#1.5

if CURVE != 0:
    assert(OKADAN == 0)

corners = [[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]]
src_mesh = tct.make_rect(TCTN, TCTN, corners)
src_mesh[0][:,1] = CURVE * src_mesh[0][:,0] ** 2
def gauss_slip_fnc(x, z):
    r2 = x ** 2 + z ** 2
    R = 0.5
    out = (np.cos(np.sqrt(r2) * np.pi / R) + 1) / 2.0
    out[np.sqrt(r2) > R] = 0.0
    return out
dof_pts = src_mesh[0][src_mesh[1]]
x = dof_pts[:,:,0]
z = dof_pts[:,:,2]
slip = np.zeros((src_mesh[1].shape[0], 3, 3)).astype(np.float32)
slip[:,:,0] = gauss_slip_fnc(x, z)

# plt.triplot(src_mesh[0][:,0], src_mesh[0][:,2], src_mesh[1])
# plt.show()
#
# pt_slip = np.zeros((src_mesh[0].shape[0]))
# pt_slip[src_mesh[1]] = slip[:,:,0]
# levels = np.linspace(0, 1, 11)
# my_cmap = 'OrRd'
# plt.figure(figsize = (8,3.3))
# cntf = plt.tricontourf(src_mesh[0][:,0], src_mesh[0][:,2], src_mesh[1], pt_slip, levels = levels, cmap = my_cmap)
# plt.tricontour(src_mesh[0][:,0], src_mesh[0][:,2], src_mesh[1], pt_slip, levels = levels, linestyles='solid', colors='k', linewidths=0.5)
# plt.colorbar(cntf)
# plt.show()

# xs = np.linspace(-3, 3, 50)
# X, Y = np.meshgrid(xs, xs)
# Z = np.ones_like(X) * sep
# obs_pts = np.array([e.flatten() for e in [X, Y, Z]]).T.copy()
obs_pts = src_mesh[0].copy()
obs_pts[:,1] += SEP

obs_ns = np.zeros(obs_pts.shape)
obs_ns[:,1] = 1.0

if which == 'T':
    K = 'elasticT3'
elif which == 'A':
    K = 'elasticA3'
elif which == 'H':
    K = 'elasticH3'
params = [1.0, 0.25]
op = tct.InteriorOp(obs_pts, obs_ns, src_mesh, K, VERTEX, FAR, NEAR, params, np.float32)
out = op.dot(slip.flatten())

def plot_fnc(triang, f):
    levels = np.linspace(np.min(f) - 1e-12, np.max(f) + 1e-12, 21)
    cntf = plt.tricontourf(triang, f, levels = levels, cmap = 'RdBu')
    plt.tricontour(triang, f, levels = levels, linestyles = 'solid', colors = 'k', linewidths = 0.5)
    plt.colorbar(cntf)

triang = matplotlib.tri.Triangulation(obs_pts[:,0], obs_pts[:,2])
for d in range(3):
    f = out.reshape((-1,3))[:,d]
    plt.subplot(2, 3, d + 1)
    plot_fnc(triang, f)

if OKADAN == 0:
    plt.show(); import sys; sys.exit()

sm, pr = 1.0, 0.25
lam = 2 * sm * pr / (1 - 2 * pr)
alpha = (lam + sm) / (lam + 2 * sm)
X_vals = np.linspace(-1.0, 1.0, OKADAN + 1)
Z_vals = np.linspace(-1.0, 1.0, OKADAN + 1)
def okada_pt(pt):
    disp = np.zeros(3)
    trac = np.zeros(3)
    D = 100000
    pt_copy = pt.copy()
    pt_copy[2] += -D
    for j in range(OKADAN):
        X1 = X_vals[j]
        X2 = X_vals[j+1]
        midX = (X1 + X2) / 2.0
        for k in range(OKADAN):
            Z1 = Z_vals[k]
            Z2 = Z_vals[k+1]
            midZ = (Z1 + Z2) / 2.0
            slip = -gauss_slip_fnc(np.array([midX]), np.array([midZ]))[0]

            [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
                alpha, pt_copy, D, 90.0,
                [X1, X2], [Z1, Z2], [slip, 0.0, 0.0]
            )
            disp += uv
            strain = (grad_uv + grad_uv.T) / 2.0
            strain_trace = sum([strain[d,d] for d in [0,1,2]])
            kronecker = np.array([[1,0,0],[0,1,0],[0,0,1]])
            stress = lam * strain_trace * kronecker + 2 * sm * strain
            trac += stress.dot(obs_ns[0])
            assert(suc == 0)
    if which == 'T':
        return disp
    elif which == 'A':
        assert(False)
    elif which == 'H':
        return trac
Xc = (X_vals[1:] + X_vals[:-1]) / 2.0
Zc = (Z_vals[1:] + Z_vals[:-1]) / 2.0
CX, CZ = np.meshgrid(Xc, Zc)
okada_obs_pts = np.array([CX.flatten(), SEP * np.ones(CX.size), CZ.flatten()]).T.copy()
okada_out = np.array([okada_pt(okada_obs_pts[i,:]) for i in range(okada_obs_pts.shape[0])])

okada_triang = matplotlib.tri.Triangulation(okada_obs_pts[:,0], okada_obs_pts[:,2])
for d in range(3):
    f = okada_out[:,d]
    plt.subplot(2, 3, 3 + d + 1)
    plot_fnc(okada_triang, f)
plt.show()
