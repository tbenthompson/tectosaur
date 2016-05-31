import matplotlib.pyplot as plt
import numpy as np
import okada_wrapper
from slow_test import slow
from tectosaur.mesh import rect_surface
from tectosaur.quadrature import richardson_quad
from tectosaur.triangle_rules import coincident_quad
from tectosaur.gpu import load_gpu
import pycuda.driver as drv

sm = 30e9
pr = 0.25

def coincident(pts, tris):
    q = richardson_quad([0.1, 0.01], lambda e: coincident_quad(e, 8, 8, 5, 10))
    qx = q[0].astype(np.float32)
    qw = q[1].astype(np.float32)

    result = np.empty((tris.shape[0], 3, 3, 3, 3)).astype(np.float32)

    block = (32, 1, 1)
    grid = (tris.shape[0] // block[0], 1, 1)

    mod = load_gpu('tectosaur/integrals.cu')
    coincidentH = mod.get_function('single_pairsSH')
    print(coincidentH(
        drv.Out(result),
        np.int32(q[0].shape[0]),
        drv.In(qx),
        drv.In(qw),
        drv.In(pts.astype(np.float32)),
        drv.In(tris.astype(np.int32)),
        drv.In(tris.astype(np.int32)),
        np.float32(sm),
        np.float32(pr),
        block = block,
        grid = grid
    ))
    return result

def find_adjacents(tris):
    max_pt_idx = np.max(tris)
    touching_pt = [[] for i in range(max_pt_idx + 1)]
    for i, t in enumerate(tris):
        for d in range(3):
            touching_pt[t[d]].append((i, d))

    vert_adjacents = []
    edge_adjacents = []
    for i, t in enumerate(tris):
        touching_tris = []
        for d in range(3):
            for other_t in touching_pt[t[d]]:
                touching_tris.append((other_t[0], d, other_t[1]))

        already = []
        for other_t in touching_tris:
            if other_t[0] in already or other_t[0] == i:
                continue
            already.append(other_t[0])

            shared_verts = []
            for other_t2 in touching_tris:
                if other_t2[0] != other_t[0]:
                    continue
                shared_verts.append((other_t2[1], other_t2[2]))

            n_shared_verts = len(shared_verts)
            if n_shared_verts == 1:
                vert_adjacents.append((i, other_t[0], shared_verts))
            elif n_shared_verts == 2:
                edge_adjacents.append((i, other_t[0], shared_verts))
            else:
                raise Exception("Duplicate triangles!")

    return vert_adjacents, edge_adjacents

def test_find_adjacents():
    tris = [[0, 1, 2], [2, 1, 3], [0, 4, 5]]
    va, ea = find_adjacents(tris)
    assert(len(va) == 2)
    assert(len(ea) == 2)
    assert(va[0] == (0, 2, [(0, 0)]))
    assert(va[1] == (2, 0, [(0, 0)]))
    assert(ea[0] == (0, 1, [(1, 1), (2, 0)]))
    assert(ea[1] == (1, 0, [(0, 2), (1, 1)]))

def self_integral_operator(pts, tris):
    # coincident
    co_mat = coincident(pts, tris)
    # edge adjacent
    # -- determine from topology
    # -- rotate triangles so that src vtx 0 = obs vtx 1 and src vtx 1 = obs vtx 0
    # vertex adjacent
    # -- determine from topology
    # -- rotate triangles so that src vtx 0 = obs vtx 0
    # nearfield
    # -- sphere tree distances
    # -- use 3 point gauss?
    # subtract farfield correction
    # -- same quadrature as for farfield
    # farfield
    # -- use 2 point gauss? or 3 point tri rule?

def constraints():
    pass
    # continuity from topology is easy, it's already there! one value per point
    # discontinuity across fault shouldn't be too hard...

@slow
def test_okada():
    lam = 2 * sm * pr / (1 - 2 * pr)

    n = 150
    w = 4
    surface = rect_surface(n, n, [[-w, -w, 0], [w, -w, 0], [w, w, 0], [-w, w, 0]])
    top_depth = -0.2
    fault = rect_surface(1, 1, [
        [-0.5, 0, top_depth], [-0.5, 0, top_depth - 1],
        [0.5, 0, top_depth - 1], [0.5, 0, top_depth]]
    )

    alpha = (lam + sm) / (lam + 2 * sm)

    n_pts = surface[0].shape[0]
    obs_pts = surface[0]
    u = np.empty((n_pts, 3))
    for i in range(n_pts):
        pt = surface[0][i, :]
        [suc, uv, grad_uv] = okada_wrapper.dc3dwrapper(
            alpha, pt, 0, 90, [-0.5, 0.5], [top_depth - 1, top_depth], [1, 0, 0]
        )
        u[i, :] = uv

    self_integral_operator(*surface)

    # plt.figure()
    # plt.quiver(obs_pts[:, 0], obs_pts[:, 1], u[:, 0], u[:, 1])
    # plt.figure()
    # plt.streamplot(obs_pts[:, 0].reshape((n,n)), obs_pts[:, 1].reshape((n,n)), u[:, 0].reshape((n,n)), u[:, 1].reshape((n,n)))
    # for d in range(3):
    #     plt.figure()
    #     plt.tripcolor(
    #         obs_pts[:, 0], obs_pts[:, 1], surface[1],
    #         u[:, d], shading='gouraud'
    #     )
    #     plt.colorbar()
    # plt.show()

