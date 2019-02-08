import os
import sys
import numpy as np
from tectosaur.mesh.refine import refine_to_size
import tectosaur.util.gpu as gpu
from slip_vectors import get_slip_vectors
from tectosaur_topo import solve_topo

def make_tri_greens_functions(surf, fault, fault_refine_size, basis_idx, i):
    gfs = []
    slip_vecs = []
    subfault_pts = fault[0][fault[1][i,:]]
    subfault_tris = [[0,1,2]]
    subfault_unrefined = [np.array(subfault_pts), np.array(subfault_tris)]
    for s in get_slip_vectors(subfault_pts):
        print(subfault_pts)
        print('slip vector is ' + str(s))
        slip = np.zeros((1,3,3))
        if basis_idx is None:
            slip[0,:,:] = s
        else:
            slip[0,basis_idx,:] = s

        subfault, refined_slip = refine_to_size(
            subfault_unrefined, fault_refine_size,
            [slip[:,:,0], slip[:,:,1], slip[:,:,2]]
        )
        print(
            'Building GFs for fault triangle ' + str(i) +
            ' with ' + str(subfault[1].shape[0]) + ' subtris.'
        )
        full_slip = np.concatenate([s[:,:,np.newaxis] for s in refined_slip], 2)
        result = solve_topo(surf, subfault, full_slip.flatten(), 1.0, 0.25)
        surf_pts = result[0]
        slip_vecs.append(slip[0,:,:])
        gfs.append(result[1])
    return surf_pts, slip_vecs, gfs

def split(a, n):
    # From https://stackoverflow.com/questions/2130016/splitting-a-list-of-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))

def build_greens_functions(surf, fault, fault_refine_size, basis_idx, proc_idx, n_procs):
    print("Building GFs in " + str(proc_idx) + "/" + str(n_procs))

    indices = list(list(split(range(fault[1].shape[0]), n_procs))[proc_idx])
    print(indices)
    results = []
    for i in indices:
        how_many_done = i - indices[0]
        print('Percent progress: ' + str((float(how_many_done) / len(indices)) * 100))
        results.append(
            make_tri_greens_functions(surf, fault, fault_refine_size, basis_idx, i)
        )

    surf_pts = results[0][0]
    slip_vecs = np.array([r[1] for r in results]).reshape((-1, 3, 3))
    gfs = np.array([r[2] for r in results]).reshape((-1, surf_pts.shape[0], 3))
    return surf_pts, slip_vecs, gfs, indices

def combine_gfs(fault, gf_results):
    surf_pts = gf_results[0][0]
    slip_vecs = np.empty((fault[1].shape[0] * 2, 3, 3))
    gfs = np.empty((fault[1].shape[0] * 2, surf_pts.shape[0], 3))
    for i in range(len(gf_results)):
        _, chunk_slip_vecs, chunk_gfs, chunk_indices = gf_results[i]
        gf_indices = np.tile((2 * np.array(chunk_indices))[:,np.newaxis], (1,2))
        gf_indices[:,1] += 1
        gf_indices = gf_indices.flatten()
        slip_vecs[gf_indices] = chunk_slip_vecs
        gfs[gf_indices] = chunk_gfs
    return surf_pts, slip_vecs, gfs

# def build_basis_greens_functions(surf, fault, fault_refine_size):
#     surf_pts, slip_vecs0, gfs0 = build_greens_functions(surf, fault, fault_refine_size, 0)
#     _, slip_vecs1, gfs1 = build_greens_functions(surf, fault, fault_refine_size, 1)
#     _, slip_vecs2, gfs2 = build_greens_functions(surf, fault, fault_refine_size, 2)

