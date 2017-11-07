import numpy as np
import scipy.spatial
from tectosaur.util.timer import Timer

import tectosaur.util.profile

from cppimport import cppimport
fast_find_nearfield = cppimport('tectosaur.mesh.fast_find_nearfield')

split_adjacent_close = fast_find_nearfield.split_adjacent_close

def get_tri_centroids_rs(pts, tris):
    tri_pts = pts[tris]
    centroid = np.sum(tri_pts, axis = 1) / 3.0
    r = np.sqrt(np.max(
        np.sum((tri_pts - centroid[:,np.newaxis,:]) ** 2, axis = 2),
        axis = 1
    ))
    return centroid, r

def find_close_or_touching(pts, tris, threshold):
    out = fast_find_nearfield.self_get_nearfield(
        *get_tri_centroids_rs(pts, tris), threshold, 50
    )
    return out
