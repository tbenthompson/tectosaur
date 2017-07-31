import numpy as np
import scipy.spatial
from tectosaur.util.timer import Timer

import tectosaur.util.profile

from cppimport import cppimport
fast_find_nearfield = cppimport('tectosaur.mesh.fast_find_nearfield')

split_adjacent_close = fast_find_nearfield.split_adjacent_close

@profile
def find_close_or_touching(pts, tris, threshold):
    tri_pts = pts[tris]
    tri_centroid = np.sum(tri_pts, axis = 1) / 3.0
    tri_r = np.sqrt(np.max(
        np.sum((tri_pts - tri_centroid[:,np.newaxis,:]) ** 2, axis = 2),
        axis = 1
    ))

    out = fast_find_nearfield.get_nearfield(tri_centroid, tri_r, threshold, 50)
    return out
