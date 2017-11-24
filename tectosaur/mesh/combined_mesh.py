import numpy as np
import itertools
from tectosaur.mesh.modify import concat

class CombinedMesh:
    def __init__(self, named_pieces):
        self.names = [np[0] for np in named_pieces]
        pieces = [np[1] for np in named_pieces]
        self.pts, self.tris = concat(*pieces)
        self.sizes = [p[1].shape[0] for p in pieces]
        bounds = [0] + list(itertools.accumulate(self.sizes))
        self.start = bounds[:-1]
        self.past_end = bounds[1:]

    def _get_name_idx(self, name):
        return self.names.index(name)

    def get_start(self, name):
        return self.start[self._get_name_idx(name)]

    def get_past_end(self, name):
        return self.past_end[self._get_name_idx(name)]

    def n_tris(self, name = None):
        if name is None:
            return self.past_end[-1]
        else:
            idx = self._get_name_idx(name)
            return self.past_end[idx] - self.start[idx]

    def n_dofs(self, name = None):
        return self.n_tris(name) * 9

    def get_piece_tri_idxs(self, name):
        idx = self._get_name_idx(name)
        return np.arange(self.start[idx], self.past_end[idx])

    def get_piece_tris(self, name):
        idx = self._get_name_idx(name)
        return self.tris[self.start[idx]:self.past_end[idx]]

    def get_piece_pt_idxs(self, name):
        return np.unique(self.get_piece_tris(name))

    def get_dofs(self, v, name):
        idx = self._get_name_idx(name)
        return v[(9 * self.start[idx]):(9 * self.past_end[idx])]

