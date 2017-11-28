import numpy as np
import itertools
from tectosaur.mesh.modify import concat
from tectosaur.mesh.refine import refine


class CombinedMesh:
    @staticmethod
    def from_named_pieces(named_pieces):
        names = [np[0] for np in named_pieces]
        pieces = [np[1] for np in named_pieces]
        pts, tris = concat(*pieces)
        sizes = [p[1].shape[0] for p in pieces]
        bounds = [0] + list(itertools.accumulate(sizes))
        return CombinedMesh(names, pts, tris, bounds)

    def __init__(self, names, pts, tris, bounds):
        self.names = names
        self.pts = pts
        self.tris = tris
        self.bounds = bounds
        self.start = bounds[:-1]
        self.end = bounds[1:]

    def refine(self):
        _m = refine((self.pts, self.tris))
        tris_ratio = _m[1].shape[0] // self.tris.shape[0]
        m_refined = CombinedMesh(self.names, _m[0], _m[1], np.array(self.bounds) * tris_ratio)
        return m_refined

    def _get_name_idx(self, name):
        return self.names.index(name)

    def get_start(self, name):
        return self.start[self._get_name_idx(name)]

    def get_end(self, name):
        return self.end[self._get_name_idx(name)]

    def n_tris(self, name = None):
        if name is None:
            return self.end[-1]
        else:
            idx = self._get_name_idx(name)
            return self.end[idx] - self.start[idx]

    def n_dofs(self, name = None):
        return self.n_tris(name) * 9

    def get_tri_idxs(self, name):
        idx = self._get_name_idx(name)
        return np.arange(self.start[idx], self.end[idx])

    def get_tris(self, name):
        idx = self._get_name_idx(name)
        return self.tris[self.start[idx]:self.end[idx]]

    def get_pt_idxs(self, name):
        return np.unique(self.get_tris(name))

    def get_dofs(self, v, name):
        idx = self._get_name_idx(name)
        return v[(9 * self.start[idx]):(9 * self.end[idx])]

