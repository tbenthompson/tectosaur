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

    def n_total_tris(self):
        return self.past_end[-1]

    def n_dofs(self):
        return self.n_total_tris() * 9

    def get_name_idx(self, name):
        return self.names.index(name)

    def get_piece_tris(self, name):
        idx = self.get_name_idx(name)
        return self.tris[self.start[idx]:self.past_end[idx]]

    def get_piece_pt_idxs(self, name):
        return np.unique(self.get_piece_tris(name))

    def get_start(self, name):
        return self.start[self.get_name_idx(name)]

    def get_past_end(self, name):
        return self.past_end[self.get_name_idx(name)]

    def extract_pts_vals(self, name, soln):
        idx = self.get_name_idx(name)
        dof_vals = soln.reshape((-1, 3, 3))
        all_pt_vals = np.empty_like(self.pts)
        all_pt_vals[self.get_piece_tris(name)] = dof_vals[self.start[idx]:self.past_end[idx]]

        pt_idxs = self.get_piece_pt_idxs(name)
        piece_pts = self.pts[pt_idxs]
        piece_pt_vals = all_pt_vals[pt_idxs]
        return piece_pts, piece_pt_vals
