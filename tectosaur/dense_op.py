import numpy as np

class DenseOp:
    def __init__(self, mat):
        self.mat = mat
        self.shape = mat.shape

    def nearfield_dot(self, v):
        return self.mat.dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.mat.dot(v)

    def dot(self, v):
        return self.mat.dot(v)

    def farfield_dot(self, v):
        return np.zeros(self.shape[0])
