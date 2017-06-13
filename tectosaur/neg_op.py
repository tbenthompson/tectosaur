class NegOp:
    def __init__(self, op):
        self.op = op
        self.shape = op.shape

    def nearfield_dot(self, v):
        return -self.op.nearfield_dot(v)

    def nearfield_no_correction_dot(self, v):
        return -self.op.nearfield_no_correction_dot(v)

    def dot(self, v):
        return -self.op.dot(v)

    def farfield_dot(self, v):
        return -self.op.farfield_dot(v)
