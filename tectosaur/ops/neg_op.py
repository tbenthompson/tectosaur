class MultOp:
    def __init__(self, op, factor):
        self.op = op
        self.shape = op.shape
        self.factor = factor

    def nearfield_dot(self, v):
        return self.factor * self.op.nearfield_dot(v)

    def nearfield_no_correction_dot(self, v):
        return self.factor * self.op.nearfield_no_correction_dot(v)

    def dot(self, v):
        return self.factor * self.op.dot(v)

    def farfield_dot(self, v):
        return self.factor * self.op.farfield_dot(v)

#TODO: Deprecate
def NegOp(op):
    return MultOp(op, -1)
