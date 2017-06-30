class SumOp:
    def __init__(self, ops):
        self.ops = ops
        self.shape = ops[0].shape

    def nearfield_dot(self, v):
        return sum([op.nearfield_dot(v) for op in self.ops])

    def nearfield_no_correction_dot(self, v):
        return sum([op.nearfield_no_correction_dot(v) for op in self.ops])

    def dot(self, v):
        return sum([op.dot(v) for op in self.ops])

    def farfield_dot(self, v):
        return sum([op.farfield_dot(v) for op in self.ops])
