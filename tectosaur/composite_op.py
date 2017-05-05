import numpy as np

class CompositeOp:
    def __init__(self, *ops_and_starts):
        self.ops = [el[0] for el in ops_and_starts]
        self.row_start = [el[1] for el in ops_and_starts]
        self.col_start = [el[2] for el in ops_and_starts]
        n_rows = max([el[1] + el[0].shape[0] for el in ops_and_starts])
        n_cols = max([el[2] + el[0].shape[1] for el in ops_and_starts])
        self.shape = (n_rows, n_cols)

    def generic_dot(self, v, dot_name):
        out = np.zeros(self.shape[0])
        for i in range(len(self.ops)):
            op = self.ops[i]
            start_row_idx = self.row_start[i]
            end_row_idx = start_row_idx + op.shape[0]
            start_col_idx = self.col_start[i]
            end_col_idx = start_col_idx + op.shape[1]
            input_v = v[start_col_idx:end_col_idx]
            out[start_row_idx:end_row_idx] += getattr(op, dot_name)(input_v)
        return out

    def nearfield_dot(self, v):
        return self.generic_dot(v, "nearfield_dot")

    def nearfield_no_correction_dot(self, v):
        return self.generic_dot(v, "nearfield_no_correction_dot")

    def dot(self, v):
        return self.generic_dot(v, "dot")

    def farfield_dot(self, v):
        return self.generic_dot(v, "farfield_dot")
