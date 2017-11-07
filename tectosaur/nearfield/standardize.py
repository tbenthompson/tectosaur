import numpy as np

# THE PLAN!
# Find longest edge
# relabel to put that longest edge as the 0-1 edge
# flip to put the 2 vertex closer to the 0 vertex than the 1 vertex
# translate 0 vertex to origin
# rotate 1 vertex to be at (A, 0, 0) and store rotation
# rotate 2 vertex to be at (B, C, 0) and store rotation
# scale triangle so that 1 vertex is at (1, 0, 0) and store scale factor
# check that triangle internal angles are greater than 20 degrees

from tectosaur.util.cpp import imp
cppmod = imp("tectosaur.nearfield._standardize")

class BadTriangleError(Exception):
    def __init__(self, code):
        super(BadTriangleError, self).__init__("Bad tri: %d" % code)
        self.code = code

def tolist_args(f):
    def wrapper(*args):
        new_args = []
        for a in args:
            new_args.append(a.tolist() if type(a) == np.ndarray else a)
        return f(*new_args)
    return wrapper

get_edge_lens = tolist_args(cppmod.get_edge_lens)
get_longest_edge = tolist_args(cppmod.get_longest_edge)
get_origin_vertex = tolist_args(cppmod.get_origin_vertex)
relabel = tolist_args(cppmod.relabel)
check_bad_tri = tolist_args(cppmod.check_bad_tri)
transform_from_standard = tolist_args(cppmod.transform_from_standard)
translate = tolist_args(cppmod.translate)
rotate1_to_xaxis = tolist_args(cppmod.rotate1_to_xaxis)
rotate2_to_xyplane = tolist_args(cppmod.rotate2_to_xyplane)
full_standardize_rotate = tolist_args(cppmod.full_standardize_rotate)
scale = tolist_args(cppmod.scale)

standardize_unwrapped = tolist_args(cppmod.standardize);
def standardize(*args):
    out = standardize_unwrapped(*args)
    should_relabel = args[-1];
    if should_relabel and out[0] != 0:
        raise BadTriangleError(out[0])
    return out
