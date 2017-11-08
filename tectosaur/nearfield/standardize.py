import numpy as np

from tectosaur.util.cpp import imp
cppmod = imp("tectosaur.nearfield._standardize")

locals().update({k:v for k,v in cppmod.__dict__.items() if not k.startswith('__')})

class BadTriangleError(Exception):
    def __init__(self, code):
        super(BadTriangleError, self).__init__("Bad tri: %d" % code)
        self.code = code

def standardize(*args):
    out = cppmod.standardize(*args)
    should_relabel = args[-1];
    if should_relabel and out[0] != 0:
        raise BadTriangleError(out[0])
    return out
