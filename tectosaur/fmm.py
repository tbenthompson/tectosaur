import numpy as np

import cppimport
_fmm = cppimport.imp("tectosaur._fmm._fmm")._fmm._fmm
for k in dir(_fmm):
    locals()[k] = getattr(_fmm, k)
