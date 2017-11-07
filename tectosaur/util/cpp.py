import sys
import ctypes
import cppimport

def imp(name):
    flags = sys.getdlopenflags()
    sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)
    out = cppimport.cppimport(name)
    sys.setdlopenflags(flags)
    return out
