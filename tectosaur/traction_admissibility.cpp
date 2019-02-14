<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'].extend([
    '../include/pybind11_nparray.hpp',
])

import sympy
import pickle
symbolic = pickle.load(open('traction_admissibility_sympy.pkl', 'rb'))
%>
#include <pybind11/pybind11.h>
#include "include/pybind11_nparray.hpp"

namespace py = pybind11;

void t2t(double nx, double ny, double nz, double nx2, double ny2, double nz, NPArrayD out) 
{
    as_ptr<double>(out):     
}

PYBIND11_MODULE(fast_modify, m) {
    m.def("t2t", t2t);
}
