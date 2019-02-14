/*cppimport
<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'].extend([
    'include/pybind11_nparray.hpp',
])

import tectosaur
from tectosaur.util.sympy_to_cpp import to_cpp
import os
import sympy
import pickle
expr_filename = os.path.join(
    tectosaur.source_dir,
    'data',
    'traction_admissibility_sympy.pkl'
)
symbolic = pickle.load(open(expr_filename, 'rb'))
%>
*/

#include <pybind11/pybind11.h>
#include "include/pybind11_nparray.hpp"

namespace py = pybind11;

template <typename T> int sign(T val) {
    return (T(0) < val) - (val < T(0));
}

void traction_projector(double nx, double ny, double nz,
    double nx2, double ny2, double nz2, NPArrayD out) 
{
    double* out_ptr = as_ptr<double>(out);
    % for i in range(6):
        % for j in range(6):
            out_ptr[${i} * 6 + ${j}] = ${to_cpp(symbolic[i,j], dict(no_caching = True))};
        % endfor
    % endfor
}

PYBIND11_MODULE(traction_admissibility, m) {
    m.def("traction_projector", traction_projector);
}
