<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
%>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "include/vec_tensor.hpp"
namespace py = pybind11;

PYBIND11_MODULE(_geometry, m) {
    m.def("rotation_matrix", rotation_matrix);
}
