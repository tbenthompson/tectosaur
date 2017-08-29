<%
from tectosaur.util.build_cfg import fmm_test_cfg
fmm_test_cfg(cfg)
%>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_PLUGIN(test_main) {
    py::module m("test_main");
    m.def("run_tests", [] () { main(0, nullptr); });
    return m.ptr();
}
