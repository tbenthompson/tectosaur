<%
from tectosaur.util.build_cfg import fmm_test_cfg
fmm_test_cfg(cfg)
%>

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "include/doctest.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(test_main,m) {
    m.def("run_tests", [] () { main(0, nullptr); });
}
