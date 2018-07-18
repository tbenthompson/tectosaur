/*cppimport
<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['parallel'] = False
%>
*/
#include <pybind11/pybind11.h>

PYBIND11_MODULE(_check_for_problems, m) {
}
