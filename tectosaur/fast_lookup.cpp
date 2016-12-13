<%
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-Wall'])
%>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "lib/pybind11_nparray.hpp"

namespace py = pybind11;

double barycentric_evalnd(NPArray<double> pts, NPArray<double> wts, NPArray<double> vals, NPArray<double> xhat) {
    auto pts_buf = pts.request();
    auto* pts_ptr = reinterpret_cast<double*>(pts.request().ptr);
    auto* wts_ptr = reinterpret_cast<double*>(wts.request().ptr);
    auto* vals_ptr = reinterpret_cast<double*>(vals.request().ptr);
    auto* xhat_ptr = reinterpret_cast<double*>(xhat.request().ptr);

    size_t n_pts = pts_buf.shape[0];
    size_t n_dims = pts_buf.shape[1];
    double denom = 0;
    double numer = 0;
    for (size_t i = 0; i < n_pts; i++) {
        double kernel = 1.0;
        for (size_t d = 0; d < n_dims; d++) {
            kernel *= (xhat_ptr[d] - pts_ptr[i * 3 + d]);
        }
        kernel = wts_ptr[i] / kernel; 
        denom += kernel;
        numer += kernel * vals_ptr[i];
    }
    return numer / denom;
}

PYBIND11_PLUGIN(fast_lookup) {
    py::module m("fast_lookup");
    m.def("barycentric_evalnd", barycentric_evalnd);
    return m.ptr();
}
