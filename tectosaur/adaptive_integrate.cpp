<%
setup_pybind11(cfg)
cfg['compiler_args'].extend(['-std=c++14', '-O3', '-Wall'])
cfg['sources'] = ['cubature/hcubature.c', 'cubature/pcubature.c']
%>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "cubature/cubature.h"
#include "lib/pybind11_nparray.hpp"

namespace py = pybind11;

struct IntegrateData {
    py::object f;
};

NPArray<double> call_f(const py::object& f, const NPArray<double>& in) {
    py::object result_py = f(in);
    auto res = result_py.cast<NPArray<double>>();
    return res;
}

int py_integrand(unsigned ndim, size_t npts, const double *x, void *fdata,
      unsigned fdim, double *fval) {
    IntegrateData& d = *reinterpret_cast<IntegrateData*>(fdata);
    auto np_array = make_array({npts, ndim}, const_cast<double*>(x));
    auto res = call_f(d.f, np_array);
    auto res_ptr = res.data();
    for (unsigned i = 0; i < npts * fdim; i++) {
        fval[i] = res_ptr[i]; 
    }
    return 0;
}

std::vector<double> get_center(std::vector<double> mins, std::vector<double> maxs) {
    std::vector<double> center(mins.size());
    for (size_t i = 0; i < mins.size(); i++) {
        center[i] = (mins[i] + maxs[i]) / 2.0;
    }
    return center;
}

PYBIND11_PLUGIN(adaptive_integrate) {
    py::module m("adaptive_integrate");

    m.def("integrate", 
        [] (py::object f, 
            std::vector<double> mins, std::vector<double> maxs,
            double tol) 
        {
            size_t n_in_dims = mins.size();
            assert(n_in_dims == maxs.size());
            
            auto center = get_center(mins, maxs);
            auto center_np = make_array({1,n_in_dims}, center.data());
            auto center_res = call_f(f, center_np);
            int n_out_dims = center_res.shape(1);

            std::vector<double> out(n_out_dims);
            std::vector<double> err(n_out_dims);

            IntegrateData d{f};
            hcubature_v(
                n_out_dims, py_integrand, &d, n_in_dims,
                mins.data(), maxs.data(), 0, 0,
                tol, ERROR_INDIVIDUAL, out.data(), err.data()
            );
            return py::make_tuple(out, err);
        }
    );
    return m.ptr();
}
