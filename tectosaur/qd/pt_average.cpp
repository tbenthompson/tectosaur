/*cppimport*/
<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
%> 
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "include/pybind11_nparray.hpp"
#include "include/timing.hpp"

template <typename FloatT, typename IntT>
NPArray<FloatT> pt_average(NPArray<FloatT> pts, NPArray<IntT> tris, NPArray<FloatT> field) {
    (void)pts;
    auto* tris_ptr = as_ptr<IntT>(tris);
    auto* field_ptr = as_ptr<FloatT>(field);

    size_t n_pts = pts.request().shape[0];
    size_t n_tris = tris.request().shape[0];
    std::vector<FloatT> pt_vals(n_pts);
    std::vector<std::vector<int>> pt_touchings(n_pts);
    for (size_t i = 0; i < n_tris; i++) {
        for (size_t d = 0; d < 3; d++) {
            auto pt_idx = tris_ptr[i * 3 + d];
            pt_vals[pt_idx] += field_ptr[i * 3 + d];
            pt_touchings[pt_idx].push_back(i * 3 + d);
        }
    }

    auto out = make_array<FloatT>({static_cast<size_t>(field.request().shape[0])});
    auto* out_ptr = as_ptr<FloatT>(out);
    for (size_t i = 0; i < n_pts; i++) {
        auto n_touching = pt_touchings[i].size();
        pt_vals[i] /= n_touching;
        for (size_t j = 0; j < n_touching; j++) {
            out_ptr[pt_touchings[i][j]] = pt_vals[i];        
        }
    }
    return out;
}

PYBIND11_MODULE(pt_average,m) {
    m.def("pt_averageF", &pt_average<float,int>);
    m.def("pt_averageD", &pt_average<double,long>);
}
