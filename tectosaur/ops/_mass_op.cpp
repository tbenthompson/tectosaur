<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'] += ['../include/pybind11_nparray.hpp']
%> 

#include "../include/pybind11_nparray.hpp"

namespace py = pybind11;

py::tuple build_op(NPArray<double> basis_factors, NPArray<double> jacobians, int tensor_dim) {
    size_t n_tris = jacobians.request().shape[0];
    size_t n_entries = n_tris * 9 * tensor_dim;

    auto basis_factors_ptr = as_ptr<double>(basis_factors);
    auto jacobians_ptr = as_ptr<double>(jacobians);

    auto rows = make_array<size_t>({n_entries});
    auto cols = make_array<size_t>({n_entries});
    auto vals = make_array<double>({n_entries});

    auto rows_ptr = as_ptr<size_t>(rows);
    auto cols_ptr = as_ptr<size_t>(cols);
    auto vals_ptr = as_ptr<double>(vals);

    for (size_t i = 0; i < n_tris; i++) {
        for (size_t b1 = 0; b1 < 3; b1++) {
            for (size_t b2 = 0; b2 < 3; b2++) {
                double entry = jacobians_ptr[i] * basis_factors_ptr[b1 * 3 + b2];
                for (int d = 0; d < tensor_dim; d++) {
                    size_t idx = (i * 9 + b1 * 3 + b2) * tensor_dim + d;
                    rows_ptr[idx] = (i * 3 + b1) * tensor_dim + d;    
                    cols_ptr[idx] = (i * 3 + b2) * tensor_dim + d;    
                    vals_ptr[idx] = entry;
                }
            }
        }
    }
    return py::make_tuple(rows, cols, vals);
}

PYBIND11_MODULE(_mass_op,m) {
    m.def("build_op", &build_op);
}
