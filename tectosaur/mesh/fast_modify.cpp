<%
from tectosaur.util.build_cfg import setup_module
setup_module(cfg)
cfg['dependencies'].extend([
    '../include/pybind11_nparray.hpp',
])
%>
#include <pybind11/pybind11.h>
#include "include/pybind11_nparray.hpp"
#include <iostream>
#include <map>

namespace py = pybind11;

template <size_t dim>
py::tuple remove_duplicate_pts(NPArray<double> pts, NPArray<long> els, NPArray<long> pairs) {
    auto* pts_ptr = as_ptr<double>(pts);
    auto* els_ptr = as_ptr<long>(els);
    auto* pairs_ptr = as_ptr<long>(pairs);

    size_t n_pts = pts.request().shape[0];

    std::map<size_t,size_t> idx_map;
    long cur_out_idx = 0;
    long next_pair_idx = 0;
    for (size_t i = 0; i < n_pts; i++) {
        if (static_cast<size_t>(pairs_ptr[next_pair_idx * 2 + 1]) == i) {
            idx_map[i] = idx_map[pairs_ptr[next_pair_idx * 2 + 0]];
            while (static_cast<size_t>(pairs_ptr[next_pair_idx * 2 + 1]) == i) {
                next_pair_idx++;
            }
        } else {
            idx_map[i] = cur_out_idx;
            cur_out_idx++;
        }
    }

    auto out_pts = make_array<double>({static_cast<size_t>(cur_out_idx), dim});
    auto* out_pts_ptr = as_ptr<double>(out_pts);
    auto out_els = make_array<long>({static_cast<size_t>(els.request().shape[0]), dim});
    auto* out_els_ptr = as_ptr<long>(out_els);

    for (size_t i = 0; i < n_pts; i++) {
        auto out_pt_idx = idx_map[i]; 
        for (size_t d = 0; d < dim; d++) {
            out_pts_ptr[out_pt_idx * dim + d] = pts_ptr[i * dim + d];
        }
    }

    size_t n_tri_vals = els.request().size;
    for (size_t i = 0; i < n_tri_vals; i++) {
        out_els_ptr[i] = idx_map[els_ptr[i]];
    }

    return py::make_tuple(out_pts, out_els);
}

PYBIND11_MODULE(fast_modify, m) {
    m.def("remove_duplicate_pts3", remove_duplicate_pts<3>);
    m.def("remove_duplicate_pts2", remove_duplicate_pts<2>);
}
