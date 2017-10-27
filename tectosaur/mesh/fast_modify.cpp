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
py::tuple remove_duplicate_pts(NPArray<double> pts, NPArray<long> els, double threshold) {
    double* pts_ptr = as_ptr<double>(pts);
    auto* els_ptr = as_ptr<long>(els);

    size_t n_pts = pts.request().shape[0];

    double inv_min_length = 1.0 / threshold;

    std::vector<std::array<long,dim>> quantized_pts(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        for (size_t d = 0; d < dim; d++) {
            quantized_pts[i][d] = lround(pts_ptr[i * dim + d] * inv_min_length);
        }
    }

    auto compare_quantized_pts = [&] (size_t p1idx, size_t p2idx) {
        auto p1 = quantized_pts[p1idx];
        auto p2 = quantized_pts[p2idx];
        for (size_t d = 0; d < dim - 1; d++) {
            if (p1[d] != p2[0]) {
                return p1[d] > p2[d];
            }
        }
        return p1[dim - 1] > p2[dim - 1];
    };

    std::vector<size_t> sorted_pt_idxs(quantized_pts.size());
    std::iota(sorted_pt_idxs.begin(), sorted_pt_idxs.end(), 0);
    std::sort(
        sorted_pt_idxs.begin(), sorted_pt_idxs.end(), compare_quantized_pts
    );

    std::map<size_t,size_t> idx_map;
    long cur_in_idx = -1;
    long cur_out_idx = -1;
    for (size_t i = 0; i < n_pts; i++) {
        bool same = cur_in_idx != -1;
        for (size_t d = 0; d < dim; d++) {
            same = same && (quantized_pts[sorted_pt_idxs[i]][d] == quantized_pts[cur_in_idx][d]);
        }
        if (!same) {
            cur_in_idx = sorted_pt_idxs[i];
            cur_out_idx++;
        }
        idx_map[sorted_pt_idxs[i]] = cur_out_idx;
    }

    auto out_pts = make_array<double>({static_cast<size_t>(cur_out_idx + 1), dim});
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
