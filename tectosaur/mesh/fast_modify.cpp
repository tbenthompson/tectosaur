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

py::tuple remove_duplicate_pts(NPArray<double> pts, NPArray<long> tris) {
    double* pts_ptr = as_ptr<double>(pts);
    auto* tris_ptr = as_ptr<long>(tris);

    size_t n_pts = pts.request().shape[0];
    auto n_pts_vals = pts.request().size;
    auto minmax_pt_vals = std::minmax_element(pts_ptr, pts_ptr + n_pts_vals);

    double min_length = (*minmax_pt_vals.second - *minmax_pt_vals.first) * 1e-13;
    double inv_min_length = 1.0 / min_length;

    std::vector<std::array<long,3>> quantized_pts(n_pts);
    for (size_t i = 0; i < n_pts; i++) {
        for (int d = 0; d < 3; d++) {
            quantized_pts[i][d] = lround(pts_ptr[i * 3 + d] * inv_min_length);
        }
    }

    auto compare_quantized_pts = [&] (size_t p1idx, size_t p2idx) {
        auto p1 = quantized_pts[p1idx];
        auto p2 = quantized_pts[p2idx];
        if (p1[0] != p2[0]) {
            return p1[0] > p2[0];
        } else if (p1[1] != p2[1]) {
            return  p1[1] > p2[1];
        } else {
            return p1[2] > p2[2];
        }
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
        for (int d = 0; d < 3; d++) {
            same = same && (quantized_pts[sorted_pt_idxs[i]][d] == quantized_pts[cur_in_idx][d]);
        }
        if (!same) {
            cur_in_idx = sorted_pt_idxs[i];
            cur_out_idx++;
        }
        idx_map[sorted_pt_idxs[i]] = cur_out_idx;
    }

    auto out_pts = make_array<double>({static_cast<size_t>(cur_out_idx + 1), 3});
    auto* out_pts_ptr = as_ptr<double>(out_pts);
    auto out_tris = make_array<long>({static_cast<size_t>(tris.request().shape[0]), 3});
    auto* out_tris_ptr = as_ptr<long>(out_tris);

    for (size_t i = 0; i < n_pts; i++) {
        auto out_pt_idx = idx_map[i]; 
        for (int d = 0; d < 3; d++) {
            out_pts_ptr[out_pt_idx * 3 + d] = pts_ptr[i * 3 + d];
        }
    }

    size_t n_tri_vals = tris.request().size;
    for (size_t i = 0; i < n_tri_vals; i++) {
        out_tris_ptr[i] = idx_map[tris_ptr[i]];
    }

    return py::make_tuple(out_pts, out_tris);
}

PYBIND11_MODULE(fast_modify, m) {
    m.def("remove_duplicate_pts", remove_duplicate_pts);
}
