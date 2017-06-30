<%
setup_pybind11(cfg)
cfg['dependencies'] = ['../lib/pybind11_nparray.hpp']
%> 

#include <iostream>
#include "../lib/pybind11_nparray.hpp"

namespace py = pybind11;

int positive_mod(int i, int n) {
        return (i % n + n) % n;
}

std::array<int,3> rotate_tri(int clicks) {
    return {positive_mod(clicks, 3), positive_mod(clicks + 1, 3), positive_mod(clicks + 2, 3)};
}

void derotate_adj_mat(NPArray<float> adj_mat, NPArray<long> obs_clicks, NPArray<long> src_clicks) 
{
    auto n_pairs = obs_clicks.request().shape[0];

    auto* obs_clicks_ptr = as_ptr<long>(obs_clicks);
    auto* src_clicks_ptr = as_ptr<long>(src_clicks);
    auto* adj_mat_ptr = as_ptr<float>(adj_mat);

    std::array<float,81> temp{};
    for (size_t i = 0; i < n_pairs; i++) {
        auto obs_derot = rotate_tri(-obs_clicks_ptr[i]);
        auto src_derot = rotate_tri(-src_clicks_ptr[i]);

        for (int b1 = 0; b1 < 3; b1++) {
            for (int b2 = 0; b2 < 3; b2++) {
                for (int d1 = 0; d1 < 3; d1++) {
                    for (int d2 = 0; d2 < 3; d2++) {
                        auto out_idx = b1 * 27 + d1 * 9 + b2 * 3 + d2;
                        auto in_idx = i * 81 + obs_derot[b1] * 27 + d1 * 9 + src_derot[b2] * 3 + d2;
                        temp[out_idx] = adj_mat_ptr[in_idx];
                    }
                }
            }
        }

        for (int j = 0; j < 81; j++) {
            adj_mat_ptr[i * 81 + j] = temp[j];
        }
    }
}

py::tuple make_bsr_matrix(size_t n_rows, size_t n_cols, size_t blockrows, size_t blockcols,
        NPArray<double> in_data, NPArray<long> rows, NPArray<long> cols) 
{
    auto n_row_blocks = n_rows / blockrows;
    auto n_blocks = rows.request().shape[0];
    auto blocksize = blockrows * blockcols;

    auto* rows_ptr = as_ptr<long>(rows);
    auto* cols_ptr = as_ptr<long>(cols);
    auto* in_data_ptr = as_ptr<double>(in_data);

    auto indptr = make_array<int>({n_row_blocks + 1});
    auto indices = make_array<int>({n_blocks});
    auto data = make_array<double>({n_blocks, blockrows, blockcols});

    auto* indptr_ptr = as_ptr<int>(indptr);
    auto* indices_ptr = as_ptr<int>(indices);
    auto* data_ptr = as_ptr<double>(data);

    std::fill(indptr_ptr, indptr_ptr + n_row_blocks, 0.0);

    for (size_t i = 0; i < n_blocks; i++) {
        indptr_ptr[rows_ptr[i]]++;
    }

    for (size_t i = 0, cumsum = 0; i < n_row_blocks; i++) {
        int temp = indptr_ptr[i];
        indptr_ptr[i] = cumsum;
        cumsum += temp;
    }
    indptr_ptr[n_row_blocks] = n_blocks;

    for (size_t n = 0; n < n_blocks; n++) {
        int row = rows_ptr[n];
        int dest = indptr_ptr[row];

        indices_ptr[dest] = cols_ptr[n];
        for (size_t k = 0; k < blocksize; k++) {
            data_ptr[dest * blocksize + k] = in_data_ptr[n * blocksize + k];
        }

        indptr_ptr[row]++;
    }

    for (size_t i = 0, last = 0; i <= n_row_blocks; i++) {
        int temp = indptr_ptr[i];
        indptr_ptr[i] = last;
        last = temp;
    }
    return py::make_tuple(data, indices, indptr);
}


PYBIND11_PLUGIN(fast_assembly) {
    pybind11::module m("fast_assembly", "");
    m.def("make_bsr_matrix", &make_bsr_matrix);
    m.def("derotate_adj_mat", &derotate_adj_mat);
    return m.ptr();
}
